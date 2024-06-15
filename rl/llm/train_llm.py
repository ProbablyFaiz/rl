# type: ignore
"""A script to train an LLM with LoRA."""

import hashlib
from pathlib import Path

import click
import datasets
import pandas as pd
import peft
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import rl.llm.config
import rl.llm.merge_lora
import rl.utils
import rl.utils.io
from rl.llm.data_collator import DataCollatorForCausalLM

_DEFAULT_BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
_BNB_CONFIG = rl.llm.config.get_quantization_config()

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bias": "none",
    # "task_type": "CAUSAL_LM",
}

_DPO_BETA = 0.1
_VALIDATION_SPLIT = 0.1
_MAX_VALIDATION_SIZE = 500


_DEFAULT_BASE_OUTPUT_DIR = rl.utils.io.get_model_path("lora")
_DEFAULT_MERGED_DIR = rl.utils.io.get_model_path("merged")


@click.command()
@click.option(
    "--base_model_id",
    "-b",
    type=str,
    default=_DEFAULT_BASE_MODEL_ID,
    show_default=True,
)
@click.option(
    "--train_data_path",
    "-t",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the JSONL training data file. Keys are `input` `output`, and `metadata`.",
)
@click.option(
    "--val_data_path",
    "-v",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to the JSONL validation data file. If not specified, will use a random subset of the training data.",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default=None,
    help="The name of the model to save. Must set if --output_dir is not set.",
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=f"Path to the output directory to save the trained model to. If not set, defaults to {_DEFAULT_BASE_OUTPUT_DIR}/<name>.",
)
@click.option(
    "--model_architecture",
    "-a",
    type=click.Choice(["llama", "mistral", "stablelm", "phi"]),
    default="mistral",
    show_default=True,
    help="The model architecture to use. Needed to configure LoRA training properly.",
)
@click.option(
    "--learning_rate",
    "-lr",
    type=float,
    default=3e-4,
    show_default=True,
    help="The learning rate to use.",
)
@click.option(
    "--num_epochs",
    type=float,
    default=3,
    show_default=True,
    help="The number of epochs to train for.",
)
@click.option(
    "--batch_size",
    type=int,
    default=0,
    show_default=True,
    help="The batch size to use. 0 for auto.",
)
@click.option(
    "--eval_steps",
    type=int,
    default=250,
    show_default=True,
    help="The number of steps between evaluations/checkpoints.",
)
@click.option(
    "--skip_confirmation/--no-skip_confirmation",
    "-y",
    default=False,
    show_default=True,
    help="If set, skips the confirmation prompt.",
)
@click.option(
    "--deepspeed_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    show_default=True,
    help="Path to a DeepSpeed config file. If set, uses DeepSpeed to train.",
)
@click.option(
    "--unsloth",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, uses Unsloth to train.",
)
@click.option(
    "--dpo",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, uses Direct Preference Optimization to train. Requires a different format for the training data.",
)
@click.option(
    "--dpo_beta",
    type=float,
    default=_DPO_BETA,
    show_default=True,
    help="The beta parameter for Direct Preference Optimization. Generally between 0.01 and 0.5",
)
@click.option(
    "--merge_after",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, merges the model with the base model after training.",
)
def main(
    base_model_id: str,
    train_data_path: Path,
    val_data_path: Path,
    name: str,
    output_dir: Path,
    model_architecture: str,
    learning_rate: float,
    num_epochs: float,
    batch_size: int,
    eval_steps: int,
    skip_confirmation: bool,
    deepspeed_config: Path,
    unsloth: bool,
    dpo: bool,
    dpo_beta: float,
    merge_after: bool,
):
    rl.utils.io.ensure_dotenv_loaded()

    assert name is not None or output_dir is not None, "Must set --name or --output_dir"
    if output_dir is None:
        output_dir = _get_default_output_dir(name)

    output_exists = output_dir.exists() and any(output_dir.iterdir())
    if not skip_confirmation and output_exists:
        click.confirm(
            f"{output_dir} already exists and is not empty. Y for resume training, N to abort.",
            abort=True,
        )

    full_dataset = get_dataset(train_data_path, val_data_path)

    world_size = int(rl.utils.io.getenv("WORLD_SIZE", 1))
    model, tokenizer = get_model(base_model_id, model_architecture, world_size, unsloth)

    trainer = get_trainer(
        model=model,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        world_size=world_size,
        output_dir=output_dir,
        tokenizer=tokenizer,
        full_dataset=full_dataset,
        dpo=dpo,
        dpo_beta=dpo_beta,
        deepspeed_config=deepspeed_config,
    )

    if trainer.accelerator.is_main_process:
        log_initial_run_info(base_model_id, name, train_data_path, val_data_path)

    trainer.train(resume_from_checkpoint=True if output_exists else None)
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        save_model(trainer, tokenizer, output_dir)

    del trainer, tokenizer, model
    if merge_after:
        merged_dir = _DEFAULT_MERGED_DIR / (name or output_dir.name)
        rl.llm.merge_lora.merge_lora.callback(
            base_model_id=base_model_id,
            lora_model_id=output_dir,
            output_path=merged_dir,
        )


def _get_default_output_dir(name: str) -> Path:
    return _DEFAULT_BASE_OUTPUT_DIR / name


def get_dataset(train_data_path: Path, val_data_path: Path) -> datasets.Dataset:
    df = pd.read_json(train_data_path, lines=True)
    # Removing the metadata because it causes weird problems when loading the dataset.
    df = df.drop(columns=["metadata"])
    dataset = datasets.Dataset.from_pandas(df)
    if val_data_path:
        val_df = pd.read_json(val_data_path, lines=True)
        val_df = val_df.head(_MAX_VALIDATION_SIZE)
        val_df = val_df.drop(columns=["metadata"])
        val_dataset = datasets.Dataset.from_pandas(val_df)
        dataset = datasets.DatasetDict({"train": dataset, "test": val_dataset})
    else:
        val_size = min(_MAX_VALIDATION_SIZE, int(len(dataset) * _VALIDATION_SPLIT))
        dataset = dataset.train_test_split(test_size=val_size, seed=42)
    return dataset


def get_tokenizer(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    return tokenizer


def get_model(
    base_model_id: str, model_architecture: str, world_size: int, unsloth: bool
) -> tuple[peft.PeftModel, AutoTokenizer]:
    if model_architecture in ("llama", "stablelm", "mistral"):
        LORA_CONFIG["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif model_architecture == "phi":
        LORA_CONFIG["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
        ]
    else:
        # TODO: Just leaving as a placeholder for if/when we add other architectures.
        pass

    if unsloth:
        assert model_architecture in (
            "mistral",
            "llama",
        ), "Unsloth only supports Mistral and LLaMA"
        assert (
            world_size == 1
        ), "TODO: Figure out how to do multi-GPU training with Unsloth"
        assert (
            _BNB_CONFIG is None or not _BNB_CONFIG.load_in_8bit
        ), "Unsloth only supports 4-bit and no quantization"
        return _get_unsloth_model(base_model_id)

    if world_size > 1:
        assert rl.utils.io.getenv("LOCAL_RANK") is not None, "LOCAL_RANK must be set"
    device_map = (
        {"": int(rl.utils.io.getenv("LOCAL_RANK"))} if world_size > 1 else "auto"
    )
    rl.utils.LOGGER.info(f"Device map for this process: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=_BNB_CONFIG,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, peft.LoraConfig(**LORA_CONFIG))  # type: ignore
    model.config.use_cache = False
    torch.compile(model)
    return model, get_tokenizer(base_model_id)


def _get_unsloth_model(base_model_id: str) -> tuple:
    from unsloth import FastLanguageModel

    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=_BNB_CONFIG.load_in_4bit if _BNB_CONFIG else False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **LORA_CONFIG,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    return model, tokenizer


def get_trainer(
    *,
    model,
    learning_rate,
    num_epochs,
    batch_size,
    eval_steps,
    world_size,
    output_dir,
    tokenizer,
    full_dataset,
    dpo=False,
    dpo_beta=None,
    deepspeed_config=None,
):
    gradient_accumulation_steps = 64 // (batch_size or 4)
    if world_size > 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if world_size == 1 and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        # Training hyperparameters
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=15,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Deepspeed
        deepspeed=deepspeed_config,
        # Save configuration
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=6,
        load_best_model_at_end=True,
        output_dir=str(output_dir),
        save_safetensors=False,  # Safetensors breaks because of: https://huggingface.co/docs/safetensors/torch_shared_tensors
        # Logging
        report_to="wandb",  # type: ignore
        logging_steps=2,
        # Other
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )
    if batch_size:
        training_args.per_device_train_batch_size = batch_size
        training_args.per_device_eval_batch_size = batch_size
    else:
        rl.utils.LOGGER.info("Auto-detecting batch size...")
        training_args.auto_find_batch_size = True

    if dpo:
        assert dpo_beta is not None, "dpo_beta must be set if dpo is True"
        assert all(
            col in full_dataset["train"].column_names
            for col in ("prompt", "chosen", "rejected")
        ), "DPO training requires 'prompt', 'chosen', and 'rejected' columns in the training data. Did you pass the right file?"
        from trl import DPOTrainer

        trainer_class = DPOTrainer
        extra_trainer_kwargs = {"beta": dpo_beta, "tokenizer": tokenizer}
    else:
        trainer_class = Trainer
        extra_trainer_kwargs = {
            "data_collator": DataCollatorForCausalLM(
                tokenizer,
                train_on_input=False,
                predict_with_generate=False,
                input_max_len=4096,
                output_max_len=2048,
            ),
        }

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=full_dataset["train"],
        eval_dataset=full_dataset["test"],
        **extra_trainer_kwargs,
    )
    return trainer


def log_initial_run_info(base_model_id, run_name, train_data_path, val_data_path):
    train_data_md5_hash = hashlib.md5(train_data_path.read_bytes()).hexdigest()
    val_data_md5_hash = (
        hashlib.md5(val_data_path.read_bytes()).hexdigest()
        if val_data_path is not None
        else None
    )
    run_config = {
        "base_model_id": base_model_id,
        "lora_config": LORA_CONFIG,
        "bnb_config": (
            _BNB_CONFIG.to_dict() if _BNB_CONFIG is not None else "not quantized"
        ),
        "train_data_path": str(train_data_path),
        "train_data_md5": train_data_md5_hash,
        "val_data_path": str(val_data_path) if val_data_path is not None else None,
        "val_data_md5": val_data_md5_hash,
    }
    wandb.init(
        project=rl.utils.io.getenv("WANDB_PROJECT"),
        entity=rl.utils.io.getenv("WANDB_ENTITY"),
        name=run_name,
        config=run_config,
    )


def confirm_run(output_dir):
    if any(output_dir.iterdir()):
        click.confirm(
            f"{output_dir} already exists and is not empty. Y for resume training, N to abort.",
            abort=True,
        )


def save_model(trainer, tokenizer, output_dir):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
