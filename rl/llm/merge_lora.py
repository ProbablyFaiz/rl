from pathlib import Path

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import rl.utils.click as click
import rl.utils.io


@click.command()
@click.option(
    "--base_model_id",
    "-b",
    type=str,
    required=True,
    help="Path or slug of the base model to merge with.",
)
@click.option(
    "--lora_model_id",
    "-l",
    type=str,
    required=True,
    help="Path or slug of the LoRA adapter to merge.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=False,
    help="Path to write the merged model to. Will default to the same path as the LoRA model in the merged models dir.",
)
def merge_lora(base_model_id: str, lora_model_id: str, output_path: Path):
    """Merge a LoRA adapter into a PeftModelForCausalLM."""
    output_path = (
        _get_output_path(Path(lora_model_id)) if output_path is None else output_path
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if output_path.exists():
        if not click.confirm(f"{output_path} already exists. Overwrite?"):
            return

    rl.utils.LOGGER.info(f"Loading base model from {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    try:
        # In case there's a modified tokenizer associated with the LoRA training.
        tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
    except Exception:
        rl.utils.LOGGER.warning(
            f"Couldn't find a tokenizer at {lora_model_id}. Using tokenizer from base model {base_model_id}."
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if base_model.vocab_size != len(tokenizer):
        rl.utils.LOGGER.warning(
            f"Vocab size mismatch between model and tokenizer: {base_model.vocab_size} vs {len(tokenizer)}. "
            "Will resize the model to match the tokenizer."
        )
        base_model.resize_token_embeddings(len(tokenizer))

    rl.utils.LOGGER.info(f"Loading LoRA model from {lora_model_id}...")
    lora_model = PeftModelForCausalLM.from_pretrained(
        base_model,
        model_id=lora_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    rl.utils.LOGGER.info("Merging model...")
    merged_model = lora_model.merge_and_unload(progressbar=True)
    # There is some kind of regression in PEFT 0.10.0 that causes it to think the adapter is still
    #  loaded after merging. This is a workaround. # TODO: File an issue on the PEFT repo.
    if hasattr(merged_model, "_hf_peft_config_loaded"):
        merged_model._hf_peft_config_loaded = False
    rl.utils.LOGGER.info(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    rl.utils.LOGGER.info(f"Saving tokenizer to {output_path}...")
    tokenizer.save_pretrained(output_path)


def _get_output_path(input_path: Path) -> Path:
    if input_path.exists():
        # Then this is a filesystem path, not a HuggingFace model slug.
        return rl.utils.io.get_model_path("merged", input_path.name)
    else:
        return rl.utils.io.get_model_path("merged", str(input_path).split("/")[-1])


if __name__ == "__main__":
    merge_lora()
