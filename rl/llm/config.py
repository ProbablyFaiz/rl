from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import torch
from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizer

import rl.utils.io
from rl.utils import LOGGER


@dataclass
class LLMConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str = ""
    lora_name_or_path: str | None = None
    context_window_tokens: int | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    frequency_penalty: float = 0.2  # Experiment with this
    num_gpus: int | None = None
    visible_devices: str | None = None
    json_output: bool = False

    def __post_init__(self):
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path
        if model_override := rl.utils.io.getenv("MODEL_OVERRIDE"):
            LOGGER.warning(
                f"Using model override: {model_override}. This will override the model name or path provided."
            )
            self.model_name_or_path = model_override
        if lora_override := rl.utils.io.getenv("LORA_OVERRIDE"):
            LOGGER.warning(
                f"Using LORA override: {lora_override}. This will override the LORA name or path provided."
            )
            self.lora_name_or_path = lora_override

        if context_window_override := rl.utils.io.getenv("CONTEXT_WINDOW"):
            LOGGER.warning(
                f"Using context window override: {context_window_override}. This will override the context window size provided."
            )
            self.context_window_tokens = int(context_window_override)
        # elif not self.context_window_tokens:
        #     try:
        #         cfg = AutoConfig.from_pretrained(self.model_name_or_path)
        #         if hasattr(cfg, "model_max_length"):
        #             self.context_window_tokens = cfg.model_max_length
        #         elif hasattr(cfg, "max_position_embeddings"):
        #             self.context_window_tokens = cfg.max_position_embeddings
        #         LOGGER.warning(
        #             f"No context window size provided. Guessing the model's max size based on its config: "
        #             f"{self.context_window_tokens}. You can override this by providing the env variable CONTEXT_WINDOW."
        #         )
        #     except OSError:
        #         LOGGER.warning(
        #             "No context window size provided, and it could not be inferred. "
        #             "Setting context_window_tokens to None; this may cause downstream errors."
        #         )


class QuantizationType(str, Enum):
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    FULL = "full"


def get_quantization_config(quant_type: str | None = None) -> BitsAndBytesConfig | None:
    quant_config = rl.utils.io.getenv("QUANT", default="").lower() or quant_type
    if quant_config == QuantizationType.FOUR_BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quant_config == QuantizationType.EIGHT_BIT:
        return BitsAndBytesConfig(load_in_8bit=True)
    if not quant_config or quant_config == QuantizationType.FULL:
        if not quant_config:
            LOGGER.warning(
                "No quantization config set. Defaulting to 16-bit precision. "
                "You can provide the env variable QUANT as '4bit', '8bit', or 'full' to "
                "override this."
            )
        return None
    raise ValueError(
        f"Invalid quantization config set: {quant_config}. Should be one of '4bit', '8bit', or 'full'."
    )


class KShotExample(TypedDict):
    input: str
    output: str


class KShotPrompt(TypedDict):
    instruction: str
    examples: list[KShotExample]


def get_k_shot_prompt(
    k_shot: KShotPrompt, prompt: str, tokenizer: PreTrainedTokenizer
) -> str:
    messages = [
        {
            "role": "system",
            "content": k_shot["instruction"],
        }
    ]
    for ex in k_shot["examples"]:
        messages.append(
            {
                "role": "user",
                "content": ex["input"],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": ex["output"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
