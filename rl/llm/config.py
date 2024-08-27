from typing import TYPE_CHECKING, Optional, TypedDict

from strenum import StrEnum

import rl.utils.io

# Need this import here for backwards compatibility
from rl.llm.engines.config import LLMConfig  # noqa: F401
from rl.utils import LOGGER

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig, PreTrainedTokenizer


class QuantizationType(StrEnum):
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    FULL = "full"


def get_quantization_config(
    quant_type: str | None = None,
) -> Optional["BitsAndBytesConfig"]:
    import torch

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
    k_shot: KShotPrompt, prompt: str, tokenizer: "PreTrainedTokenizer"
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
