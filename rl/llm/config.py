from typing import TYPE_CHECKING, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from strenum import StrEnum

import rl.utils.io
from rl.utils import LOGGER

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig, PreTrainedTokenizer


class LLMConfig(BaseModel):
    model_name_or_path: str
    tokenizer_name_or_path: str = ""
    lora_name_or_path: Optional[str] = None

    context_window_tokens: Optional[int] = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    frequency_penalty: float = Field(0.2, description="Experiment with this")

    json_output: bool = False
    return_logprobs: bool = False

    num_gpus: Optional[int] = None
    visible_devices: Optional[str] = None

    engine_name: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            kwargs.update({"model_name_or_path": args[0]})
        elif len(args) > 1:
            raise ValueError(
                "LLMConfig takes at most one positional argument for model_name_or_path."
            )
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def set_defaults_and_overrides(self):
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
        return self

    @property
    def features(self) -> set["EngineFeature"]:
        features = set()
        if self.json_output:
            features.add(EngineFeature.JSON_OUTPUT)
        if self.return_logprobs:
            features.add(EngineFeature.RETURN_LOGPROBS)
        return features


class EngineFeature(StrEnum):
    JSON_OUTPUT = "json_output"
    RETURN_LOGPROBS = "return_logprobs"


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
