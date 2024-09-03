from pydantic import BaseModel, ConfigDict, Field
from strenum import StrEnum

import rl.utils.io
from rl.utils.logger import LOGGER


class LLMConfig(BaseModel):
    model_name_or_path: str
    tokenizer_name_or_path: str = ""
    lora_name_or_path: str | None = None

    context_window_tokens: int | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    frequency_penalty: float = Field(0.2, description="Experiment with this")

    json_output: bool = False
    return_logprobs: bool = False

    num_gpus: int | None = None
    visible_devices: str | None = None

    engine_name: str | None = None

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            kwargs.update({"model_name_or_path": args[0]})
        elif len(args) > 1:
            raise ValueError(
                "LLMConfig takes at most one positional argument for model_name_or_path."
            )
        super().__init__(**kwargs)

    def model_post_init(self, __context):
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
