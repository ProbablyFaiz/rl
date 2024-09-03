import importlib
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import tqdm.asyncio
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import rl.utils.click as click
import rl.utils.io
from rl.llm.engines.config import EngineFeature, LLMConfig
from rl.utils import LOGGER

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class ChatMessage(TypedDict):
    role: str
    content: str


ChatInput = list[ChatMessage]
InferenceInput = str | ChatInput


class InferenceOutput(BaseModel):
    prompt: InferenceInput
    text: str

    logprobs: list[dict[int, float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceEngineError(Exception):
    pass


class MissingEngineNameError(InferenceEngineError):
    pass


class EngineNotSupportedError(InferenceEngineError):
    pass


class EngineNotEnteredContextError(InferenceEngineError):
    pass


class FeatureNotSupportedError(InferenceEngineError):
    pass


class NotSupportedSentinel:
    reason: str

    def __init__(self, reason: str):
        self.reason = reason

    def __call__(self, *_, **__):
        raise EngineNotSupportedError(self.reason)


_WARNED_GEMMA = False


def apply_chat_template(tokenizer: "PreTrainedTokenizer", messages: ChatInput) -> str:
    if "gemma-2" in tokenizer.name_or_path:
        if isinstance(messages, list) and messages and messages[0]["role"] == "system":
            messages[0]["role"] = "user"
            messages.insert(1, {"role": "assistant", "content": ""})
            global _WARNED_GEMMA
            if not _WARNED_GEMMA:
                LOGGER.warning(
                    "You passed a system message to a Gemma-2 tokenizer, and it "
                    "doesn't support those. I'll try to fix it by changing the "
                    "role to 'user' and adding an empty assistant message, but "
                    "there's no guarantee this will work."
                )
            _WARNED_GEMMA = True
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def register_engine(
    name: str,
    required_modules: tuple[str, ...] = (),
    supported_features: tuple[EngineFeature, ...] = (),
):
    def init_decorator(cls):
        original_init = cls.__init__
        original_enter = cls.__enter__
        original_exit = cls.__exit__
        original_generate = cls.generate
        original_batch_generate = cls.batch_generate

        def wrapped_init(self, llm_config: LLMConfig, *args, **kwargs):
            if not llm_config.features.issubset(cls.SUPPORTED_FEATURES):
                raise FeatureNotSupportedError(
                    f"Engine {name} does not support the following features: "
                    f"{llm_config.features - cls.SUPPORTED_FEATURES}"
                )
            self.enabled_features = llm_config.features.copy()
            self.entered_context = False
            original_init(self, llm_config, *args, **kwargs)

        def wrapped_enter(self):
            self.entered_context = True
            return original_enter(self)

        def wrapped_exit(self, exc_type, exc_value, traceback):
            if self.entered_context:
                self.entered_context = False
                original_exit(self, exc_type, exc_value, traceback)

        def wrapped_generate(self, prompt: InferenceInput) -> InferenceOutput:
            if not self.entered_context:
                raise EngineNotEnteredContextError(
                    "You must enter the context of the engine by calling "
                    "`with engine as e:` before you can call `e.generate()`."
                )
            return original_generate(self, prompt)

        def wrapped_batch_generate(
            self, prompts: list[InferenceInput]
        ) -> list[InferenceOutput]:
            if not self.entered_context:
                raise EngineNotEnteredContextError(
                    "You must enter the context of the engine by calling "
                    "`with engine as e:` before you can call `e.batch_generate()`."
                )
            return original_batch_generate(self, prompts)

        cls.__init__ = wrapped_init
        cls.__enter__ = wrapped_enter
        cls.__exit__ = wrapped_exit
        cls.generate = wrapped_generate
        cls.batch_generate = wrapped_batch_generate
        return cls

    def decorator(cls):
        cls.NAME = name
        cls.REQUIRED_MODULES = set(required_modules)
        cls.SUPPORTED_FEATURES = set(supported_features)

        new_cls = init_decorator(cls)

        missing_modules = []
        for module in required_modules:
            if not _import_if_available(module):
                missing_modules.append(module)
        if missing_modules:
            new_cls = NotSupportedSentinel(
                f"Engine {name} requires the following modules which "
                f"could not be imported: {missing_modules}"
            )

        ENGINES[name] = new_cls
        return new_cls

    return decorator


def _import_if_available(module_name: str) -> bool:
    try:
        # Forgive me, lord
        globals()[module_name] = importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False


ENGINES = {}


class InferenceEngine:
    NAME: str
    REQUIRED_MODULES: set[str]
    SUPPORTED_FEATURES: set[EngineFeature]
    _REGISTERED: bool = False

    llm_config: LLMConfig
    enabled_features: set[EngineFeature]

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @staticmethod
    def from_config(llm_config: LLMConfig) -> "InferenceEngine":
        if llm_config.engine_name is None:
            raise MissingEngineNameError(
                "When initializing an inference engine via the base class InferenceEngine, "
                "you must pass an llm_config with an engine_name set. Available engine names: "
                f"{', '.join(ENGINES.keys())}"
            )
        return get_inference_engine(llm_config)

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        """Given the input prompt, returns the generated text.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text (not including the prompt).
        """
        raise NotImplementedError

    def batch_generate(self, prompts: list[InferenceInput]) -> list[InferenceOutput]:
        """Given the input prompts, returns the generated texts.

        Args:
            prompts: The input prompts.

        Returns:
            The generated texts (not including the prompts).
        """
        return [
            self.generate(prompt) for prompt in tqdm.tqdm(prompts, desc="Generating")
        ]


class AsyncInferenceEngine:
    NAME: str
    llm_config: LLMConfig
    tokenizer: "PreTrainedTokenizer"

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    def __enter__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.tokenizer_name_or_path
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[InferenceOutput, bool]:
        """Given the input prompt, returns an async generator that yields InferenceOutput objects
        and can be sent an `abort` boolean via .asend(True) to stop the generation.

        Args:
            prompt: The input prompt.

        Returns:
            An async generator that yields InferenceOutput objects.
        """
        pass

    async def generate(self, prompt: InferenceInput) -> InferenceOutput:
        """Given the input prompt, returns the generated text.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text (not including the prompt).
        """
        if not isinstance(prompt, str):
            prompt = apply_chat_template(self.tokenizer, prompt)
        res = None
        async for res in self.stream(prompt):  # noqa: B007
            pass
        return res

    @abstractmethod
    async def batch_generate(
        self, prompts: list[InferenceInput]
    ) -> list[InferenceOutput]:
        """Given the input prompts, returns the generated texts.

        Args:
            prompts: The input prompts.

        Returns:
            The generated texts (not including the prompts).
        """
        tasks = [
            self.generate(
                prompt
                if isinstance(prompt, str)
                else apply_chat_template(self.tokenizer, prompt)
            )
            for prompt in prompts
        ]
        return await tqdm.asyncio.tqdm.gather(*tasks)


def get_inference_engine_cls(engine_name: str = "vllm") -> type[InferenceEngine]:
    return ENGINES[engine_name]


def get_inference_engine(
    llm_config: LLMConfig, engine_name: str | None = None
) -> InferenceEngine:
    engine_name = engine_name or llm_config.engine_name

    assert engine_name in ENGINES, (
        f"Engine {llm_config.engine_name} not found. "
        f"Available engines: {', '.join(ENGINES.keys())}"
    )
    engine_cls = get_inference_engine_cls(engine_name)
    return engine_cls(llm_config)


def inject_llm_engine(defaults: dict[str, Any] | None):
    """A decorator which injects engine configuration click options, reads them,
    then constructs the engine and passes it to the decorated function."""

    def decorator(func):
        @click.option(
            "--engine-name",
            "-e",
            default=defaults.get("engine_name", "vllm"),
            show_default=True,
            help="The name of the engine to use.",
        )
        @click.option(
            "--model-name-or-path",
            "-m",
            required=defaults.get("model_name_or_path") is None,
            default=defaults.get("model_name_or_path"),
            show_default=True,
            help="The model name or path to use for the engine.",
        )
        @click.option(
            "--tokenizer-name-or-path",
            default=defaults.get("tokenizer_name_or_path")
            or defaults.get("model_name_or_path"),
            show_default=defaults.get("tokenizer_name_or_path"),
            help="The tokenizer name or path to use for the engine, if different from model.",
        )
        @click.option(
            "--context-window",
            "context_window_tokens",
            type=int,
            default=defaults.get("context_window_tokens", 8192),
            help="The number of tokens in the context window.",
        )
        @click.option(
            "--max-new-tokens",
            type=int,
            default=defaults.get("max_new_tokens", 1024),
            help="The maximum number of new tokens to generate.",
        )
        @click.option(
            "--temperature",
            type=float,
            default=defaults.get("temperature", 0.3),
            help="The temperature to use for sampling.",
        )
        @click.option(
            "--num-gpus",
            type=int,
            default=defaults.get("num_gpus", 1),
            help="The number of GPUs to use for inference.",
        )
        @click.option(
            "--json-output",
            is_flag=True,
            help="Whether to output the response in JSON format.",
        )
        @click.option(
            "--return-logprobs",
            is_flag=True,
            help="Whether to return logprobs in the response.",
        )
        def wrapper(*args, **kwargs):
            llm_config_kwargs = {
                key: kwargs.pop(key)
                for key in (
                    "engine_name",
                    "model_name_or_path",
                    "tokenizer_name_or_path",
                    "context_window_tokens",
                    "max_new_tokens",
                    "temperature",
                    "num_gpus",
                    "json_output",
                    "return_logprobs",
                )
            }
            llm_config = LLMConfig(**llm_config_kwargs)
            engine = InferenceEngine.from_config(llm_config)
            return func(*args, engine=engine, **kwargs)

        return wrapper

    return decorator
