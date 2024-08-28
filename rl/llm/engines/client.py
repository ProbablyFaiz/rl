import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tqdm.contrib.concurrent import thread_map

import rl.utils.io
from rl.llm.engines.core import (
    ChatInput,
    EngineFeature,
    InferenceEngine,
    InferenceInput,
    InferenceOutput,
    LLMConfig,
    apply_chat_template,
    register_engine,
)
from rl.utils import LOGGER

if TYPE_CHECKING:
    import modal
    import openai
    from transformers import PreTrainedTokenizer


_CLIENT_ENGINE_MAX_WORKERS = int(rl.utils.io.getenv("RL_MAX_WORKERS", 4))


class ClientEngine(InferenceEngine, ABC):
    BASE_URL: str
    API_KEY_NAME: str

    @abstractmethod
    def generate(self, prompt: ChatInput) -> InferenceOutput:
        pass

    def batch_generate(self, prompts: list[ChatInput]) -> InferenceOutput:
        return thread_map(
            self.generate, prompts, max_workers=_CLIENT_ENGINE_MAX_WORKERS
        )


class _OpenAIClientEngine(ClientEngine, ABC):
    BASE_URL: str
    API_KEY_NAME: str
    client: "openai.Client"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        import openai

        self.client = openai.Client(
            api_key=rl.utils.io.getenv(self.API_KEY_NAME), base_url=self.BASE_URL
        )
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
        """Given the input prompt, returns the generated text.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text (not including the prompt).
        """
        if not isinstance(prompt, list):
            raise ValueError(
                "ClientEngine requires a list of dicts, in the OpenAI API style."
            )

        completion_kwargs = {
            "model": self.llm_config.model_name_or_path,
            "messages": prompt,
        }
        if EngineFeature.JSON_OUTPUT in self.enabled_features:
            completion_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**completion_kwargs)
        return InferenceOutput(
            prompt=prompt,  # type: ignore
            text=response.choices[0].message.content,
            metadata={
                "model": self.llm_config.model_name_or_path,
                "base_url": self.BASE_URL,
            },
        )


@register_engine(
    "together",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT,),
)
class TogetherEngine(_OpenAIClientEngine):
    BASE_URL = "https://api.together.xyz/v1"
    API_KEY_NAME = "TOGETHER_API_KEY"


@register_engine(
    "openai",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT,),
)
class OpenAIEngine(_OpenAIClientEngine):
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_NAME = "OPENAI_API_KEY"


@register_engine(
    "groq",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT,),
)
class GroqEngine(_OpenAIClientEngine):
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_NAME = "GROQ_API_KEY"


@register_engine(
    "gemini",
    required_modules=("google.generativeai",),
    supported_features={EngineFeature.JSON_OUTPUT},
)
class GeminiEngine(InferenceEngine):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        import google.generativeai as genai

        genai.configure(api_key=rl.utils.io.getenv("GEMINI_API_KEY"))
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        if not isinstance(prompt, list):
            raise ValueError(
                "ClientEngine requires a list of dicts, in the Gemini API style."
            )
        system_message, prev_messages, last_message = self._convert_openai_to_gemini(
            prompt
        )
        # One might reasonably ask, why not initialize the model in __enter__?
        #  Well, I'll tell you: Google's moronic abstraction requires you to
        #  pass the system instruction when *initializing* the model object,
        #  because that makes sense.
        model = genai.GenerativeModel(
            model_name=self.llm_config.model_name_or_path,
            generation_config={
                "temperature": self.llm_config.temperature,
                "max_output_tokens": self.llm_config.max_new_tokens,
                "response_mime_type": "application/json"
                if self.llm_config.json_output
                else "text/plain",
            },
            system_instruction=system_message,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        chat_session = model.start_chat(history=prev_messages)
        # Can't include the last message in the history, because
        #  that would make too much sense!
        response = chat_session.send_message(last_message)

        return InferenceOutput(
            prompt=prompt,
            text=response.text,
            metadata={
                "model": self.llm_config.model_name_or_path,
            },
        )

    def _convert_openai_to_gemini(
        self, prompt: ChatInput
    ) -> tuple[str | None, list, str]:
        """Returns the system instruction, the previous messages, and the last message in the Gemini format."""
        system_prompt = None
        if prompt and prompt[0]["role"] == "system":
            system_prompt = prompt[0]["content"]
            prompt = prompt[1:]
        last_message = prompt[-1]["content"]
        prompt = prompt[:-1]
        return (
            system_prompt,
            [
                {
                    "role": "model" if msg["role"] == "assistant" else msg["role"],
                    "parts": [msg["content"]],
                }
                for msg in prompt
            ],
            last_message,
        )


@register_engine("anthropic", required_modules=("anthropic",))
class AnthropicEngine(ClientEngine):
    BASE_URL = "https://api.anthropic.com/v1"
    API_KEY_NAME = "ANTHROPIC_API_KEY"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=rl.utils.io.getenv(self.API_KEY_NAME))
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
        """Given the input prompt, returns the generated text.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text (not including the prompt).
        """
        if not isinstance(prompt, list):
            raise ValueError(
                "ClientEngine requires a list of dicts, in the OpenAI API style."
            )

        system_prompt = None
        if prompt[0]["role"] == "system":
            system_prompt = prompt[0]["content"]
            prompt = prompt[1:]

        message = self.client.messages.create(
            model=self.llm_config.model_name_or_path,
            system=system_prompt,
            messages=prompt,
            max_tokens=self.llm_config.max_new_tokens,
        )
        return InferenceOutput(
            prompt=prompt,  # type: ignore
            text=message.content[0].text,
            metadata={
                "model": self.llm_config.model_name_or_path,
                "base_url": self.BASE_URL,
            },
        )


@register_engine("modal", required_modules=("modal",))
class ModalEngine(InferenceEngine):
    app_name: str
    modal_call: "modal.Function"
    tokenizer: "PreTrainedTokenizer"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.app_name = self._get_modal_app_name(self.llm_config.model_name_or_path)

    def __enter__(self):
        import modal
        from transformers import AutoTokenizer

        self.modal_call = modal.Function.lookup(self.app_name, "ModalModel.call")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.tokenizer_name_or_path
        )
        LOGGER.warning(
            f"Waiting for Modal app {self.app_name} to be ready. "
            "This may take a few minutes."
        )
        start_time = time.time()
        self.modal_call.remote("generate", "foo", {"max_tokens": 1})
        LOGGER.warning(
            f"Modal app {self.app_name} is ready! Took {time.time() - start_time:.2f}s"
        )
        return self

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: list[InferenceInput]) -> list[InferenceOutput]:
        sampling_params = {
            "max_tokens": self.llm_config.max_new_tokens,
            "temperature": self.llm_config.temperature,
            "frequency_penalty": self.llm_config.frequency_penalty,
            "top_p": 1.0,
        }
        prompts = [
            apply_chat_template(self.tokenizer, prompt)
            if not isinstance(prompt, str)
            else prompt
            for prompt in prompts
        ]
        output_texts = self.modal_call.remote(
            "batch_generate", prompts, sampling_params
        )
        return [
            InferenceOutput(
                prompt=prompt,
                text=output_text,
                metadata={
                    "model_name_or_path": self.llm_config.model_name_or_path,
                },
            )
            for prompt, output_text in zip(prompts, output_texts, strict=True)
        ]

    def _get_modal_app_name(self, model_name: str) -> str:
        return "vllm_" + re.sub(r"[^a-zA-Z0-9-]", "_", model_name)
