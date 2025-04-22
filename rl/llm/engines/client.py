import copy
import json
import re
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import tqdm
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


class ClientEngine(InferenceEngine, ABC):
    BASE_URL: str
    API_KEY_NAME: str

    @abstractmethod
    def generate(self, prompt: ChatInput) -> InferenceOutput:
        pass

    def batch_generate(self, prompts: list[ChatInput]) -> list[InferenceOutput]:
        return thread_map(
            self.generate,
            prompts,
            max_workers=int(rl.utils.io.getenv("RL_MAX_WORKERS", 4)),
        )


_NUM_LOGPROBS = 20


class _OAIClientEngine(ClientEngine, ABC):
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
        if self.llm_config.max_new_tokens is not None:
            completion_kwargs["max_tokens"] = self.llm_config.max_new_tokens
        if EngineFeature.JSON_OUTPUT in self.enabled_features:
            completion_kwargs["response_format"] = {"type": "json_object"}
        if EngineFeature.RETURN_LOGPROBS in self.enabled_features:
            completion_kwargs["logprobs"] = True
            completion_kwargs["top_logprobs"] = _NUM_LOGPROBS

        response = self.client.chat.completions.create(**completion_kwargs)
        choice = response.choices[0]

        output = InferenceOutput(
            prompt=prompt,  # type: ignore
            text=choice.message.content,
            metadata={
                "model": self.llm_config.model_name_or_path,
                "base_url": self.BASE_URL,
            },
        )
        if EngineFeature.RETURN_LOGPROBS in self.enabled_features:
            output.logprobs = [
                {tkn.token: tkn.logprob for tkn in pos.top_logprobs}
                for pos in choice.logprobs.content
            ]
        return output


@register_engine(
    "together",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT,),
)
class TogetherEngine(_OAIClientEngine):
    BASE_URL = "https://api.together.xyz/v1"
    API_KEY_NAME = "TOGETHER_API_KEY"


@register_engine(
    "openai",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT, EngineFeature.RETURN_LOGPROBS),
)
class OpenAIEngine(_OAIClientEngine):
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_NAME = "OPENAI_API_KEY"


@register_engine(
    "groq",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT,),
)
class GroqEngine(_OAIClientEngine):
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_NAME = "GROQ_API_KEY"


@register_engine(
    "gemini",
    required_modules=("google.genai",),
    supported_features={EngineFeature.JSON_OUTPUT},
)
class GeminiEngine(InferenceEngine):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        from google import genai

        self.client = genai.Client(api_key=rl.utils.io.getenv("GEMINI_API_KEY"))
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
        from google.genai import types

        # Convert OpenAI format to Gemini format
        contents = []
        for message in prompt:
            role = message["role"]
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # System messages in Gemini are handled differently
                continue
            elif role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = role

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=message["content"])],
                )
            )

        # Handle system message if present
        generation_config = types.GenerateContentConfig(
            temperature=self.llm_config.temperature,
        )

        if self.llm_config.max_new_tokens is not None:
            generation_config.max_output_tokens = self.llm_config.max_new_tokens

        if EngineFeature.JSON_OUTPUT in self.enabled_features:
            generation_config.response_mime_type = "application/json"

        # Extract system message if present
        system_instruction = None
        if prompt and prompt[0]["role"] == "system":
            system_instruction = prompt[0]["content"]

        if system_instruction:
            generation_config.system_instruction = system_instruction

        # Generate response using the new SDK
        response = self.client.models.generate_content(
            model=self.llm_config.model_name_or_path,
            contents=contents,
            config=generation_config,
        )

        return InferenceOutput(
            prompt=prompt,
            text=response.text,
            metadata={
                "model": self.llm_config.model_name_or_path,
            },
        )

    def batch_generate(self, prompts: list[ChatInput]) -> list[InferenceOutput]:
        return thread_map(
            self.generate,
            prompts,
            max_workers=int(rl.utils.io.getenv("RL_MAX_WORKERS", 10)),
        )


_WARNED_MAX_TOKENS = False


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

        original_prompt = copy.deepcopy(prompt)

        extra_kwargs = {}
        if prompt[0]["role"] == "system":
            extra_kwargs["system"] = prompt[0]["content"]
            prompt = prompt[1:]

        if self.llm_config.max_new_tokens is None:
            global _WARNED_MAX_TOKENS
            if not _WARNED_MAX_TOKENS:
                LOGGER.warning(
                    "Anthropic requires a max_tokens value. Using 4096 by default. "
                    "You can override this by setting max_new_tokens in the LLMConfig."
                )
                _WARNED_MAX_TOKENS = True

        message = self.client.messages.create(
            model=self.llm_config.model_name_or_path,
            messages=prompt,
            max_tokens=self.llm_config.max_new_tokens or 4096,
            **extra_kwargs,
        )
        return InferenceOutput(
            prompt=original_prompt,
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


@register_engine(
    "batch_openai",
    required_modules=("openai",),
    supported_features=(EngineFeature.JSON_OUTPUT, EngineFeature.RETURN_LOGPROBS),
)
class BatchOpenAIEngine(InferenceEngine):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.client = None

    def __enter__(self):
        import openai

        self.client = openai.Client(api_key=rl.utils.io.getenv("OPENAI_API_KEY"))
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
        LOGGER.warning(
            "You called single-item generate() on the BatchOpenAIEngine. I'll allow it, "
            " but you almost certainly didn't mean to do this; just use the regular OpenAIEngine!"
        )
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: list[ChatInput]) -> list[InferenceOutput]:
        from openai.types.chat import ChatCompletion

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".jsonl", delete=False
        ) as temp_file:
            for i, prompt in enumerate(prompts):
                body_kwargs = {
                    "model": self.llm_config.model_name_or_path,
                    "messages": prompt,
                }
                if self.llm_config.max_new_tokens is not None:
                    body_kwargs["max_tokens"] = self.llm_config.max_new_tokens
                if self.llm_config.temperature is not None:
                    body_kwargs["temperature"] = self.llm_config.temperature
                if EngineFeature.JSON_OUTPUT in self.enabled_features:
                    body_kwargs["response_format"] = {"type": "json_object"}
                if EngineFeature.RETURN_LOGPROBS in self.enabled_features:
                    body_kwargs["logprobs"] = True
                    body_kwargs["top_logprobs"] = _NUM_LOGPROBS
                request = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body_kwargs,
                }
                json.dump(request, temp_file)
                temp_file.write("\n")
            temp_file_path = temp_file.name

        # Upload file
        with Path(temp_file_path).open("rb") as file:
            batch_input_file = self.client.files.create(file=file, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        pbar = tqdm.tqdm(total=len(prompts), desc="Polling batch status")
        while batch.status not in ["completed", "failed", "expired"]:
            time.sleep(5)  # Poll every 5 seconds
            batch = self.client.batches.retrieve(batch.id)
            pbar.n = batch.request_counts.completed
            pbar.set_postfix(status=batch.status)
            pbar.refresh()

        pbar.close()

        if batch.status != "completed":
            raise RuntimeError(f"Batch failed with status: {batch.status}")

        output_file = self.client.files.content(batch.output_file_id)
        results = [json.loads(line) for line in output_file.text.strip().split("\n")]

        outputs = []
        for result in results:
            parsed_result = ChatCompletion.model_validate(result["response"]["body"])
            choice = parsed_result.choices[0]
            output = InferenceOutput(
                prompt=prompts[int(result["custom_id"].split("-")[1])],
                text=choice.message.content,
                metadata={
                    "model": self.llm_config.model_name_or_path,
                    "base_url": "https://api.openai.com/v1",
                },
            )
            if EngineFeature.RETURN_LOGPROBS in self.enabled_features:
                output.logprobs = [
                    {tkn.token: tkn.logprob for tkn in pos.top_logprobs}
                    for pos in choice.logprobs.content
                ]
            outputs.append(output)

        Path(temp_file_path).unlink()

        return outputs
