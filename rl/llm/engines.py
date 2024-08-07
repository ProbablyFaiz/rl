import datetime
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap as tw
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Iterator, Union, cast

import google.generativeai as genai
import huggingface_hub
import modal
import modal.runner
import openai
import torch
import tqdm.asyncio
from anthropic import Anthropic
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

import rl.llm.modal_utils
import rl.utils.click as click
import rl.utils.core
import rl.utils.io
from rl.llm.config import LLMConfig
from rl.utils import LOGGER

if (
    torch.cuda.is_available()
    and rl.utils.io.getenv("USE_GPU", "true").lower() != "false"
):
    from vllm import (
        AsyncEngineArgs,
        AsyncLLMEngine,
        EngineArgs,
        LLMEngine,
        RequestOutput,
        SamplingParams,
    )
    from vllm.lora.request import LoRARequest


ChatInput = list[ChatCompletionMessageParam]
InferenceInput = Union[str, ChatInput]


@dataclass(frozen=True)
class InferenceOutput:
    prompt: InferenceInput
    text: str
    metadata: dict[str, Any]  # For now, not used for anything


_WARNED_GEMMA = False


def _apply_chat_template(tokenizer, messages):
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


ENGINES = {}


def _register_engine(name: str):
    def decorator(cls):
        ENGINES[name] = cls
        return cls

    return decorator


class InferenceEngine(ABC):
    NAME: str
    llm_config: LLMConfig

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        """Given the input prompt, returns the generated text.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text (not including the prompt).
        """
        pass

    def batch_generate(self, prompts: list[InferenceInput]) -> list[InferenceOutput]:
        """Given the input prompts, returns the generated texts.

        Args:
            prompts: The input prompts.

        Returns:
            The generated texts (not including the prompts).
        """
        return [self.generate(prompt) for prompt in prompts]


_RESPONSE_CANARY = "### Response template begins now, delete this line. ###"


@_register_engine("manual_edit")
class ManualEditEngine(InferenceEngine):
    _EDITOR = os.environ.get("EDITOR", "vim")

    tokenizer: PreTrainedTokenizer

    def __init__(
        self, llm_config: LLMConfig | None = None, response_template: str = ""
    ):
        super().__init__(llm_config)
        self.response_template = response_template

    def generate(
        self, prompt: InferenceInput, wrap_prompt: bool = True
    ) -> InferenceOutput:
        """Open a temp file, and put the prompt in there. Then open the file in EDITOR,
        and wait for the user to write the response. make any necessary imports in the method."""

        if not isinstance(prompt, str):
            if not hasattr(self, "tokenizer"):
                model = "meta-llama/Meta-Llama-3-8B"
                if self.llm_config:
                    model = self.llm_config.model_name_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            prompt = _apply_chat_template(self.tokenizer, prompt)

        prompt = (
            tw.fill(prompt, width=80, replace_whitespace=False)
            if wrap_prompt
            else prompt
        )

        with tempfile.NamedTemporaryFile(suffix=".md") as tf:
            tf.write(prompt.encode())
            if self.response_template:
                tf.write(f"\n{_RESPONSE_CANARY}\n{self.response_template}".encode())
            tf.flush()
            subprocess.call([self._EDITOR, tf.name])
            tf.seek(0)
            edited_message = tf.read().decode()
        if not edited_message.startswith(prompt):
            raise ValueError(
                "The prompt has been modified. Please do not modify the prompt. "
                "Future attempts to do so will lead to disciplinary action."
            )
        response = edited_message[len(prompt) :].strip()
        if not response:
            raise ValueError("The response is empty or unsaved.")
        if _RESPONSE_CANARY in response:
            raise ValueError("The response template marker has not been deleted.")
        return InferenceOutput(
            prompt=prompt,
            text=response,
            metadata={
                "model": self.NAME,
                "created_at": datetime.datetime.now().isoformat(),
            },
        )


class ClientEngine(InferenceEngine, ABC):
    NAME: str
    BASE_URL: str
    API_KEY_NAME: str
    llm_config: LLMConfig

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    @abstractmethod
    def generate(self, prompt: ChatInput) -> InferenceOutput:
        pass


class OpenAIClientEngine(InferenceEngine, ABC):
    BASE_URL: str = "https://api.openai.com/v1"
    API_KEY_NAME: str = "OPENAI_API_KEY"
    llm_config: LLMConfig
    client: openai.Client

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
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

        response = self.client.chat.completions.create(
            model=self.llm_config.model_name_or_path,
            messages=prompt,
        )
        return InferenceOutput(
            prompt=prompt,  # type: ignore
            text=response.choices[0].message.content,
            metadata={
                "model": self.llm_config.model_name_or_path,
                "base_url": self.BASE_URL,
            },
        )


@_register_engine("together")
class TogetherEngine(OpenAIClientEngine):
    BASE_URL = "https://api.together.xyz/v1"
    API_KEY_NAME = "TOGETHER_API_KEY"


@_register_engine("openai")
class OpenAIEngine(OpenAIClientEngine):
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_NAME = "OPENAI_API_KEY"


@_register_engine("groq")
class GroqEngine(OpenAIClientEngine):
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_NAME = "GROQ_API_KEY"


@_register_engine("gemini")
class GeminiEngine(InferenceEngine):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        genai.configure(api_key=rl.utils.io.getenv("GEMINI_API_KEY"))
        return self

    def generate(self, prompt: ChatInput) -> InferenceOutput:
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


@_register_engine("anthropic")
class AnthropicEngine(ClientEngine):
    BASE_URL = "https://api.anthropic.com/v1"
    API_KEY_NAME = "ANTHROPIC_API_KEY"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
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


@_register_engine("modal")
class ModalEngine(InferenceEngine):
    app_name: str
    modal_call: modal.Function
    tokenizer: PreTrainedTokenizer

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.app_name = self._get_modal_app_name(self.llm_config.model_name_or_path)

    def __enter__(self):
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
            _apply_chat_template(self.tokenizer, prompt)
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
            for prompt, output_text in zip(prompts, output_texts)
        ]

    def _get_modal_app_name(self, model_name: str) -> str:
        return "vllm_" + re.sub(r"[^a-zA-Z0-9-]", "_", model_name)


class AsyncInferenceEngine:
    NAME: str
    llm_config: LLMConfig
    tokenizer: "PreTrainedTokenizer"

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    def __enter__(self):
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
            prompt = _apply_chat_template(self.tokenizer, prompt)
        res = None
        async for res in self.stream(prompt):  # type: ignore
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
                else _apply_chat_template(self.tokenizer, prompt)
            )
            for prompt in prompts
        ]
        return await tqdm.asyncio.tqdm.gather(*tasks)


def _get_vllm_engine(
    llm_config: LLMConfig,
    use_async: bool = False,
) -> tuple[Union["LLMEngine", "AsyncLLMEngine"], dict]:
    if not torch.cuda.is_available():
        raise ValueError(
            "VLLM requires a CUDA-compatible GPU and PyTorch with CUDA support."
        )

    if llm_config.num_gpus > 1:
        if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
            LOGGER.warning(
                "Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn' to avoid issues with "
                "CUDA re-initialization."
            )
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if "ENFORCE_EAGER" not in os.environ:
            LOGGER.warning(
                "Setting ENFORCE_EAGER to 1 to avoid freezing on multi-GPU graph capturing."
            )
            os.environ["ENFORCE_EAGER"] = "1"
    engine_args_kwargs = _get_vllm_kwargs(llm_config)

    engine_cls = AsyncLLMEngine if use_async else LLMEngine
    engine_args_cls = AsyncEngineArgs if use_async else EngineArgs
    engine_args = engine_args_cls(**engine_args_kwargs)  # type: ignore
    if use_async:
        engine_args.disable_log_requests = True
    print(engine_args)
    engine = engine_cls.from_engine_args(engine_args)  # type: ignore

    sampling_params = SamplingParams(
        max_tokens=llm_config.max_new_tokens,
        temperature=llm_config.temperature,
        frequency_penalty=llm_config.frequency_penalty,
        top_p=1.0,
    )

    lora_path = None
    if llm_config.lora_name_or_path:
        lora_path = Path(llm_config.lora_name_or_path)
        if not lora_path.exists():
            lora_path = Path(
                huggingface_hub.snapshot_download(llm_config.lora_name_or_path)
            ).resolve()
        else:
            lora_path = lora_path.resolve()

    generate_kwargs: dict[str, Any] = {
        "params": sampling_params,
    }
    if lora_path is not None:
        generate_kwargs["lora_request"] = LoRARequest(
            lora_name=cast(str, llm_config.lora_name_or_path),
            lora_int_id=1,
            lora_local_path=str(lora_path),
        )

    return engine, generate_kwargs


def _get_vllm_kwargs(llm_config):
    num_gpus = llm_config.num_gpus
    # VLLM only supports a number of GPUs that is a factor of the number of
    #  attention heads (typically 32). To be safe, let's just use the closest
    #  power of 2.
    if num_gpus is None:
        LOGGER.warning(
            "num_gpus is not set. Defaulting to the number of available GPUs. "
            "Set CUDA_VISIBLE_DEVICES to control which/how many GPUs are used."
        )
        num_gpus = int(torch.cuda.device_count())
        num_gpus = 2 ** int(math.log2(num_gpus))
    rl.utils.io.ensure_dotenv_loaded()
    engine_args_kwargs = {
        "model": llm_config.model_name_or_path,
        "tensor_parallel_size": num_gpus,
        "max_model_len": llm_config.context_window_tokens,
        "enforce_eager": rl.utils.io.getenv("ENFORCE_EAGER", "0") == "1",
        "disable_log_stats": True,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
        "enable_lora": llm_config.lora_name_or_path is not None,
        "max_lora_rank": 32,
    }

    # TODO: Re-enable once VLLM supports prefix caching for FP8 KV caches
    # model_config = AutoConfig.from_pretrained(llm_config.model_name_or_path)
    # if (
    #     hasattr(model_config, "quantization_config")
    #     and model_config.quantization_config.get("quant_method", None) == "fp8"
    #     and model_config.quantization_config.get("kv_cache_scheme") is not None
    # ):
    #     LOGGER.warning(
    #         "Model appears to be FP8-quantized with a KV cache scheme set. "
    #         "Enabling VLLM's kv_cache_dtype=fp8 option."
    #     )
    #     engine_args_kwargs["kv_cache_dtype"] = "fp8"
    return engine_args_kwargs


@_register_engine("vllm")
class VLLMEngine(InferenceEngine):
    vllm: "LLMEngine"
    generate_kwargs: dict

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        self.vllm, self.generate_kwargs = _get_vllm_engine(
            self.llm_config, use_async=False
        )
        self.tokenizer = self.vllm.get_tokenizer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.vllm

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: list[InferenceInput]) -> list[InferenceOutput]:
        formatted_prompts: list[str] = [
            (
                prompt
                if isinstance(prompt, str)
                else _apply_chat_template(self.tokenizer, prompt)
            )
            for prompt in prompts
        ]
        vllm_outputs = self._get_vllm_outputs(formatted_prompts)

        inference_outputs = []
        for prompt, output in zip(prompts, vllm_outputs):
            inference_outputs.append(
                InferenceOutput(
                    prompt=prompt,
                    text=output.outputs[0].text,
                    metadata={},
                )
            )
        return inference_outputs

    def _get_vllm_outputs(self, prompts: list[str]):
        vllm_outputs: list[tuple[str, RequestOutput]] = []
        curr_uuid = str(uuid.uuid4())
        for i, prompt in enumerate(prompts):
            self.vllm.add_request(
                request_id=str(f"{curr_uuid}_{i}"),
                inputs=prompt,
                **self.generate_kwargs,
            )

        pbar = (
            tqdm.tqdm(total=len(prompts), desc="Generating", unit="prompt")
            if len(prompts) > 1
            else None
        )
        while self.vllm.has_unfinished_requests() and len(vllm_outputs) < len(prompts):
            outputs = self.vllm.step()
            for req_output in outputs:
                if req_output.finished and req_output.request_id.startswith(curr_uuid):
                    if pbar is not None:
                        pbar.update(1)
                    vllm_outputs.append(
                        (req_output.request_id, cast(RequestOutput, req_output))
                    )

        vllm_outputs.sort(key=lambda x: int(x[0].split("_", 1)[1]))
        return [output[1] for output in vllm_outputs]


@_register_engine("server_vllm")
class WorkerVLLMEngine(InferenceEngine):
    client: openai.OpenAI

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        engine_kwargs = _get_vllm_kwargs(self.llm_config)
        engine_options = []
        for key, value in engine_kwargs.items():
            key = key.replace("_", "-")
            if not isinstance(value, bool):
                engine_options.append(f"--{key}")
                engine_options.append(str(value))
            elif value:
                engine_options.append(f"--{key}")
            else:
                # idk
                pass

        env = {**os.environ}
        if self.llm_config.visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.llm_config.visible_devices

        port = self._find_free_port()
        api_key = "rosebud"
        self.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--port",
                str(port),
                "--api-key",
                api_key,
                *engine_options,
            ],
            env=env,
            close_fds=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.tokenizer_name_or_path
        )

        client_kwargs = {
            "base_url": f"http://localhost:{port}/v1",
            "api_key": api_key,
        }
        self.client = OpenAI(**client_kwargs)

        # Poll the list models endpoint until the server is ready
        for _ in range(100):
            try:
                self.client.models.list()
                break
            except Exception:
                time.sleep(0.5)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Terminating server")
        self.server.terminate()
        self.server.wait()

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        formatted_prompt: str = (
            prompt
            if isinstance(prompt, str)
            else _apply_chat_template(self.tokenizer, prompt)
        )
        res = self.client.completions.create(
            model=self.llm_config.model_name_or_path,
            prompt=formatted_prompt,
            max_tokens=self.llm_config.max_new_tokens,
            temperature=self.llm_config.temperature,
            frequency_penalty=self.llm_config.frequency_penalty,
        )
        return self._wrap_output(prompt, res.choices[0].text)

    def stream(self, prompt: InferenceInput) -> Iterator[InferenceOutput]:
        formatted_prompt: str = (
            prompt
            if isinstance(prompt, str)
            else _apply_chat_template(self.tokenizer, prompt)
        )
        completion = self.client.completions.create(
            model=self.llm_config.model_name_or_path,
            prompt=formatted_prompt,
            max_tokens=self.llm_config.max_new_tokens,
            temperature=self.llm_config.temperature,
            frequency_penalty=self.llm_config.frequency_penalty,
            stream=True,
        )
        curr_text = ""
        for chunk in completion:
            curr_text += chunk.choices[0].text
            yield self._wrap_output(prompt, curr_text)

    def _wrap_output(self, prompt: InferenceInput, output: str) -> InferenceOutput:
        return InferenceOutput(
            prompt=prompt,
            text=output,
            metadata={
                "model_name_or_path": self.llm_config.model_name_or_path,
            },
        )

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]


def get_inference_engine_cls(engine_name: str = "vllm") -> type[InferenceEngine]:
    return ENGINES[engine_name]


def get_inference_engine(llm_config: LLMConfig) -> InferenceEngine:
    assert llm_config.engine_name in ENGINES, (
        f"Engine {llm_config.engine_name} not found. "
        f"Available engines: {', '.join(ENGINES.keys())}"
    )
    engine_cls = get_inference_engine_cls(llm_config.engine_name)
    return engine_cls(llm_config)


# A decorator which injects engine configuration click options, reads them, then constructs the engine
#  and passes it to the decorated function.
def inject_llm_engine(defaults: dict[str, Any] | None):
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
        def wrapper(*args, **kwargs):
            llm_config_kwargs = {
                key: kwargs.pop(key)
                for key in (
                    "model_name_or_path",
                    "tokenizer_name_or_path",
                    "context_window_tokens",
                    "max_new_tokens",
                    "temperature",
                    "num_gpus",
                )
            }
            llm_config = LLMConfig(**llm_config_kwargs)
            engine = get_inference_engine_cls(kwargs.pop("engine_name"))(llm_config)
            return func(*args, engine=engine, **kwargs)

        return wrapper

    return decorator
