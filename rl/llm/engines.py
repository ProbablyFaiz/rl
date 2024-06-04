import math
import os
import socket
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Iterator, Union, cast
import textwrap as tw

import click
import huggingface_hub
import openai
import torch
import tqdm.asyncio
from openai import OpenAI
from anthropic import Anthropic
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoTokenizer, PreTrainedTokenizer

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

InferenceInput = Union[str, ChatCompletionMessageParam]


@dataclass(frozen=True)
class InferenceOutput:
    prompt: InferenceInput
    text: str
    metadata: dict[str, Any]  # For now, not used for anything


def _apply_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


class InferenceEngine(ABC):
    NAME: str
    llm_config: LLMConfig

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    def __enter__(self):
        pass

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


class ManualEditEngine(InferenceEngine):
    NAME = "human input"

    def __init__(
        self, llm_config: LLMConfig | None = None, response_template: str = ""
    ):
        super().__init__(llm_config)
        self.response_template = response_template

    def generate(self, prompt: InferenceInput, wrap_prompt: bool=True) -> InferenceOutput:
        """Open a temp file, and put the prompt in there. Then open the file in EDITOR, and wait for the user to write the response. make any necessary imports in the mehtod"""
        import sys, tempfile, os
        import datetime
        from subprocess import call

        EDITOR = os.environ.get("EDITOR", "vim")

        if not isinstance(prompt, str):
            if not hasattr(self, "tokenizer"):
                model = "meta-llama/Meta-Llama-3-8B"
                if self.llm_config:
                    model = self.llm_config.model_name_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            prompt = _apply_chat_template(self.tokenizer, prompt)

        prompt = tw.fill(prompt, width=80, replace_whitespace=False) if wrap_prompt else prompt

        with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
            tf.write(prompt.encode())
            if self.response_template:
                tf.write(
                    f"\n### response template begins now, delete this line ###\n{self.response_template}".encode()
                )
            tf.flush()
            call([EDITOR, tf.name])
            tf.seek(0)
            edited_message = tf.read().decode()
        if not edited_message.startswith(prompt):
            raise ValueError(
                "The prompt has been modified. Please do not modify the prompt."
            )
        response = edited_message[len(prompt) :].strip()
        if not response:
            raise ValueError("The response is empty or unsaved.")
        if "### response template begins now, delete this line ###" in response:
            raise ValueError("The response template marker has not been deleted.")
        return InferenceOutput(
            prompt=prompt,
            text=response,
            metadata={
                "model": "manual edit",
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
    def generate(
        self, prompt: openai.types.chat.ChatCompletionMessageParam
    ) -> InferenceOutput:
        pass


class OpenAIClientEngine(InferenceEngine, ABC):
    NAME: str
    BASE_URL: str
    API_KEY_NAME: str
    llm_config: LLMConfig

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.client = openai.OpenAI(
            api_key=rl.utils.io.getenv(self.API_KEY_NAME), base_url=self.BASE_URL
        )

    def generate(
        self, prompt: openai.types.chat.ChatCompletionMessageParam
    ) -> InferenceOutput:
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


class TogetherEngine(OpenAIClientEngine):
    NAME = "together"
    BASE_URL = "https://api.together.xyz/v1"
    API_KEY_NAME = "TOGETHER_API_KEY"


class OpenAIEngine(OpenAIClientEngine):
    NAME = "openai"
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_NAME = "OPENAI_API_KEY"


class AnthropicEngine(ClientEngine):
    NAME = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1"
    API_KEY_NAME = "ANTHROPIC_API_KEY"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.client = Anthropic(api_key=rl.utils.io.getenv(self.API_KEY_NAME))

    def generate(
        self,
        prompt: openai.types.chat.ChatCompletionMessageParam,
        max_tokens: int = 1024,
    ) -> InferenceOutput:
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
            messages=prompt,
            system=system_prompt,
            max_tokens=max_tokens,
        )
        return InferenceOutput(
            prompt=prompt,  # type: ignore
            text=message.content[0].text,
            metadata={
                "model": self.llm_config.model_name_or_path,
                "base_url": self.BASE_URL,
            },
        )


class AsyncInferenceEngine:
    NAME: str
    llm_config: LLMConfig
    tokenizer: "PreTrainedTokenizer"

    def __init__(self, llm_config: LLMConfig):
        rl.utils.io.ensure_dotenv_loaded()
        self.llm_config = llm_config

    async def __aenter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.tokenizer_name_or_path
        )
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
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

    engine_args_kwargs = _get_vllm_kwargs(llm_config)
    engine_cls = AsyncLLMEngine if use_async else LLMEngine
    engine_args_cls = AsyncEngineArgs if use_async else EngineArgs
    engine_args = engine_args_cls(**engine_args_kwargs)  # type: ignore
    if use_async:
        engine_args.disable_log_requests = True
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
        "sampling_params": sampling_params,
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
        "enforce_eager": rl.utils.io.getenv("ENFORCE_EAGER", "").lower() == "true",
        "disable_log_stats": True,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "enable_lora": llm_config.lora_name_or_path is not None,
        "max_lora_rank": 32,
    }
    return engine_args_kwargs


class VLLMEngine(InferenceEngine):
    NAME = "vllm"

    vllm: "LLMEngine"
    generate_kwargs: dict

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        self.vllm, self.generate_kwargs = _get_vllm_engine(
            self.llm_config, use_async=False
        )
        self.tokenizer = self.vllm.get_tokenizer()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.vllm

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        if not isinstance(prompt, str):
            prompt = self.tokenizer.apply_chat_template(prompt)
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
                prompt=prompt,
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


class WorkerVLLMEngine(InferenceEngine):
    NAME = "server_vllm"

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


class AsyncVLLMEngine(AsyncInferenceEngine):
    NAME = "vllm_async"

    vllm: "AsyncLLMEngine"
    generate_kwargs: dict

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    async def __aenter__(self):
        self.vllm, self.generate_kwargs = _get_vllm_engine(
            self.llm_config, use_async=True
        )
        self.tokenizer = await self.vllm.get_tokenizer()

    async def __aexit__(self, exc_type, exc_value, traceback):
        del self.vllm

    async def stream(
        self, prompt: InferenceInput
    ) -> AsyncGenerator[InferenceOutput, bool]:  # type: ignore
        formatted_prompt: str = (
            prompt
            if isinstance(prompt, str)
            else _apply_chat_template(self.tokenizer, prompt)
        )
        curr_uuid = str(uuid.uuid4())
        result_generator = self.vllm.generate(
            formatted_prompt,
            **self.generate_kwargs,
            request_id=curr_uuid,
        )
        async for request_output in result_generator:
            curr_res = self._wrap_output(request_output)
            abort = yield curr_res
            if abort:
                await self.vllm.abort(curr_uuid)
                break

    async def generate(self, prompt: InferenceInput) -> InferenceOutput:
        if isinstance(prompt, list):
            tokenizer = await self.vllm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(prompt)
        res = None
        async for res in self.stream(prompt):  # type: ignore
            pass
        return res

    def _wrap_output(self, req_output) -> InferenceOutput:
        return InferenceOutput(
            prompt=req_output.prompt,
            text=req_output.outputs[0].text,
            metadata={
                "model_name_or_path": self.llm_config.model_name_or_path,
                "lora_name_or_path": self.llm_config.lora_name_or_path,
            },
        )


ENGINES = {
    e.NAME: e
    for e in (
        VLLMEngine,
        WorkerVLLMEngine,
        OpenAIEngine,
        TogetherEngine,
    )
}


def get_inference_engine_cls(engine_name: str = "vllm") -> type[InferenceEngine]:
    return ENGINES[engine_name]


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
        def wrapper(*args, **kwargs):
            llm_config_kwargs = {
                key: kwargs.pop(key)
                for key in (
                    "model_name_or_path",
                    "tokenizer_name_or_path",
                    "context_window_tokens",
                    "max_new_tokens",
                    "temperature",
                )
            }
            llm_config = LLMConfig(**llm_config_kwargs)
            engine = get_inference_engine_cls(kwargs.pop("engine_name"))(llm_config)
            return func(*args, engine=engine, **kwargs)

        return wrapper

    return decorator
