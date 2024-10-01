import datetime
import gc
import math
import os
import socket
import subprocess
import sys
import tempfile
import textwrap as tw
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import tqdm

import rl.utils.io
from rl.llm.engines.core import (
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
    import openai
    import vllm
    import vllm.sequence
    from transformers import PreTrainedTokenizer


_RESPONSE_CANARY = "### Response template begins now, delete this line. ###"


@register_engine("manual_edit", required_modules=("transformers",))
class ManualEditEngine(InferenceEngine):
    _EDITOR = os.environ.get("EDITOR", "vim")

    tokenizer: "PreTrainedTokenizer"

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
        from transformers import AutoTokenizer

        if not isinstance(prompt, str):
            if not hasattr(self, "tokenizer"):
                model = "meta-llama/Meta-Llama-3-8B"
                if self.llm_config:
                    model = self.llm_config.model_name_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            prompt = apply_chat_template(self.tokenizer, prompt)

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


@register_engine("server_vllm", required_modules=("openai", "vllm"))
class WorkerVLLMEngine(InferenceEngine):
    client: "openai.Client"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        from openai import Client
        from transformers import AutoTokenizer

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
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.tokenizer_name_or_path
        )

        client_kwargs = {
            "base_url": f"http://localhost:{port}/v1",
            "api_key": api_key,
        }
        self.client = Client(**client_kwargs)

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

    # TODO: Refactor to rely on a ClientEngine under the hood
    #  Because this supports both text and chat input, we can't
    #  currently support JSON mode.
    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        formatted_prompt: str = (
            prompt
            if isinstance(prompt, str)
            else apply_chat_template(self.tokenizer, prompt)
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
            else apply_chat_template(self.tokenizer, prompt)
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


# There's little value in making this configurable, so we
#  just set it to 20, which is OpenAI's default.
#  https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs
_VLLM_NUM_LOGPROBS = 20


def _get_vllm_engine(
    llm_config: LLMConfig,
) -> tuple["VLLMEngine", dict]:
    import huggingface_hub
    import torch
    from vllm import (
        EngineArgs,
        SamplingParams,
    )
    from vllm import (
        LLMEngine as InternalVLLMEngine,
    )
    from vllm.lora.request import LoRARequest

    if not torch.cuda.is_available():
        raise ValueError(
            "VLLM requires a CUDA-compatible GPU and PyTorch with CUDA support."
        )

    if llm_config.num_gpus is not None and llm_config.num_gpus > 1:
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

    engine_cls = InternalVLLMEngine
    engine_args_cls = EngineArgs
    engine_args = engine_args_cls(**engine_args_kwargs)  # type: ignore
    print(engine_args)
    engine = engine_cls.from_engine_args(engine_args)  # type: ignore

    sampling_params = SamplingParams(
        max_tokens=llm_config.max_new_tokens,
        temperature=llm_config.temperature,
        frequency_penalty=llm_config.frequency_penalty,
        top_p=1.0,
    )
    if EngineFeature.RETURN_LOGPROBS in llm_config.features:
        sampling_params.logprobs = _VLLM_NUM_LOGPROBS

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
    import torch
    from transformers import AutoConfig

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
    transformers_config = AutoConfig.from_pretrained(llm_config.model_name_or_path)
    enable_prefix_caching = True
    if hasattr(transformers_config, "sliding_window"):
        enable_prefix_caching = False
        LOGGER.warning(
            "Model appears to have a sliding window, which VLLM doesn't support "
            "with prefix caching. Disabling prefix caching."
        )
    engine_args_kwargs = {
        "model": llm_config.model_name_or_path,
        "tensor_parallel_size": num_gpus,
        "max_model_len": llm_config.context_window_tokens,
        "enforce_eager": rl.utils.io.getenv("ENFORCE_EAGER", "0") == "1",
        "disable_log_stats": True,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": enable_prefix_caching,
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


def _parse_vllm_logprobs(
    logprobs: list[dict[int, "vllm.sequence.Logprob"]] | None,
) -> list[dict[int, float]]:
    if logprobs is None:
        raise ValueError("Expected logprobs in vLLM output but got None")

    output = []
    for logprob_dict in logprobs:
        output.append({token_id: lp.logprob for token_id, lp in logprob_dict.items()})
    return output


@register_engine(
    "vllm",
    required_modules=("vllm",),
    supported_features=(EngineFeature.RETURN_LOGPROBS,),
)
class VLLMEngine(InferenceEngine):
    vllm: "vllm.LLMEngine"
    generate_kwargs: dict

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)

    def __enter__(self):
        self.vllm, self.generate_kwargs = _get_vllm_engine(self.llm_config)
        self.tokenizer = self.vllm.get_tokenizer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import torch
        import torch.distributed
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        LOGGER.info("Unloading VLLM model from GPU memory...")
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.vllm.model_executor
        del self.vllm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        LOGGER.info("VLLM model unloaded.")

    def generate(self, prompt: InferenceInput) -> InferenceOutput:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: list[InferenceInput]) -> list[InferenceOutput]:
        formatted_prompts: list[str] = [
            (
                prompt
                if isinstance(prompt, str)
                else apply_chat_template(self.tokenizer, prompt)
            )
            for prompt in prompts
        ]
        vllm_outputs = self._get_vllm_outputs(formatted_prompts)

        inference_outputs = []
        for prompt, output in zip(prompts, vllm_outputs, strict=True):
            inf_output = InferenceOutput(
                prompt=prompt,
                text=output.outputs[0].text,
                metadata={},
            )
            if EngineFeature.RETURN_LOGPROBS in self.llm_config.features:
                inf_output.logprobs = _parse_vllm_logprobs(output.outputs[0].logprobs)
            inference_outputs.append(inf_output)
        return inference_outputs

    def _get_vllm_outputs(self, prompts: list[str]):
        from vllm import RequestOutput

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
