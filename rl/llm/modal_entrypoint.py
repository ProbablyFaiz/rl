import json
import os
import subprocess
import time
from pathlib import Path

import modal
import modal.gpu

_IMAGE_MODEL_DIR = "/model"
_DEPLOY_CONFIG = json.loads(os.getenv("MODAL_DEPLOY_CONFIG", "{}"))

if not _DEPLOY_CONFIG:
    raise ValueError("MODAL_DEPLOY_CONFIG not set")

print(f"🚀 Deploying with config: {json.dumps(_DEPLOY_CONFIG, indent=2)}")


def _download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


def _derive_gpu_config(deploy_config):
    return modal.gpu.A100(size="80GB", count=deploy_config.get("num_gpus", 1))


def _install_deps():
    return subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "rl[llm] @ git+https://github.com/ProbablyFaiz/rl.git@main",
        ],
        check=True,
    )


def _get_vllm_image(deploy_config):
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "uv",
        )
        .run_function(_install_deps)
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .run_function(
            _download_model_to_image,
            timeout=60 * 20,
            kwargs={
                "model_dir": _IMAGE_MODEL_DIR,
                "model_name": deploy_config["model_name_or_path"],
            },
            secrets=[modal.Secret.from_name("huggingface-token")],
        )
    )


_VLLM_IMAGE = _get_vllm_image(_DEPLOY_CONFIG)
_GPU_CONFIG = _derive_gpu_config(_DEPLOY_CONFIG)

app = modal.App(name=_DEPLOY_CONFIG["app_name"])


@app.cls(
    gpu=_GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=_VLLM_IMAGE,
)
class Model:
    engine = None

    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        if "model" in _DEPLOY_CONFIG["vllm_kwargs"]:
            del _DEPLOY_CONFIG["vllm_kwargs"]
        engine_args = AsyncEngineArgs(
            model=_IMAGE_MODEL_DIR,
            **_DEPLOY_CONFIG["vllm_kwargs"],
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    async def completion_stream(self, input_text: str):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,
            repetition_penalty=1.1,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            input_text,
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield (
            f"\n\tGenerated {num_tokens} tokens in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {_GPU_CONFIG}.\n"
        )

    @modal.exit()
    def stop_engine(self):
        if _DEPLOY_CONFIG["num_gpus"] > 1:
            import ray

            ray.shutdown()


@app.local_entrypoint()
def main():
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    model = Model()
    for question in questions:
        print("Sending new request:", question, "\n\n")
        for text in model.completion_stream.remote_gen(question):
            print(text, end="", flush=text.endswith("\n"))
