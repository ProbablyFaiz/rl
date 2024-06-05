import json
import os
import subprocess
import time

import modal
import modal.gpu

import rl.llm.modal_utils

_IMAGE_MODEL_DIR = "/model"
_DEPLOY_CONFIG = json.loads(os.getenv("MODAL_DEPLOY_CONFIG", "{}"))

if not _DEPLOY_CONFIG:
    raise ValueError("MODAL_DEPLOY_CONFIG not set")

print(f"🚀 Deploying with config: {json.dumps(_DEPLOY_CONFIG, indent=2)}")


def _derive_gpu_config(deploy_config):
    return modal.gpu.A100(size="80GB", count=deploy_config.get("num_gpus", 1))


def _get_vllm_image(deploy_config):
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("git")
        .run_function(rl.llm.modal_utils.install_deps)
        .run_function(rl.llm.modal_utils.install_rl)
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .run_function(
            rl.llm.modal_utils.download_model_to_image,
            timeout=60 * 20,
            kwargs={
                "model_dir": _IMAGE_MODEL_DIR,
                "model_name": deploy_config["llm_config"]["model_name_or_path"],
            },
            secrets=[modal.Secret.from_name("huggingface-token")],
        )
        .env(
            {"MODAL_DEPLOY_CONFIG": json.dumps(deploy_config), "ENFORCE_EAGER": "true"}
        )
    )


_VLLM_IMAGE = _get_vllm_image(_DEPLOY_CONFIG)
_GPU_CONFIG = _derive_gpu_config(_DEPLOY_CONFIG)

app = modal.App(name=_DEPLOY_CONFIG["app_name"])


@app.cls(
    cpu=4.0,
    gpu=_GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=_VLLM_IMAGE,
)
class Model:
    engine = None
    config = None

    @modal.enter()
    def start_engine(self):
        from rl.llm.config import LLMConfig
        from rl.llm.engines import VLLMEngine

        self.config = LLMConfig(**_DEPLOY_CONFIG["llm_config"])
        self.config.model_name_or_path = _IMAGE_MODEL_DIR
        self.config.tokenizer_name_or_path = _IMAGE_MODEL_DIR

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        self.engine = VLLMEngine(self.config)
        self.engine.__enter__()
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    def generate(self, inference_input):
        return self.engine.generate(inference_input)

    @modal.method()
    def batch_generate(self, inference_inputs):
        return self.engine.batch_generate(inference_inputs)

    @modal.exit()
    def stop_engine(self):
        self.engine.__exit__(None, None, None)
        if self.config.num_gpus > 1:
            import ray

            ray.shutdown()


@app.local_entrypoint()
async def main():
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    model = Model()
    results = model.batch_generate.remote(questions)
    print("\n".join([r.text for r in results]))
    print("\n".join([r.text for r in results]))
    print("\n".join([r.text for r in results]))
