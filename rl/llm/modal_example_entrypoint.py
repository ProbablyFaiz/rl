import os
import time

import modal

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# For this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# like Mistral 7B, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [terms of use](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal as `huggingface-secret`.
#
# Tip: avoid using global variables in this function.
# Changes to code outside this function will not be detected, and the download step will not re-run.
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


# ### Image definition
# We’ll start from Modal's Debian slim image.
# Then we’ll use `run_function` with `download_model_to_image` to write the model into the container image.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "example-vllm-inference", image=image
)  # Note: prior to April 2024, "app" was called "stub"

# Using `image.imports` allows us to have a reference to vLLM in global scope without getting an error when our script executes locally.
with image.imports():
    import vllm

# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions),
# using a `load_model` method decorated with `@modal.enter`. This enables us to load the model into memory just once,
# every time a container starts up, and to keep it cached on the GPU for subsequent invocations of the function.
#
# The `vLLM` library allows the code to remain quite clean.

# Hint: try out an H100 if you've got a large model or big batches!
GPU_CONFIG = modal.gpu.A100(count=1)  # 40GB A100 by default


@app.cls(gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def load_model(self):
        # Tip: models that are not fully implemented by Hugging Face may require `trust_remote_code=true`.
        self.llm = vllm.LLM(MODEL_DIR, tensor_parallel_size=GPU_CONFIG.count)
        self.template = """[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST]"""

    @modal.method()
    def generate(self, user_questions):
        prompts = [self.template.format(system="", user=q) for q in user_questions]

        sampling_params = vllm.SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=256,
            presence_penalty=1.15,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )
