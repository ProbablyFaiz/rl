import json
import subprocess
from pathlib import Path


def get_deployed_id(app_name: str) -> str:
    # Run `modal app list --json`
    result = subprocess.run(
        ["modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list Modal apps: {result.stderr}")
    apps = json.loads(result.stdout)
    return next(
        (
            app["App ID"]
            for app in apps
            if app["Description"] == app_name and app["State"] == "Deployed"
        ),
        None,
    )


def _get_uv_install_prefix():
    return ["python", "-m", "uv", "pip", "install", "--system", "--no-cache"]


def install_deps():
    subprocess.run(
        ["python", "-m", "pip", "install", "uv"],
        check=True,
    )
    subprocess.run(
        [*_get_uv_install_prefix(), "packaging", "wheel", "torch", "psutil"],
        check=True,
    )
    subprocess.run([*_get_uv_install_prefix(), "hf-transfer", "huggingface-hub"])


def install_rl():
    subprocess.run(
        [
            *_get_uv_install_prefix(),
            "rl[llm] @ git+https://github.com/ProbablyFaiz/rl.git@main",
            "--no-build-isolation",
        ],
        check=True,
    )


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Using safetensors
    )
    move_cache()
