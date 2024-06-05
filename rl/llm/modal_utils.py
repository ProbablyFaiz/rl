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
            if app["Description"] == app_name and app["State"] == "deployed"
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


def install_rl(revision: str = "main"):
    subprocess.run(
        [
            *_get_uv_install_prefix(),
            f"rl[llm] @ git+https://github.com/ProbablyFaiz/rl.git@{revision}",
            "--no-build-isolation",
        ],
        check=True,
    )


def download_model_to_image(model_dir, model_name):
    from transformers import AutoModel

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(model_dir)
