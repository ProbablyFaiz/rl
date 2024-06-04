import json
import subprocess
import time
from pathlib import Path

import modal
import modal.gpu

from rl.llm.config import LLMConfig


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
