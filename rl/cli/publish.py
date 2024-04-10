"""Builds the RL library with pyinstaller (--onefile) and uploads it to the specified GitHub release."""
import os
import subprocess
from pathlib import Path

import click
import github
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

ENTRY_POINT = PROJECT_ROOT_DIR / "rl" / "cli" / "cli.py"
DIST_DIR = PROJECT_ROOT_DIR / "dist"
DIST_FILE = DIST_DIR / "rl"

GITHUB_REPO = "ProbablyFaiz/rl"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


@click.command()
@click.option("--tag", help="Tag to create for the release", required=True)
def publish(tag: str):
    os.chdir(PROJECT_ROOT_DIR)

    subprocess.run(["git", "tag", tag])
    subprocess.run(["git", "push", "origin", tag])

    subprocess.run(["pyinstaller", "--onefile", str(ENTRY_POINT), "-n", DIST_FILE.name])

    gh = github.Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    release_name = f"Release {tag}"
    release = repo.create_git_release(tag, release_name, release_name, draft=True)
    release.upload_asset(str(DIST_FILE), label="rl")
    release.update_release(release_name, release_name, draft=False)


if __name__ == "__main__":
    publish()
