#!/usr/bin/env python3

import subprocess
from pathlib import Path
from threading import Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import rl.utils.click as click


class DebounceTimer:
    def __init__(self, timeout, callback):
        self.timeout = timeout
        self.callback = callback
        self.timer = None

    def start(self):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.timeout, self.callback)
        self.timer.start()


class SyncHandler(FileSystemEventHandler):
    def __init__(
        self,
        local_dir: Path,
        remote_user: str,
        remote_host: str,
        remote_dir: str,
        delay: float,
    ):
        self.local_dir = local_dir
        self.remote_user = remote_user
        self.remote_host = remote_host
        self.remote_dir = remote_dir
        self.debounce_timer = DebounceTimer(delay, self.sync_repo)

    def on_any_event(self, event):
        self.debounce_timer.start()

    def sync_repo(self):
        click.echo("Syncing repository...")
        git_files = subprocess.check_output(
            ["git", "ls-files"], cwd=self.local_dir, text=True
        ).splitlines()
        file_list = "\n".join(git_files)
        subprocess.run(
            [
                "rsync",
                "-avz",
                "--files-from=-",
                str(self.local_dir),
                f"{self.remote_user}@{self.remote_host}:{self.remote_dir}/",
            ],
            input=file_list,
            text=True,
        )
        click.echo("Sync completed.")


@click.command()
@click.option(
    "-l",
    "--local",
    "local_dir",
    type=click.Path(exists=True),
    required=True,
    help="Local directory to sync",
)
@click.option("-r", "--remote-dir", required=True, help="Remote directory to sync to")
@click.option("-u", "--user", required=True, help="Remote user")
@click.option("-h", "--host", required=True, help="Remote host")
@click.option("-d", "--delay", default=2.0, help="Debounce delay in seconds")
def main(local_dir: str, remote_dir: str, user: str, host: str, delay: float):
    local_path = Path(local_dir).resolve()
    handler = SyncHandler(local_path, user, host, remote_dir, delay)

    handler.sync_repo()  # Initial sync

    observer = Observer()
    observer.schedule(handler, str(local_path), recursive=True)
    observer.start()

    click.echo(f"Watching {local_path}. Press Ctrl+C to stop. Press Enter to resync.")
    try:
        while True:
            if input() == "":
                handler.sync_repo()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
