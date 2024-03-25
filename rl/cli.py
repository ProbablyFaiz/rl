import datetime
import functools
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import click
import pexpect
import regex
import rich
import rich.progress

from rl.duo import Duo, DuoConfig

CURRENT_USER = subprocess.run(
    ["whoami"], stdout=subprocess.PIPE, text=True
).stdout.strip()
CURRENT_GROUP = subprocess.run(
    ["id", "-gn"], stdout=subprocess.PIPE, text=True
).stdout.strip()

# Must use full path to avoid issues with conda environments
SSH_PATH = "/bin/ssh"

# Check if fish is installed, otherwise use bash
SHELL_PATH = (
    subprocess.run(["which", "fish"], stdout=subprocess.PIPE, text=True).stdout.strip()
    or subprocess.run(
        ["which", "bash"], stdout=subprocess.PIPE, text=True
    ).stdout.strip()
)

LOG_DIR = Path("/scratch/users") / CURRENT_USER / "logs"

CHECK_BATCH_EVERY = 10

DEFAULT_JOB_NAME = (
    f"interactive-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
)

BASE_CONFIG_DIR = Path("~/.config/rl").expanduser()

CREDENTIALS_FILE = BASE_CONFIG_DIR / "sherlock.json"
DUO_FILE = BASE_CONFIG_DIR / "duo.json"
NODE_OPTIONS = [
    "sh03-ln01",
    "sh03-ln02",
    "sh03-ln03",
    "sh03-ln04",
]


class RLError(Exception):
    pass


@click.group()
def cli():
    pass


def _requires_duo(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        duo_config = _read_duo()
        if not duo_config:
            raise RLError("Duo not configured. Run `rl configure duo` to configure it.")
        return func(*args, **kwargs)

    return wrapper


def _requires_sherlock_credentials(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        credentials = _read_credentials()
        if not credentials:
            raise RLError(
                "Sherlock credentials not found. Run `rl configure sherlock` to set them."
            )
        return func(*args, **kwargs)

    return wrapper


def _must_run_on_sherlock(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not os.path.exists("/usr/bin/sbatch"):
            raise RLError("This command must be run on Sherlock.")
        return func(*args, **kwargs)

    return wrapper


@cli.command(help="Create an interactive job, even on the owners partition")
@click.option(
    "--partition",
    "-p",
    help="Partition. If 'owners' specified, will create a batch job then ssh into it",
    default=CURRENT_GROUP,
    show_default=True,
    type=str,
)
@click.option(
    "--name",
    "-n",
    help="Name of the job",
    default=DEFAULT_JOB_NAME,
    show_default=True,
    type=str,
)
@click.option(
    "--gpus",
    "-g",
    help="Number of GPUs",
    default=0,
    show_default=True,
    type=int,
)
@click.option(
    "--gpu-constraint",
    "-gc",
    help="GPU constraint",
    default="GPU_MEM:40GB|GPU_MEM:80GB",
    show_default=True,
    type=str,
)
@click.option(
    "--cpus",
    "-c",
    help="Number of CPUs per GPU or total CPUs if no GPUs specified",
    default=4,
    show_default=True,
    type=int,
)
@click.option(
    "--mem",
    "-m",
    help="Memory",
    default="32GB",
    show_default=True,
    type=str,
)
@click.option(
    "--time",
    "-t",
    help="Time",
    default="24:00:00",
    show_default=True,
    type=str,
)
@_must_run_on_sherlock
def job(
    partition: str,
    name: str,
    gpus: int,
    gpu_constraint: str,
    cpus: int,
    mem: str,
    time: str,
):
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    common_args = [
        "--partition",
        partition,
        "--job-name",
        name,
        "-C",
        gpu_constraint,
        "--gpu_cmode",
        "shared",
        "--gpus",
        str(gpus),
        "--cpus-per-gpu" if gpus else "--cpus",
        str(cpus),
        "--mem",
        mem,
        "--time",
        time,
    ]

    if partition == "owners":
        create_batch_job(common_args, name, time)
        return

    rich.print("[green]Starting interactive job...[/green]")
    # Show the output to the user
    subprocess.run(
        [
            "srun",
            *common_args,
            "--pty",
            SHELL_PATH,
        ]
    )


def create_batch_job(sbatch_args, name, job_time):
    parsed_time = [int(x) for x in job_time.split(":")]
    sleep_time = 0
    for i, t in enumerate(reversed(parsed_time)):
        sleep_time += t * 60**i

    sbatch_args = [
        "sbatch",
        *sbatch_args,
        "--wrap",
        f"python -c 'import time; time.sleep({sleep_time})'",
    ]
    subprocess.run(sbatch_args, check=True)

    job_node, job_id = None, None
    with rich.progress.Progress(transient=True) as progress:
        task = progress.add_task(
            "[green]Waiting for job to start...", start=True, total=None
        )
        while True:
            time.sleep(CHECK_BATCH_EVERY)
            job_info = subprocess.run(
                ["squeue", "-u", CURRENT_USER, "-n", name, "-h"],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            ).stdout.split()
            if len(job_info) == 0:
                raise RLError(
                    "Job not found when checking status with squeue, what happened?"
                )
            if job_info[4] == "R":
                job_node, job_id = job_info[7], job_info[0]
                progress.update(task, completed=1)
                break

    rich.print(
        f"[green]Job {job_id} started on node {job_node}. SSHing into node...[/green]"
    )
    ssh_args = [
        SSH_PATH,
        job_node,
    ]
    subprocess.run(ssh_args)
    if click.confirm("Left the job, do you want to cancel it?"):
        subprocess.run(["scancel", job_id])
        rich.print("[red]Job ended[/red]")
    else:
        rich.print(f"[green]Job {job_id} will continue running[/green]")


@cli.command(short_help="Temporarily modify files to avoid Sherlock auto-deletion")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively touch files in a directory, if passed.",
)
@_must_run_on_sherlock
def touch(paths: list[Path], recursive: bool):
    # Merely 'touch'ing is insufficient; Sherlock requires an actual modification to the file
    file_paths = []
    if any(p.is_dir() for p in paths):
        if not recursive:
            raise click.ClickException(
                "You must pass --recursive/-r if you want to recursively touch a directory"
            )
        rich.print("[green]Finding all files within directories to touch...[/green]")
    for path in paths:
        if path.is_dir():
            file_paths.extend([p for p in Path(path).rglob("*") if p.is_file()])
        else:
            file_paths.append(Path(path))

    with rich.progress.Progress() as progress:
        task = progress.add_task("[green]Touching files...", total=len(file_paths))
        for file_path in file_paths:
            try:
                _touch_file(file_path)
            except Exception as e:
                rich.print(f"[red]Error touching {file_path}: {e}[/red]")
            progress.update(task, advance=1)


def _touch_file(path: Path):
    with open(path, "ab") as f:
        f.write(b" ")
    subprocess.run(["truncate", "-s", "-1", str(path)])


@cli.command(help="SSH into Sherlock")
@click.option(
    "--node",
    "-n",
    help="Override the node to ssh into. E.g., `sh03-ln01`.",
    required=False,
    type=str,
)
@_requires_duo
@_requires_sherlock_credentials
def ssh(node: str):
    credentials = _read_credentials()
    duo = Duo.from_config(_read_duo())

    node = node or credentials["node"]
    node_url = f"{node}.sherlock.stanford.edu"
    rich.print(f"[green]Logging in to {node_url}[/green]")
    ssh_command = f"ssh {credentials['username']}@{node_url}"
    _run_sherlock_ssh(ssh_command, credentials, duo)


@cli.command(
    help="SCP files to/from Sherlock",
    context_settings=dict(ignore_unknown_options=True),
)
@click.argument("direction", type=click.Choice(["to", "from"]))
@click.argument("source", type=str)
@click.argument("destination", type=str)
@click.argument("scp_options", nargs=-1, type=str)
@_requires_duo
@_requires_sherlock_credentials
def scp(direction: str, source: str, destination: str, scp_options: list[str]):
    credentials = _read_credentials()
    duo = Duo.from_config(_read_duo())

    node = credentials["node"]
    node_url = f"{node}.sherlock.stanford.edu"
    rich.print(f"[green]Logging in to {node_url}[/green]")
    scp_command = (
        f"scp {' '.join(scp_options)} {source} {credentials['username']}@{node_url}:{destination}"
        if direction == "to"
        else f"scp {' '.join(scp_options)} {credentials['username']}@{node_url}:{source} {destination}"
    )
    _run_sherlock_ssh(scp_command, credentials, duo)


_MFA_LINE_REGEX = regex.compile(
    r"\s*(?P<number>\d+)\. Duo Push to XXX-XXX-0199\s*", regex.IGNORECASE
)


def _run_sherlock_ssh(ssh_command: str, credentials: dict, duo: Duo) -> None:
    ssh = pexpect.spawn(ssh_command)
    ssh.expect("password:")
    ssh.sendline(credentials["password"])

    ssh.expect(r"Passcode or option \(\d+-\d+\): ")
    duo_output = ssh.before.decode()
    option_to_select = _MFA_LINE_REGEX.search(duo_output)
    if not option_to_select:
        raise RLError("Could not find Duo MFA option")
    option_to_select = option_to_select.group("number")
    ssh.sendline(option_to_select)
    # Clear the buffer before we hand over control to the user
    ssh.expect("\n")

    if ssh_command.startswith("ssh"):
        term_size = os.get_terminal_size()
        ssh.setwinsize(term_size.lines, term_size.columns)

    # It seems like the Duo MFA doesn't actually go through until we .interact()
    #  So we spin up a thread to approve it in the background when it's ready
    threading.Thread(target=_approve_when_ready, args=(duo,)).start()
    ssh.interact()


def _approve_when_ready(duo):
    for _ in range(10):
        if transactions := duo.get_transactions():
            for tr in transactions:
                duo.answer_transaction(tr.id, approve=True)
            return
    raise RLError("Failed to approve Duo MFA after 10 attempts.")


@cli.group(help="Configure different aspects of rl")
def configure():
    pass


@configure.command(help="Configure RL's access to Duo")
def duo():
    if _read_duo():
        rich.print(
            "[yellow]Warning: Duo already configured. Continuing will overwrite.[/yellow]"
        )
    qr_url = click.prompt("Enter image address of the Duo activation QR code")
    duo = Duo.from_qr_url(qr_url)
    _write_duo(duo.to_config())


@configure.command(help="Configure RL's access to Sherlock")
def sherlock():
    if _read_credentials():
        rich.print(
            "[yellow]Warning: Sherlock credentials already configured. Continuing will overwrite.[/yellow]"
        )
    username = click.prompt("Stanford NetID")
    password = click.prompt("NetID Password", hide_input=True)
    node_choice = random.choice(NODE_OPTIONS)
    print(
        f"Selected node {node_choice} for you, you can change this in {CREDENTIALS_FILE}."
    )
    credentials = {
        "username": username,
        "password": password,
        "node": node_choice,
    }
    _write_credentials(credentials)
    print(f"Credentials saved to {CREDENTIALS_FILE}")


def approve_duo_login():
    duo_config = _read_duo()
    if not duo_config:
        raise RLError("Duo not configured. Run `rl configure duo` to configure it.")
    duo = Duo.from_config(duo_config)
    duo.answer_latest_transaction(approve=True)


def _read_duo() -> DuoConfig | None:
    if not DUO_FILE.exists():
        return None
    with open(DUO_FILE, "r") as f:
        return json.load(f)


def _write_duo(duo_info: DuoConfig):
    with open(DUO_FILE, "w") as f:
        json.dump(duo_info, f, indent=2)


def _write_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f, indent=2)


def _read_credentials() -> dict | None:
    if not CREDENTIALS_FILE.exists():
        return None
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    BASE_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    cli()
