import datetime
import functools
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click
import pexpect
import questionary
import regex
import rich
import rich.progress
import rich.table
from strenum import StrEnum

from rl.cli.duo import Duo, DuoConfig
from rl.cli.nodelist_parser import parse_nodes_str

CURRENT_USER = subprocess.run(
    ["whoami"], stdout=subprocess.PIPE, text=True
).stdout.strip()
CURRENT_GROUP = subprocess.run(
    ["id", "-gn"], stdout=subprocess.PIPE, text=True
).stdout.strip()
ON_SHERLOCK = os.path.exists("/usr/bin/sbatch")

# Must use full path to avoid issues with conda environments
SSH_PATH = "/bin/ssh"

# Check if fish is installed, otherwise use bash
SHELL_PATH = (
    subprocess.run(
        ["which", "fish"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout.strip()
    or subprocess.run(
        ["which", "bash"], stdout=subprocess.PIPE, text=True
    ).stdout.strip()
)

LOG_DIR = Path("/scratch/users") / CURRENT_USER / "logs"

CHECK_BATCH_EVERY = 10

DEFAULT_JOB_NAME = f"rl-{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}"

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


@dataclass
class JobInfo:
    """Information about a job returned by squeue."""

    job_id: str
    job_name: str
    user: str
    partition: str
    nodes: list[str]
    state: "JobState"


class JobState(StrEnum):
    RUNNING = "R"
    PENDING = "PD"


STATE_NAME_MAP = {
    JobState.RUNNING: "Running",
    JobState.PENDING: "Pending",
}


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
        if not ON_SHERLOCK:
            raise RLError("This command must be run on Sherlock.")
        return func(*args, **kwargs)

    return wrapper


@cli.command(
    help="Create an interactive job, even on the owners partition",
    context_settings=dict(ignore_unknown_options=True),
)
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
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Run job interactively, rather than via batch. Not recommended.",
)
@click.option(
    "--command",
    "-cmd",
    help="Command to run as a batch job",
    type=str,
)
@click.argument(
    "slurm_args",
    nargs=-1,
    type=str,
    required=False,
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
    interactive: bool,
    command: str | None,
    slurm_args: list[str],
):
    if interactive:
        assert partition != "owners", "Cannot run interactive job on owners partition"

    if slurm_args:
        match slurm_args[0]:
            case "list":
                _list_jobs()
                return
            case _:
                pass

    LOG_DIR.mkdir(exist_ok=True, parents=True)

    common_args = [
        "--partition",
        partition,
        "--job-name",
        name,
        "--cpus-per-gpu" if gpus else "--cpus",
        str(cpus),
        "--mem",
        mem,
        "--time",
        time,
        "--use-min-nodes",
        *slurm_args,
    ]
    if gpus:
        common_args.extend(
            [
                "-C",
                gpu_constraint,
                "--gpu_cmode",
                "shared",
                "--gpus",
                str(gpus),
            ]
        )

    if not interactive:
        create_batch_job(common_args, name, time)
    else:
        rich.print("[green]Starting interactive job...[/green]")
        # Show the output to the user
        subprocess.run(
            [
                "srun",
                *common_args,
                "--pty",
                command or SHELL_PATH,
            ]
        )


def _list_jobs():
    jobs = _get_all_jobs(show_progress=True)
    table = rich.table.Table(title="Jobs")
    table.add_column("Job ID")
    table.add_column("Job Name")
    table.add_column("User")
    table.add_column("Partition")
    table.add_column("Nodes")
    table.add_column("State")
    for job in jobs:
        table.add_row(
            job.job_id,
            job.job_name,
            job.user,
            job.partition,
            ", ".join(job.nodes),
            STATE_NAME_MAP[job.state],
        )
    rich.print(table)


def create_batch_job(sbatch_args, name, job_time):
    parsed_time = [int(x) for x in job_time.split(":")]
    sleep_time = 0
    for i, t in enumerate(reversed(parsed_time)):
        sleep_time += t * 60**i

    sbatch_args = [
        "sbatch",
        "--output",
        f"{LOG_DIR}/{name}-%j.out",
        "--error",
        f"{LOG_DIR}/{name}-%j.err",
        *sbatch_args,
        "--wrap",
        f"tmux new-session -d -s rl && python -c 'import time; time.sleep({sleep_time})'",
    ]
    subprocess.run(sbatch_args, check=True)

    curr_job: JobInfo
    with rich.progress.Progress(transient=True) as progress:
        # noinspection PyTypeChecker
        task = progress.add_task(
            "[green]Waiting for job to start...", start=True, total=None
        )
        while True:
            time.sleep(CHECK_BATCH_EVERY)
            curr_job = next((j for j in _get_all_jobs() if j.job_name == name), None)
            if curr_job is None:
                raise RLError(
                    "Job not found when checking status with squeue; what happened?"
                )
            if curr_job.state == JobState.RUNNING:
                progress.update(task, completed=1)
                break
    if len(curr_job.nodes) > 1:
        rich.print(
            f"[green]Job {curr_job.job_id} started on nodes {', '.join(curr_job.nodes)}. SSHing into first node...[/green]"
        )
    else:
        rich.print(
            f"[green]Job {curr_job.job_id} started on node {curr_job.nodes[0]}. SSHing into node...[/green]"
        )
    _ssh_within_sherlock(curr_job.nodes[0])
    if click.confirm("Left the job, do you want to cancel it?"):
        subprocess.run(["scancel", curr_job.job_id])
        rich.print("[red]Job ended[/red]")
    else:
        rich.print(f"[green]Job {curr_job.job_id} will continue running[/green]")


@_must_run_on_sherlock
def _get_all_jobs(show_progress=False):
    if show_progress:
        with rich.progress.Progress(transient=True) as progress:
            # noinspection PyTypeChecker
            task = progress.add_task(
                "[green]Checking jobs in queue...[/green]", total=None
            )
            results = _get_all_jobs(show_progress=False)
            progress.update(task, completed=1)
            return results

    output = subprocess.run(
        ["squeue", "-u", CURRENT_USER, "-h", "-o", "%A %j %u %P %N %t"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    results = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Splitting on exactly one whitespace is important here,
        #  split() by default treats multiple whitespaces as one,
        #  which goes wrong when the nodes field is empty.
        job_id, job_name, user, partition, nodes, state = line.split(" ")
        results.append(
            JobInfo(
                job_id=job_id,
                job_name=job_name,
                user=user,
                partition=partition,
                nodes=parse_nodes_str(nodes),
                state=state,
            )
        )
    return results


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


@cli.command(help="Approve a Duo login request")
def approve():
    approve_duo_login()
    rich.print("[green]Duo login approved[/green]")


@cli.command(help="Cancel a running job")
@click.argument("job_id", type=str, required=False)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip the confirmation prompt",
)
@_must_run_on_sherlock
def cancel(job_id: str, yes: bool):
    if not job_id:
        job_id = _select_job()
    if yes or click.confirm(f"Are you sure you want to cancel job {job_id}?"):
        subprocess.run(["scancel", job_id])
    rich.print(f"[red]Job {job_id} cancelled[/red]")


def _select_job() -> str:
    jobs = _get_all_jobs(show_progress=True)
    job_ids = [job.job_id for job in jobs]
    if not job_ids:
        raise RLError("No jobs found to cancel.")
    job_id = (
        questionary.select("Select a job to cancel", choices=job_ids).ask()
        if len(job_ids) > 1
        else job_ids[0]
    )
    return job_id


@cli.command(help="SSH into Sherlock or into a particular job while on Sherlock")
@click.argument("node", required=False, type=str)
def ssh(node: str):
    if not ON_SHERLOCK:
        _ssh_into_sherlock(node)
    else:
        _ssh_within_sherlock(node)


@_requires_duo
@_requires_sherlock_credentials
def _ssh_into_sherlock(node: str):
    credentials = _read_credentials()
    duo = Duo.from_config(_read_duo())

    node = node or credentials["node"]
    node_url = f"{node}.sherlock.stanford.edu"
    rich.print(f"[green]Logging you in to {node_url}[/green]")
    ssh_command = f"ssh -o StrictHostKeyChecking=no {credentials['username']}@{node_url} -t 'fish || bash'"
    _run_sherlock_ssh(ssh_command, credentials, duo)


@_must_run_on_sherlock
def _ssh_within_sherlock(node: str):
    if not node:
        node = _select_node()
    rich.print(f"[green]SSHing into {node}[/green]")
    # When SSHing in, we want to try to tmux attach and if that fails, just open a shell
    run_command = "tmux attach || fish || bash"
    subprocess.run([SSH_PATH, node, "-t", run_command])


def _select_node() -> str:
    running_jobs = [
        job
        for job in _get_all_jobs(show_progress=True)
        if job.state == JobState.RUNNING
    ]
    if not running_jobs:
        rich.print(
            "[yellow]No running jobs found. Please enter a node to ssh into.[/yellow]"
        )
        return click.prompt("Node", type=str)
    name_to_node_map = {}
    node_names = []
    for job in running_jobs:
        for node in job.nodes:
            node_name = f"{node} (job {job.job_id})"
            name_to_node_map[node_name] = node
            node_names.append(node_name)
    selection = (
        questionary.select("Select a node to ssh into", choices=node_names).ask()
        if len(node_names) > 1
        else node_names[0]
    )
    return name_to_node_map[selection]


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
    scp_prefix = "scp -o StrictHostKeyChecking=no"
    scp_command = (
        f"{scp_prefix} {' '.join(scp_options)} {source} {credentials['username']}@{node_url}:{destination}"
        if direction == "to"
        else f"{scp_prefix} {' '.join(scp_options)} {credentials['username']}@{node_url}:{source} {destination}"
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

    # It seems like the Duo MFA doesn't actually go through until we .interact()
    #  So we spin up a thread to approve it in the background when it's ready
    background_tasks = [
        threading.Thread(target=_approve_when_ready, args=(duo,)),
        threading.Thread(target=_resize_ssh, args=(ssh,)),
    ]
    for task in background_tasks:
        task.start()
    ssh.interact()


def _resize_ssh(ssh: pexpect.spawn):
    previous_size = None
    while ssh.isalive():
        current_size = os.get_terminal_size()
        if current_size != previous_size:
            ssh.setwinsize(current_size.lines, current_size.columns)
            previous_size = current_size
        time.sleep(0.2)


def _approve_when_ready(duo: Duo):
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
