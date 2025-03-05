import datetime
import functools
import json
import os
import random
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import paramiko
import pexpect  # type: ignore
import questionary
import rich
import rich.markup
import rich.progress
import rich.table
from pydantic import BaseModel
from strenum import StrEnum

import rl.utils.click as click
from rl.cli.duo import Duo, DuoConfig
from rl.cli.nodelist_parser import parse_nodes_str
from rl.utils.flags import set_debug_mode

set_debug_mode(False)

# region Constants

CURRENT_USER = subprocess.run(
    ["whoami"], stdout=subprocess.PIPE, text=True
).stdout.strip()
CURRENT_GROUP = subprocess.run(
    ["id", "-gn"], stdout=subprocess.PIPE, text=True
).stdout.strip()
ON_SHERLOCK = Path("/usr/bin/sbatch").exists()

# Must use full path to avoid issues with conda environments
SSH_PATH = "/bin/ssh"
SSHD_PATH = "/usr/sbin/sshd"

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

DEFAULT_JOB_NAME = f"rl_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"

BASE_CONFIG_DIR = Path("~/.config/rl").expanduser()

CREDENTIALS_FILE = BASE_CONFIG_DIR / "sherlock.json"
DUO_FILE = BASE_CONFIG_DIR / "duo.json"
NODE_OPTIONS = [
    "sh03-ln01",
    "sh03-ln02",
    "sh03-ln03",
    "sh03-ln04",
]


SHERLOCK_HOME_DIR = Path("/home/users") / CURRENT_USER
SHERLOCK_SSH_DIR = SHERLOCK_HOME_DIR / ".ssh"

TUNNEL_HOST_NAME = "rl"

_DEFAULT_SSH_SERVER_PORT = 5549
_DEFAULT_SSH_TUNNEL_PORT = 5549


_MFA_LINE_REGEX = re.compile(
    r"\s*(?P<number>\d+)\. Duo Push to XXX-XXX-0199\s*", re.IGNORECASE
)

# endregion

# region Data Structures


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
    time_elapsed: str
    time_remaining: str


class Credentials(BaseModel):
    username: str
    password: str
    node: str


class JobState(StrEnum):
    RUNNING = "R"
    PENDING = "PD"


STATE_DISPLAY_MAP = {
    JobState.RUNNING: "[green]Running[/green]",
    JobState.PENDING: "[yellow]Pending[/yellow]",
}

# endregion


@click.group()
def cli():
    pass


# region Helpers


def _require_duo(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        duo_config = _read_duo()
        if not duo_config:
            raise RLError("Duo not configured. Run `rl configure duo` to configure it.")
        if "duo" not in kwargs:
            kwargs["duo"] = Duo.from_config(duo_config)
        return func(*args, **kwargs)

    return wrapper


def _require_sherlock_credentials(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        credentials = _read_credentials()
        if not credentials:
            raise RLError(
                "Sherlock credentials not found. Run `rl configure sherlock` to set them."
            )
        if "credentials" not in kwargs:
            kwargs["credentials"] = credentials
        return func(*args, **kwargs)

    return wrapper


def _must_run_on_sherlock(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ON_SHERLOCK:
            raise RLError("This command must be run on Sherlock.")
        return func(*args, **kwargs)

    return wrapper


def _read_duo() -> DuoConfig | None:
    if not DUO_FILE.exists():
        return None
    with DUO_FILE.open() as f:
        return json.load(f)


def _write_duo(duo_info: DuoConfig):
    with DUO_FILE.open("w") as f:
        json.dump(duo_info, f, indent=2)


def _write_credentials(credentials: Credentials):
    with CREDENTIALS_FILE.open("w") as f:
        json.dump(credentials.model_dump(), f, indent=2)


def _read_credentials() -> Credentials | None:
    if not CREDENTIALS_FILE.exists():
        return None
    with CREDENTIALS_FILE.open() as f:
        return Credentials.model_validate(json.load(f))


@_require_duo
def approve_duo_login(*, duo: Duo):
    duo.answer_latest_transaction(approve=True)


def _log_command(command: list[str]):
    command_str = rich.markup.escape(shlex.join(command))
    rich.print(f"[bold]>[/bold] [green]{command_str}[/green]")


# endregion


# region Jobs


@cli.command(
    help="Create an interactive job, even on the owners partition",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--partition",
    "-p",
    help=f"Partition. Default is {CURRENT_GROUP}. If 'owners' specified, will create a batch job then ssh into it",
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
    partition: str | None,
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
                _list_jobs(partition)
                return
            case _:
                pass

    # We don't set the click default because we don't want the default partition
    #  value when the user does rl job list, so we just set it after checking
    #  that the user isn't running list
    if not partition:
        partition = CURRENT_GROUP

    LOG_DIR.mkdir(exist_ok=True, parents=True)

    common_args = [
        "--partition",
        partition,
        "--job-name",
        name,
        "-c" if gpus else "--cpus-per-task",
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
        full_args = [
            "srun",
            *common_args,
            "--pty",
            command or SHELL_PATH,
        ]
        _log_command(full_args)
        subprocess.run(full_args, check=True)


def _list_jobs(partition: str | None = None):
    jobs = _get_all_jobs(partition=partition, show_progress=False)
    table = rich.table.Table()
    table.add_column("Job ID")
    table.add_column("Job Name")
    table.add_column("User")
    table.add_column("Partition")
    table.add_column("Nodes")
    table.add_column("State")
    table.add_column("Time Elapsed")
    table.add_column("Time Remaining")
    for job in jobs:
        table.add_row(
            job.job_id,
            job.job_name,
            job.user,
            job.partition,
            ", ".join(job.nodes),
            STATE_DISPLAY_MAP[job.state],
            job.time_elapsed,
            job.time_remaining,
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
        f"{LOG_DIR}/{name}.out",
        "--error",
        f"{LOG_DIR}/{name}.err",
        *sbatch_args,
        "--wrap",
        f"tmux new-session -d -s rl && python -c 'import time; sleep_time = {sleep_time}; time.sleep(sleep_time)'",
    ]
    _log_command(sbatch_args)
    subprocess.run(sbatch_args, check=True)

    curr_job: JobInfo | None
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
    rich.print(
        f"[green]Job {curr_job.job_id} will continue running. Run [yellow]rl cancel {curr_job.job_id}[/yellow] to cancel it.[/green]"
    )


@_must_run_on_sherlock
def _get_all_jobs(partition: str | None = None, show_progress=False):
    if show_progress:
        with rich.progress.Progress(transient=True) as progress:
            # noinspection PyTypeChecker
            task = progress.add_task(
                "[green]Checking jobs in queue...[/green]", total=None
            )
            results = _get_all_jobs(partition=partition, show_progress=False)
            progress.update(task, completed=1)
            return results

    squeue_command = ["squeue", "-h", "-o", "%A %j %u %P %N %t %M %L"]
    if partition:
        squeue_command.extend(["-p", partition])
    else:
        squeue_command.extend(["-u", CURRENT_USER])

    _log_command(squeue_command)
    output = subprocess.run(
        squeue_command,
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
        (
            job_id,
            job_name,
            user,
            partition,
            nodes,
            state,
            time_elapsed,
            time_remaining,
        ) = line.split(" ")
        results.append(
            JobInfo(
                job_id=job_id,
                job_name=job_name,
                user=user,
                partition=partition,
                nodes=parse_nodes_str(nodes),
                state=state,
                time_elapsed=time_elapsed,
                time_remaining=time_remaining,
            )
        )
    return results


def _touch_file(path: Path):
    subprocess.run(["truncate", "-s", "+1", str(path)])
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
@click.option(
    "--all",
    "-a",
    is_flag=True,
    help="Cancel all running jobs",
)
@_must_run_on_sherlock
def cancel(job_id: str, yes: bool, all: bool):
    if all:
        jobs = _get_all_jobs(show_progress=True)
        if not jobs:
            rich.print("[yellow]No jobs found to cancel[/yellow]")
            return
        job_ids = [job.job_id for job in jobs]
        if yes or click.confirm(
            f"Are you sure you want to cancel {len(job_ids)} jobs?"
        ):
            for job_id in job_ids:
                command = ["scancel", job_id]
                _log_command(command)
                subprocess.run(command)
            rich.print(f"[red]Cancelled {len(job_ids)} jobs[/red]")
        return

    if not job_id:
        job_id = _select_job()
    if yes or click.confirm(f"Are you sure you want to cancel job {job_id}?"):
        command = ["scancel", job_id]
        _log_command(command)
        subprocess.run(command)
        rich.print(f"[red]Job {job_id} cancelled[/red]")


def _select_job() -> str:
    jobs = _get_all_jobs(show_progress=True)
    job_names = []
    job_name_id_map = {}
    for job in jobs:
        job_name = f"{job.job_id} ({', '.join(job.nodes)}; partition='{job.partition}'; {job.state})"
        job_names.append(job_name)
        job_name_id_map[job_name] = job.job_id
    selection = (
        questionary.select("Select a job to cancel", choices=job_names).ask()
        if len(job_names) > 1
        else job_names[0]
    )
    return job_name_id_map[selection]


# endregion


# region Connections
@cli.command(help="SSH into Sherlock or into a particular job while on Sherlock")
@click.argument("node", required=False, type=str)
def ssh(node: str):
    if not ON_SHERLOCK:
        _ssh_into_sherlock(node)
    else:
        _ssh_within_sherlock(node)


@cli.command(help="Run an SSH server on Sherlock and tunnel it to your local machine")
@click.option(
    "--host",
    "-H",
    "local_host",
    help="Local host to bind the tunnel to",
    default="localhost",
    type=str,
)
@click.option(
    "--port",
    "-p",
    "local_port",
    help="Local port to tunnel to",
    default=_DEFAULT_SSH_TUNNEL_PORT,
    type=int,
)
@click.option(
    "--remote-port",
    "-rp",
    help="Remote port to run the SSH server on",
    default=_DEFAULT_SSH_SERVER_PORT,
    type=int,
)
@_require_duo
@_require_sherlock_credentials
def tunnel(
    local_host: str,
    local_port: int,
    remote_port: int,
    *,
    credentials: Credentials,
    duo: Duo,
):
    sshd_config_path = SHERLOCK_SSH_DIR / "sshd_config"
    host_key_path = SHERLOCK_SSH_DIR / "host_rsa"

    _setup_tunnel_infra(
        sshd_config_path,
        host_key_path,
        local_port,
        remote_port,
        credentials=credentials,
        duo=duo,
    )

    server_command = f"{SSHD_PATH} -f {sshd_config_path}"
    common_args = [
        f"{credentials.username}@{credentials.node}.sherlock.stanford.edu",
    ]
    _run_sherlock_ssh(
        "ssh", common_args + [server_command], credentials=credentials, duo=duo
    )

    rich.print(
        f"[green]Tunnel to Sherlock running on {credentials.node} started on localhost:{local_port}[/green]"
    )

    rich.print(f"[green]Starting tunnel to Sherlock on {credentials.node}...[/green]")
    rich.print(f"[green]Local port: {local_port}, Remote port: {remote_port}[/green]")
    rich.print("[yellow]Press Ctrl+C to stop the tunnel[/yellow]")

    try:
        _run_sherlock_ssh(
            "ssh",
            common_args
            + ["-N", "-L", f"{local_host}:{local_port}:localhost:{remote_port}"],
            credentials=credentials,
            duo=duo,
        )
    except KeyboardInterrupt:
        rich.print("\n[red]Tunnel stopped by user[/red]")
    except Exception as e:
        rich.print(f"[red]Error: {e}[/red]")


@_require_sherlock_credentials
@_require_duo
def _setup_tunnel_infra(
    sshd_config_path: Path,
    host_key_path: Path,
    local_port: int,
    remote_port: int,
    credentials: Credentials,
    duo: Duo,
):
    tunnel_setup_path = BASE_CONFIG_DIR / ".tunnel_setup"
    if tunnel_setup_path.exists():
        return

    rich.print("[green]Running one-time setup for Sherlock tunnel...[/green]")

    key = paramiko.RSAKey.generate(bits=4096)
    private_key_path = Path.home() / ".ssh" / "sherlock_tunnel_rsa"
    key.write_private_key_file(str(private_key_path))
    public_key = f"{key.get_name()} {key.get_base64()}"

    sshd_config = {
        "Port": remote_port,
        "ListenAddress": "localhost",
        "Protocol": "2",
        "HostKey": str(host_key_path),
        "UsePrivilegeSeparation": "no",
        "PubkeyAuthentication": "yes",
        "AuthorizedKeysFile": "~/.ssh/authorized_keys",
        "PermitRootLogin": "no",
        "PasswordAuthentication": "no",
        "ChallengeResponseAuthentication": "no",
        "X11Forwarding": "no",
        "AllowUsers": credentials.username,
    }
    sshd_config_text = "\n".join(f"{k} {v}" for k, v in sshd_config.items())

    sherlock_commands = [
        f"mkdir -p {SHERLOCK_SSH_DIR}",
        f"echo '{sshd_config_text}' > {sshd_config_path}",
        f"ssh-keygen -t rsa -b 4096 -f {host_key_path} -N '' -C 'sherlock-tunnel'",
        f"echo '{public_key}' >> {SHERLOCK_SSH_DIR}/authorized_keys",
        f"chmod 600 {SHERLOCK_SSH_DIR}/authorized_keys",
    ]
    combined_command = " && ".join(sherlock_commands)
    _run_sherlock_ssh(
        "ssh",
        [
            f"{credentials.username}@{credentials.node}.sherlock.stanford.edu",
            combined_command,
        ],
        credentials=credentials,
        duo=duo,
    )

    # finally, create an `rl` entry in the user's .ssh/config
    ssh_config = paramiko.SSHConfig()
    ssh_config_path = Path.home() / ".ssh" / "config"
    ssh_config.parse(ssh_config_path.open())
    if len(ssh_config.lookup(TUNNEL_HOST_NAME)) < 2:
        with ssh_config_path.open("a") as f:
            f.write(f"""
Host {TUNNEL_HOST_NAME}
    HostName localhost
    User {credentials.username}
    IdentityFile {private_key_path}
    Port {local_port}""")

    rich.print("[green]Sherlock tunnel setup complete![/green]")
    tunnel_setup_path.touch()


@cli.command(
    help="SCP files to/from Sherlock",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("direction", type=click.Choice(["to", "from"]))
@click.argument("source", type=str)
@click.argument("destination", type=str)
@click.argument("scp_options", nargs=-1, type=str)
@_require_duo
@_require_sherlock_credentials
def scp(
    direction: str,
    source: str,
    destination: str,
    scp_options: list[str],
    *,
    credentials: Credentials,
    duo: Duo,
):
    node = credentials.node
    node_url = f"{node}.sherlock.stanford.edu"
    rich.print(f"[green]Logging in to {node_url}[/green]")
    scp_args = [*scp_options]
    if direction == "to":
        scp_args.extend([source, f"{credentials.username}@{node_url}:{destination}"])
    else:
        scp_args.extend([f"{credentials.username}@{node_url}:{source}", destination])

    _run_sherlock_ssh("scp", scp_args, credentials=credentials, duo=duo)


@_require_duo
@_require_sherlock_credentials
def _ssh_into_sherlock(node: str, *, credentials: Credentials, duo: Duo):
    node = node or credentials.node
    node_url = f"{node}.sherlock.stanford.edu"
    rich.print(f"[green]Logging you in to {node_url}...[/green]")
    ssh_args = [
        f"{credentials.username}@{node_url}",
        "-t",
        "fish || bash",
    ]
    _run_sherlock_ssh("ssh", ssh_args, credentials=credentials, duo=duo)


@_must_run_on_sherlock
def _ssh_within_sherlock(node: str):
    if not node:
        node = _select_node()
    rich.print(f"[green]SSHing into {node}...[/green]")
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


@_require_duo
@_require_sherlock_credentials
def _run_sherlock_ssh(
    ssh_command: str, args: list[str], *, credentials: Credentials, duo: Duo
):
    args = ["-o", "StrictHostKeyChecking=no", *args]

    ssh = pexpect.spawn(ssh_command, args)
    ssh.expect("password:")
    ssh.sendline(credentials.password)

    ssh.expect(r"Passcode or option \(\d+-\d+\): ")
    duo_output = ssh.before.decode()
    option_to_select = _MFA_LINE_REGEX.search(duo_output)
    if not option_to_select:
        raise RLError("Could not find Duo MFA option")
    option_number = option_to_select.group("number")
    ssh.sendline(option_number)
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


# endregion


# region Configuration


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
    credentials = Credentials(
        username=username,
        password=password,
        node=node_choice,
    )
    _write_credentials(credentials)
    print(f"Credentials saved to {CREDENTIALS_FILE}")


# endregion


# region File Operations
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


# endregion


if __name__ == "__main__":
    BASE_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    cli()
