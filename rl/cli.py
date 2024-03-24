import datetime
import json
import random
import subprocess
import time
from pathlib import Path

import click
import pexpect
import rich
import rich.progress

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


@click.group()
def cli():
    pass


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
    default=1,
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
    help="Number of CPUs per GPU",
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
                raise Exception(
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
def ssh():
    credentials = _read_credentials()
    if not _read_duo():
        print("Duo not configured. Run `rl duo` to configure it.")
        return
    if not credentials:
        CREDENTIALS_FILE.parent.mkdir(exist_ok=True, parents=True)
        print("Sherlock credentials not configured. Please enter them now.")
        username = click.prompt("Stanford NetID")
        password = click.prompt("NetID Password", hide_input=True)
        node_choice = random.choice(NODE_OPTIONS)
        credentials = {
            "username": username,
            "password": password,
            "node": node_choice,
        }

        _write_credentials(credentials)
        print(
            f"Credentials saved to {CREDENTIALS_FILE}. Edit this file to change node."
        )

    node_url = f"{credentials['node']}.sherlock.stanford.edu"
    print(f"Logging in to {node_url}")
    ssh = pexpect.spawn(f"ssh {credentials['username']}@{node_url}")
    ssh.expect("password:")
    ssh.sendline(credentials["password"])
    ssh.expect("Passcode or option")
    ssh.sendline(get_duo_code())
    ssh.interact()


@cli.command(help="Configure Duo for Sherlock")
def duo():
    if _read_duo():
        print(
            "Warning: Duo already configured. Continuing will overwrite the current configuration."
        )
    _configure_duo()
    print(f"Duo configuration saved to {DUO_FILE}")


def get_duo_code() -> str:
    import pyotp

    duo_info = _read_duo()
    if not duo_info:
        raise Exception("Duo not configured. Run `rl duo` to configure it.")
    hotp = pyotp.HOTP(duo_info["hotp_secret"])
    code = hotp.at(duo_info["count"])
    duo_info["count"] += 1
    _write_duo(duo_info)
    return code


def _read_duo() -> dict | None:
    if not DUO_FILE.exists():
        return None
    with open(DUO_FILE, "r") as f:
        return json.load(f)


def _write_duo(duo_info: dict):
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


def _configure_duo():
    from rl.duo import Duo

    qr_url = click.prompt("Enter image address of the Duo activation QR code")
    duo = Duo.from_qr_url(qr_url)
    _write_duo(duo.to_config())


if __name__ == "__main__":
    BASE_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    print(BASE_CONFIG_DIR)
    cli()
