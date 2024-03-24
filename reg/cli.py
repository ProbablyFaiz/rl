import datetime
import subprocess
import time
from pathlib import Path

import click
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


@click.group()
def cli():
    pass


@cli.command(help="Create an interactive job")
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


@cli.command(
    help="Modify files or directories (and then undo) so that Sherlock won't delete them"
)
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


if __name__ == "__main__":
    cli()
