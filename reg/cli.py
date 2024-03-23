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


SBATCH_TEMPLATE = """#!/usr/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --time={time}
#SBATCH -p {partition}
#SBATCH --gpus {gpus}
#SBATCH --cpus-per-gpu {cpus_per_gpu}
#SBATCH --mem={mem}
sleep {sleep}
"""


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
    "--cpus",
    "-c",
    help="Number of CPUs",
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
    cpus: int,
    mem: str,
    time: str,
):
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    if partition == "owners":
        create_batch_job(name, gpus, cpus, mem, time)
        return

    rich.print("[green]Starting interactive job...[/green]")
    srun_args = [
        "srun",
        "--partition",
        partition,
        "--job-name",
        name,
        "--gpus",
        str(gpus),
        "--cpus-per-task",
        str(cpus),
        "--mem",
        mem,
        "--time",
        time,
        "--pty",
        SHELL_PATH,
    ]
    # Show the output to the user
    subprocess.run(srun_args)


def create_batch_job(name: str, gpus: int, cpus: int, mem: str, job_time: str):
    parsed_time = datetime.datetime.strptime(job_time, "%H:%M:%S")
    sleep_time = parsed_time.hour * 3600 + parsed_time.minute * 60 + parsed_time.second

    sbatch_args = [
        "sbatch",
        "--job-name",
        name,
        "--output",
        LOG_DIR / f"{name}.out",
        "--error",
        LOG_DIR / f"{name}.err",
        "--time",
        job_time,
        "--partition",
        CURRENT_GROUP,
        "--gpus",
        str(gpus),
        "--cpus-per-task",
        str(cpus),
        "--mem",
        mem,
        "--wrap",
        f"sleep {sleep_time}",
    ]
    subprocess.run(sbatch_args, check=True)

    job_node = None
    with rich.progress.Progress(transient=True) as progress:
        task = progress.add_task("[green]Waiting for job to start...", total=None)
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
            job_node = job_info[7]
            if job_info[4] == "R":
                progress.update(task, completed=1)
                break

    rich.print(f"[green]Job started on node {job_node}. SSHing into node...[/green]")
    ssh_args = [
        SSH_PATH,
        job_node,
    ]
    subprocess.run(ssh_args)


if __name__ == "__main__":
    cli()
