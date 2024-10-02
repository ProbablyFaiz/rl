# RL — Research Utilities

RL is a Python library which provides several things:
- A set of utilities commonly required in Python research codebases, including:
  - IO utils for keeping an organized data directory and working with JSONL/CSV files
  - Scripts for training next-token LLMs with LoRA fine-tuning, as well as merging and quantizing those models.
  - "Inference engines" that provide a unified interface for both local and API-based LLM inference,
     including support for vLLM, OpenAI, Gemini, Anthropic, Together.AI, and other inference libraries/providers.
  - A `devsync` CLI for automatically syncing Git-tracked files in a codebase with a remote SSH host.
  - A drop-in replacement for the Click CLI library which provides rich formatting and an ipdb debugger
     that activates automatically when your code raises an unhandled exception (useful for saving data!).
- A CLI (`rl`) interacting with [Sherlock](https://www.sherlock.stanford.edu/) less painful
   by automating Duo 2FA and providing useful primitives for working with SLURM jobs.

## CLI Installation

> NOTE: The below is only for a stable release of the Sherlock CLI. Docs for using the rest
>  of the library are forthcoming.

The best way to install Sherlock is via `uv`, a fast Python installation/package manager.

If you don't have it already (or have an old version), install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, on both your local machine and Sherlock (or either one, if you just want some features):
```
uv tool install "rl[sherlock] @ git+https://github.com/ProbablyFaiz/rl.git@v0.9.0"
```

## Setup

> [!NOTE]
> RL requires you to configure Duo and NetID credentials to work. These **never** leave your device;
> use of RL is as secure as using an SSH key stored on your device to sign in to a remote server.
> For the same reason, you should not use RL on a shared computer or share your configuration with others.

### Duo

> *TODO: Add visual instructions for setting up Duo.*

- Go to Duo Central. Add a device with a phone number of the form `[some area code]-555-0199` (this is a
  [reserved dummy number](https://arc.net/l/quote/fbclpupw)). If you are asked to confirm the phone number via SMS,
  exit, delete the device, and try a different area code. Duo only prompts for verification if the number has
  been used previously, so you can keep trying different area codes until you find one that works.
- When Duo shows you the QR code, right click and copy its image address.
- Run `rl configure duo`. Paste the image address when prompted.

> TODO: In the future, RL should generate a valid Duo phone number for you, so you don't have to manually find one.

### Sherlock

To access Sherlock, RL also requires your NetID credentials. You can configure them by running `rl configure sherlock`.

That's it! You should now be able to use `rl ssh` without needing to enter a Duo code.

## Usage

```bash
Usage: rl [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  approve    Approve a Duo login request
  cancel     Cancel a running job
  configure  Configure different aspects of rl
  job        Create an interactive job, even on the owners partition
  scp        SCP files to/from Sherlock
  ssh        SSH into Sherlock or into a particular job while on Sherlock
  touch      Temporarily modify files to avoid Sherlock auto-deletion
```
