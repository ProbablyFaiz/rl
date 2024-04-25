# RL — RegLab / Sherlock utilities

RL is a command-line tool which makes interacting with [Sherlock](https://www.sherlock.stanford.edu/) less painful.

## Installation

On your local machine:
```bash
pip install "rl @ git+https://github.com/ProbablyFaiz/rl.git@v0.4.1"
```

On Sherlock, you should probably use the pre-compiled binary instead:
```bash
wget "https://github.com/ProbablyFaiz/rl/releases/download/v0.4.1/rl" -O ~/.local/bin/rl
```


## Setup

> [!NOTE]
> RL requires you to configure Duo and NetID credentials to work. These **never** leave your device;
> use of RL is as secure as using an SSH key stored on your device to sign in to a remote server.
> For the same reason, you should not use RL on a shared computer or share your configuration with others.

### Duo

> *TODO: Add visual instructions for setting up Duo.*

- Go to Duo Central. Add a device with a phone number of the form `[some area code]-555-0199` (this is a
  [reserved fake number](https://arc.net/l/quote/fbclpupw)). If you are asked to confirm the phone number via SMS,
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
