# RL — RegLab / Sherlock utilities

## Installation

On your local machine:
```bash
pip install "rl @ git+https://github.com/ProbablyFaiz/rl.git"
```

On Sherlock, you should probably use the pre-compiled binary instead:
```bash
wget "https://github.com/ProbablyFaiz/rl/releases/download/v0.3.1/rl-sherlock" -O ~/.local/bin/rl
```


## Setup

> [!NOTE]
> RL requires you to configure Duo and NetID credentials to work. These **never** leave your device;
> use of RL is as secure as using an SSH key stored on your device to sign in to a remote server.
> For the same reason, you should not use RL on a shared computer or share your configuration with others.

### Duo

> *TODO: Add visual instructions for setting up Duo.*

- Go to Duo Central. Add a device with the phone number `805-555-0199` (this is a [reserved fake number](https://arc.
  net/l/quote/fbclpupw)). When it shows the QR code, right click and copy its image address.
- Run `rl configure duo`. Paste the image address when prompted.


### Sherlock

To access Sherlock, RL also requires your NetID credentials. You can configure them by running `rl configure sherlock`.

That's it! You should now be able to use `rl ssh` without needing to enter a Duo code.

## Usage

```bash
Usage: rl [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  touch      Temporarily modify files to avoid Sherlock auto-deletion
  job        Create an interactive job, even on the owners partition
  scp        SCP files to/from Sherlock
  ssh        SSH into Sherlock
  configure  Configure different aspects of rl
```
