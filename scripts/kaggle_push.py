"""Push, poll and fetch the Kaggle GPU kernel.

    python scripts/kaggle_push.py            # push, then poll until it finishes
    python scripts/kaggle_push.py --status   # just report current status
    python scripts/kaggle_push.py --pull     # download the last run's output
    python scripts/kaggle_push.py --no-wait  # push and return immediately
    python scripts/kaggle_push.py --whoami   # which account is active

Multiple Kaggle accounts
------------------------
The kaggle CLI has no profile switching and `kaggle auth login` overwrites the
stored credentials, so accounts are kept in separate config directories and
selected with --profile:

    ~/.kaggle            default profile
    ~/.kaggle-<name>     --profile <name>

To add an account:

    KAGGLE_CONFIG_DIR=~/.kaggle-second kaggle auth login     # browser, OAuth
    # or drop that account's kaggle.json into ~/.kaggle-second/

Before pushing, the owner in kernel-metadata.json is checked against the
authenticated account: pushing to someone else's namespace fails with an opaque
error, so it is caught here instead.

The kaggle CLI is not on PATH in Git Bash on this machine, so the executable is
located explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "kaggle"
METADATA = KERNEL_DIR / "kernel-metadata.json"

# Where the kaggle CLI actually lives, most likely first.
CLI_CANDIDATES = [
    Path.home() / "AppData/Local/Python/pythoncore-3.14-64/Scripts/kaggle.exe",
    Path.home() / "AppData/Local/Programs/Python/Python39/Scripts/kaggle.exe",
    Path("/usr/local/bin/kaggle"),
]

TERMINAL_STATES = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def find_cli() -> list[str]:
    for candidate in CLI_CANDIDATES:
        if candidate.exists():
            return [str(candidate)]
    # Fall back to the module, which works whenever the package is importable
    # even if no console script was placed on PATH.
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True, check=True,
        )
        return [sys.executable, "-m", "kaggle"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "kaggle CLI not found. Install with `pip install kaggle`, then "
            "authenticate with `kaggle auth login`."
        )


def config_dir(profile: str | None) -> Path:
    """Config directory for a profile. None means the default ~/.kaggle."""
    return Path.home() / (".kaggle" if not profile else f".kaggle-{profile}")


def build_env(profile: str | None) -> dict[str, str]:
    """Environment that pins the kaggle CLI to one account.

    KAGGLE_CONFIG_DIR alone is not sufficient. kagglesdk resolves the access
    token first, and reads it from a *hard-coded* ~/.kaggle/access_token that
    ignores KAGGLE_CONFIG_DIR:

        access_token = _get_access_token_from_file(
            os.path.expanduser("~/.kaggle/access_token"))

    Since authenticate() tries the access token before the legacy API key and
    before OAuth credentials, a stale ~/.kaggle/access_token silently wins over
    whatever the selected profile holds. The only override earlier in that chain
    is the KAGGLE_API_TOKEN variable, so a profile that owns an access_token is
    pinned through it, and a profile that does not is refused rather than
    quietly running as the wrong account.
    """
    env = dict(os.environ)
    cfg = config_dir(profile)

    # An inherited KAGGLE_API_TOKEN outranks every credential file, including
    # ~/.kaggle/access_token, and a stale value there authenticates as the wrong
    # account or silently falls through to OAuth. The variable is therefore
    # always dropped first and re-set below only when this profile owns a token.
    env.pop("KAGGLE_API_TOKEN", None)

    if not profile:
        default_token = Path.home() / ".kaggle" / "access_token"
        if default_token.exists():
            env["KAGGLE_API_TOKEN"] = str(default_token)
        return env

    if not cfg.exists():
        raise SystemExit(
            f"profile {profile!r} has no config directory at {cfg}.\n"
            f"Create it with:\n"
            f"  mkdir -p {cfg}\n"
            f"  KAGGLE_CONFIG_DIR={cfg} kaggle auth login\n"
            f"or copy that account's kaggle.json into it."
        )
    env["KAGGLE_CONFIG_DIR"] = str(cfg)

    profile_token = cfg / "access_token"
    shadow = Path.home() / ".kaggle" / "access_token"
    if profile_token.exists():
        env["KAGGLE_API_TOKEN"] = str(profile_token)
    elif shadow.exists():
        raise SystemExit(
            f"\n{shadow} would override profile {profile!r}.\n"
            f"kagglesdk reads that path directly, ignoring KAGGLE_CONFIG_DIR, "
            f"and tries it before any other credential.\n\n"
            f"Fix one of these:\n"
            f"  - move it into the profile that owns it:\n"
            f"      mv {shadow} <that profile dir>/access_token\n"
            f"  - or delete it if it is stale:\n"
            f"      rm {shadow}"
        )
    return env


def read_metadata() -> dict:
    return json.loads(METADATA.read_text(encoding="utf-8"))


def kernel_id() -> str:
    return read_metadata()["id"]


def run_cli(cli: list[str], args: list[str], env: dict[str, str],
            check: bool = True) -> str:
    cmd = cli + args
    print(f"$ kaggle {' '.join(args)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    # The CLI prints an upgrade notice to stderr on every call; it is not an error.
    cleaned = "\n".join(
        line for line in out.splitlines()
        if "outdated `kaggle` version" not in line
    ).strip()
    if cleaned:
        print(cleaned, flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle CLI failed (exit {proc.returncode})")
    return cleaned


# The kaggle package prints a version-upgrade notice to stdout on import, so the
# answer is tagged with a sentinel rather than read off a bare line.
IDENTITY_TAG = "__KAGGLE_IDENTITY__="

RESOLVE_IDENTITY = f"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("{IDENTITY_TAG}" + str(api.config_values.get("username") or ""))
"""


def active_username(cli: list[str], env: dict[str, str]) -> str | None:
    """The account the API actually authenticates as.

    `kaggle config view` is NOT a usable identity check: it prints stored config
    values, so it echoes the username out of credentials.json without ever
    authenticating, and keeps reporting the old account after the credentials
    have been changed. Running authenticate() resolves the real precedence
    chain (access token -> legacy API key -> OAuth) and introspects the token
    against the server.
    """
    proc = subprocess.run(
        [sys.executable, "-c", RESOLVE_IDENTITY],
        capture_output=True, text=True, env=env,
    )
    for line in (proc.stdout or "").splitlines():
        if line.startswith(IDENTITY_TAG):
            return line[len(IDENTITY_TAG):].strip() or None
    return None


def check_owner(cli: list[str], env: dict[str, str]) -> str:
    """Fail early if the kernel namespace is not the authenticated account."""
    kid = kernel_id()
    owner = kid.split("/", 1)[0]
    user = active_username(cli, env)
    if user and user != owner:
        raise SystemExit(
            f"\nkernel-metadata.json targets '{kid}' but you are authenticated "
            f"as '{user}'.\n"
            f"Kaggle will not let you push into another account's namespace.\n\n"
            f"Fix one of these:\n"
            f"  - set \"id\": \"{user}/{kid.split('/', 1)[1]}\" in "
            f"kaggle/kernel-metadata.json\n"
            f"  - or select the right account with --profile"
        )
    return user or owner


def status(cli: list[str], env: dict[str, str]) -> str:
    out = run_cli(cli, ["kernels", "status", kernel_id()], env, check=False)
    for state in ("complete", "error", "running", "queued", "cancelRequested",
                  "cancelAcknowledged"):
        if state.lower() in out.lower():
            return state
    return "unknown"


def pull(cli: list[str], env: dict[str, str], attempts: int = 4) -> None:
    """Download kernel output, retrying on dropped connections.

    Kaggle regularly closes the connection part-way through a multi-file
    download, which leaves truncated zero-byte files behind. Retrying is the
    only remedy, so it is built in rather than left to the caller.
    """
    dest = ROOT / "outputs" / "kaggle"
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        out = run_cli(cli, ["kernels", "output", kernel_id(), "-p", str(dest)],
                      env, check=False)
        dropped = any(
            marker in out
            for marker in ("Connection aborted", "RemoteDisconnected",
                           "Read timed out", "ConnectionError")
        )
        empty = [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size == 0]
        if not dropped and not empty:
            print(f"\ndownloaded to {dest}")
            return
        for p in empty:  # a truncated file would otherwise be read as valid
            p.unlink()
        if attempt < attempts:
            wait = 5 * attempt
            print(f"  incomplete download (attempt {attempt}/{attempts}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    print(f"\n  download still incomplete after {attempts} attempts.\n"
          f"  Files on Kaggle are fine; it is the transfer that drops.\n"
          f"  Grab what you need from the kernel's Output tab in the browser.")


STAGES_NEEDING_GPU = {"image"}


def set_stage(stage: str) -> None:
    """Pin the kernel's stage and request a GPU only if that stage needs one.

    Kaggle script kernels take no environment variables, so the stage lives in
    run_kaggle.py and has to be rewritten before the push. Asking for a GPU
    also puts the run in the GPU queue, where a sixty-second credentials probe
    can wait longer than it runs -- so enable_gpu follows the stage.
    """
    driver = KERNEL_DIR / "run_kaggle.py"
    text = driver.read_text(encoding="utf-8")
    pattern = re.compile(r'^STAGE = os\.environ\.get\("DEX_STAGE", "[a-z0-9_]+"\)$',
                         re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"could not find the STAGE line in {driver}")
    driver.write_text(
        pattern.sub(f'STAGE = os.environ.get("DEX_STAGE", "{stage}")', text),
        encoding="utf-8",
    )

    meta = read_metadata()
    wants_gpu = stage in STAGES_NEEDING_GPU
    if meta.get("enable_gpu") != wants_gpu:
        meta["enable_gpu"] = wants_gpu
        METADATA.write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    print(f"  stage '{stage}'  gpu={wants_gpu}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None,
                    help="account profile; uses ~/.kaggle-<profile> as config dir")
    ap.add_argument("--whoami", action="store_true", help="show the active account")
    ap.add_argument("--status", action="store_true", help="report status only")
    ap.add_argument("--pull", action="store_true", help="download last output")
    ap.add_argument("--no-wait", action="store_true", help="push without polling")
    ap.add_argument("--stage", default=None,
                    choices=["probe", "week1", "image"],
                    help="stage to run; also decides whether a GPU is requested")
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls")
    ap.add_argument("--timeout", type=int, default=5400, help="max seconds to wait")
    args = ap.parse_args()

    if not METADATA.exists():
        raise SystemExit(f"missing {METADATA}")

    cli = find_cli()
    env = build_env(args.profile)
    kid = kernel_id()

    if args.whoami:
        user = active_username(cli, env)
        print(f"\n  config dir : {config_dir(args.profile)}")
        print(f"  account    : {user or '<not authenticated>'}")
        print(f"  kernel id  : {kid}")
        if user and user != kid.split('/', 1)[0]:
            print(f"  !! mismatch: kernel namespace is '{kid.split('/', 1)[0]}'")
        return

    if args.status:
        print(f"\nstatus: {status(cli, env)}")
        return

    if args.pull:
        pull(cli, env)
        return

    user = check_owner(cli, env)
    print(f"  pushing as '{user}'")
    if args.stage:
        set_stage(args.stage)
    run_cli(cli, ["kernels", "push", "-p", str(KERNEL_DIR)], env)
    print(f"\n  kernel: https://www.kaggle.com/code/{kid}")

    if args.no_wait:
        print("  not waiting; check with --status")
        return

    print(f"\n  polling every {args.poll}s (ctrl-c to stop; the run continues)")
    start = time.time()
    last = None
    while time.time() - start < args.timeout:
        time.sleep(args.poll)
        state = status(cli, env)
        if state != last:
            print(f"  [{int(time.time() - start)}s] {state}")
            last = state
        if state in TERMINAL_STATES:
            if state == "complete":
                print("\n  complete. fetch output with: "
                      "python scripts/kaggle_push.py --pull")
            else:
                print(f"\n  finished in state {state!r}; check the kernel log")
            return
    print(f"\n  still running after {args.timeout}s; check with --status")


if __name__ == "__main__":
    main()
