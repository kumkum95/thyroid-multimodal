"""Push, poll and fetch the Kaggle GPU kernel.

    python scripts/kaggle_push.py            # push, then poll until it finishes
    python scripts/kaggle_push.py --status   # just report current status
    python scripts/kaggle_push.py --pull     # download the last run's output
    python scripts/kaggle_push.py --no-wait  # push and return immediately

The kaggle CLI is not on PATH in Git Bash on this machine, so the executable is
located explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import json
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


def kernel_id() -> str:
    return json.loads(METADATA.read_text(encoding="utf-8"))["id"]


def run_cli(cli: list[str], args: list[str], check: bool = True) -> str:
    cmd = cli + args
    print(f"$ kaggle {' '.join(args)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
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


def status(cli: list[str]) -> str:
    out = run_cli(cli, ["kernels", "status", kernel_id()], check=False)
    for state in ("complete", "error", "running", "queued", "cancelRequested",
                  "cancelAcknowledged"):
        if state.lower() in out.lower():
            return state
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report status only")
    ap.add_argument("--pull", action="store_true", help="download last output")
    ap.add_argument("--no-wait", action="store_true", help="push without polling")
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls")
    ap.add_argument("--timeout", type=int, default=5400, help="max seconds to wait")
    args = ap.parse_args()

    if not METADATA.exists():
        raise SystemExit(f"missing {METADATA}")

    cli = find_cli()
    kid = kernel_id()

    if args.status:
        print(f"\nstatus: {status(cli)}")
        return

    if args.pull:
        dest = ROOT / "outputs" / "kaggle"
        dest.mkdir(parents=True, exist_ok=True)
        run_cli(cli, ["kernels", "output", kid, "-p", str(dest)])
        print(f"\ndownloaded to {dest}")
        return

    run_cli(cli, ["kernels", "push", "-p", str(KERNEL_DIR)])
    print(f"\n  kernel: https://www.kaggle.com/code/{kid}")

    if args.no_wait:
        print("  not waiting; check with --status")
        return

    print(f"\n  polling every {args.poll}s (ctrl-c to stop; the run continues)")
    start = time.time()
    last = None
    while time.time() - start < args.timeout:
        time.sleep(args.poll)
        state = status(cli)
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
