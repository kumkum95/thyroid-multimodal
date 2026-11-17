"""Kaggle GPU kernel entry point for DEX-Thyroid.

Pushed with `python scripts/kaggle_push.py`. Runs headlessly on Kaggle's GPU
runtime and writes everything to /kaggle/working, which Kaggle keeps as the
kernel's output.

What it does, in order:
  1. Reports the runtime (GPU model, VRAM, torch/CUDA versions) so a run's
     hardware is recorded in its own log rather than assumed.
  2. Clones the public project repo. The code lives in git, not pasted into a
     notebook cell, so a kernel run is reproducible from a commit SHA.
  3. Resolves a data source: a mounted Kaggle Dataset if one is attached,
     otherwise the synthetic cohort.
  4. Runs the stage named by STAGE.

Stage "week1" is CPU-only tabular work. It is here because it proves the whole
chain -- clone, import, split, evaluate, write outputs -- works on Kaggle before
any GPU training depends on it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/kumkum95/thyroid-multimodal.git"
REPO_DIR = Path("/kaggle/working/thyroid-multimodal")
WORKING = Path("/kaggle/working")
OUTPUTS = WORKING / "outputs"

# Set by editing this file before `kaggle kernels push`.
STAGE = os.environ.get("DEX_STAGE", "week1")
# Pin to a commit for a reproducible run; empty means current main.
COMMIT = os.environ.get("DEX_COMMIT", "")

# Candidate mount points for the real cohort, in order of preference. Kaggle
# mounts attached datasets read-only under /kaggle/input/<slug>.
DATA_CANDIDATES = [
    Path("/kaggle/input/thyroidxl"),
    Path("/kaggle/input/thyroid-xl"),
    Path("/kaggle/input/thyroidxl-dataset"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def report_runtime() -> dict:
    """Record the hardware. A result without its runtime is not reproducible."""
    info: dict = {"python": sys.version.split()[0]}
    print("=" * 70)
    print("RUNTIME")
    print("=" * 70)
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        print(f"  torch            {torch.__version__}")
        print(f"  cuda available   {info['cuda_available']}")
        print(f"  cuda version     {torch.version.cuda}")
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / 1024**3, 1)
                info["gpus"].append({"name": props.name, "vram_gb": vram_gb})
                print(f"  gpu[{i}]          {props.name}  {vram_gb} GB")
        else:
            # Loud, because a GPU kernel that silently fell back to CPU will
            # look like it worked and take 40x longer.
            print("  !! NO GPU VISIBLE -- check Accelerator is set on the kernel")
    except ImportError:
        print("  torch not importable")
        info["torch"] = None
    print()
    return info


def clone_repo() -> str:
    if REPO_DIR.exists():
        run(["git", "-C", str(REPO_DIR), "fetch", "--all", "--quiet"])
    else:
        run(["git", "clone", "--quiet", REPO_URL, str(REPO_DIR)])
    if COMMIT:
        run(["git", "-C", str(REPO_DIR), "checkout", "--quiet", COMMIT])
    sha = subprocess.run(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"  repo at {sha}\n")
    return sha


def resolve_data() -> tuple[Path, bool]:
    """Return (data_dir, is_real). Falls back to the synthetic cohort."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*.csv")):
            print(f"  using attached dataset: {candidate}")
            return candidate, True

    print("  no ThyroidXL dataset attached -- generating synthetic cohort")
    print("  (synthetic results validate the pipeline, never the science)")
    target = WORKING / "data" / "thyroidxl_synthetic"
    run([sys.executable, "scripts/make_synthetic.py", "--out", str(target)],
        cwd=REPO_DIR)
    return target, False


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    runtime = report_runtime()
    sha = clone_repo()
    data_dir, is_real = resolve_data()

    manifest = {
        "stage": STAGE,
        "commit": sha,
        "data_dir": str(data_dir),
        "data_is_real": is_real,
        "runtime": runtime,
    }
    (OUTPUTS / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\n  manifest -> {OUTPUTS / 'run_manifest.json'}\n")

    print("=" * 70)
    print(f"STAGE: {STAGE}")
    print("=" * 70)

    if STAGE == "week1":
        # Tabular, CPU-bound. Present so the chain is verified on Kaggle before
        # GPU training is layered on top.
        run(
            [
                sys.executable, "scripts/run_week1.py",
                "--data", str(data_dir),
                "--out", str(OUTPUTS / "week1"),
                "--n-boot", "2000",
            ],
            cwd=REPO_DIR,
        )
        run([sys.executable, "scripts/test_guards.py", "--data", str(data_dir)],
            cwd=REPO_DIR)

    elif STAGE == "image":
        # Wired up when src/image.py lands (M3-M6).
        script = REPO_DIR / "scripts" / "run_image.py"
        if not script.exists():
            raise SystemExit(
                "scripts/run_image.py does not exist yet at this commit. "
                "Build the M3 image branch first, or run STAGE=week1."
            )
        if not runtime.get("cuda_available"):
            raise SystemExit(
                "Refusing to train the image branch without a GPU: "
                "EfficientNet-B4 over ~9.5k images is not viable on CPU. "
                "Set the kernel Accelerator to GPU."
            )
        run(
            [
                sys.executable, "scripts/run_image.py",
                "--data", str(data_dir),
                "--out", str(OUTPUTS / "image"),
            ],
            cwd=REPO_DIR,
        )

    else:
        raise SystemExit(f"unknown STAGE: {STAGE!r} (expected 'week1' or 'image')")

    print("\n" + "=" * 70)
    print("ARTIFACTS IN /kaggle/working/outputs")
    print("=" * 70)
    for path in sorted(OUTPUTS.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(OUTPUTS)}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
