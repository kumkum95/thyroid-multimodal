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
import time
from pathlib import Path

REPO_URL = "https://github.com/kumkum95/thyroid-multimodal.git"
REPO_DIR = Path("/kaggle/tmp/thyroid-multimodal")
WORKING = Path("/kaggle/working")
# Scratch space that is NOT collected as kernel output. Data and the clone live
# here; only results belong in WORKING.
SCRATCH = Path("/kaggle/tmp")
OUTPUTS = WORKING / "outputs"

# Set by editing this file before `kaggle kernels push`.
STAGE = os.environ.get("DEX_STAGE", "image")
# Pin to a commit for a reproducible run; empty means current main.
COMMIT = os.environ.get("DEX_COMMIT", "")

# Arguments for STAGE="image". Sized to finish inside Kaggle's 9-hour kernel
# limit rather than to match the published settings: EfficientNet-B4 at 384px
# over 4 configs and 3 seeds is twelve trainings and does not fit.
#
# M3 and M6 first because they bracket the question that matters now. Mask
# geometry alone already reaches 0.9602 on the test split and TI-RADS alone
# 0.9287, so the open number is how much an image-only model gives up when it
# is denied both -- that is M3. M6 is the fully fused upper end, and its gates
# say where it is actually looking. M4 and M5 fill in the middle once these two
# have run and the per-epoch cost is known.
IMAGE_STAGE_ARGS = [
    "--configs", "M3", "M6",
    "--seeds", "0",
    "--backbone", "efficientnet_b0",
    "--image-size", "256",
    "--batch-size", "8",
    "--epochs", "20",
    "--max-images", "6",
    "--num-workers", "2",
]

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


def hf_token() -> str | None:
    """Read the Hugging Face token from Kaggle Secrets, never from the code.

    ThyroidXL is gated: access is granted per account after accepting its
    terms. Fetching it at run time under that account keeps this project from
    redistributing it, which uploading a copy as a Kaggle Dataset would do.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"  HF_TOKEN from environment ({len(token)} chars)")
        return token

    # An attached Kaggle Dataset, preferred over Secrets. `kaggle kernels push`
    # detaches a kernel's Secrets but keeps dataset_sources, because those are
    # declared in kernel-metadata.json -- so a secret has to be re-attached by
    # hand after every push, while a dataset does not.
    for name in ("hf_token.txt", "token.txt", "hf_token"):
        for path in sorted(Path("/kaggle/input").glob(f"*/{name}")):
            value = path.read_text(encoding="utf-8").strip()
            if value:
                print(f"  HF_TOKEN from {path} ({len(value)} chars)")
                return value

    # Retried: the secrets backend is a network call and a single ConnectionError
    # is not evidence that the secret is absent.
    for attempt in range(1, 4):
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token:
                print(f"  HF_TOKEN from Kaggle secrets ({len(token)} chars)")
                return token
            print("  HF_TOKEN secret is present but empty")
            return None
        except Exception as exc:
            # The class name alone cannot distinguish "secret not attached to
            # this notebook" from "backend unreachable", so print the message.
            print(f"  secrets attempt {attempt}: {type(exc).__name__}: "
                  f"{str(exc)[:300]}")
            time.sleep(3 * attempt)
    return None


def fetch_thyroidxl(target: Path) -> bool:
    """Download images, masks and metadata from Hugging Face into `target`."""
    token = hf_token()
    if not token:
        return False

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    patterns = ["train/images/*", "test/images/*",
                "train/masks/*", "test/masks/*",
                "stats/*"]
    for attempt in range(1, 5):
        try:
            snapshot_download(
                "VNPT-AI/ThyroidXL", repo_type="dataset",
                allow_patterns=patterns, local_dir=str(target),
                max_workers=16, token=token,
            )
            break
        except Exception as exc:
            print(f"  attempt {attempt}: {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(10 * attempt)
    else:
        print("  !! ThyroidXL download failed after 4 attempts")
        return False

    # The metadata CSVs are built here rather than committed, because the repo
    # is public and the CSVs carry per-patient clinical fields.
    run([sys.executable, "-c",
         "import sys; sys.path.insert(0, '.');"
         "from src.thyroidxl import build_table;"
         f"[build_table(s, meta_dir={str(target)!r},"
         f" image_root=f'{{s}}/images').to_csv("
         f"  f'{target}/{{s}}.csv', index=False) for s in ('train','test')]"],
        cwd=REPO_DIR)
    counts = {d.name: len(list(d.glob('*')))
              for d in (target / "train" / "images", target / "test" / "images")
              if d.exists()}
    print(f"  ThyroidXL fetched into {target}  {counts}")
    return True


def resolve_data() -> tuple[Path, bool]:
    """Return (data_dir, is_real). Falls back to the synthetic cohort."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*.csv")):
            print(f"  using attached dataset: {candidate}")
            return candidate, True

    # Not under WORKING: everything there becomes kernel output, so a 6.87 GB
    # cohort makes `kaggle kernels output` drag the entire dataset back before
    # it will hand over the log.
    target = SCRATCH / "data" / "thyroidxl"
    if fetch_thyroidxl(target):
        return target, True

    print("  no ThyroidXL available -- generating synthetic cohort")
    print("  (synthetic results validate the pipeline, never the science)")
    target = SCRATCH / "data" / "thyroidxl_synthetic"
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
        if not is_real:
            # Six GPU-hours spent on deterministic noise would still produce a
            # results.json full of plausible-looking AUCs. Fail now instead.
            raise SystemExit(
                "Refusing to run the image stage on the synthetic cohort. "
                "ThyroidXL is gated, so the kernel needs a Hugging Face "
                "token from an account that accepted its terms. Attach a "
                "private Kaggle Dataset containing hf_token.txt and list it in "
                "kernel-metadata.json dataset_sources -- Kaggle Secrets are "
                "detached by every `kernels push`, so they do not survive this "
                "workflow. Set DEX_STAGE=week1 for synthetic data instead."
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
                *IMAGE_STAGE_ARGS,
            ],
            cwd=REPO_DIR,
        )

    elif STAGE == "probe":
        # Credentials and dataset reachability only. No training, no GPU time.
        token = hf_token()
        print(f"\n  token found: {bool(token)}")
        if token:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            print(f"  hf identity: {api.whoami().get('name')}")
            files = api.list_repo_files("VNPT-AI/ThyroidXL", repo_type="dataset")
            print(f"  ThyroidXL readable: {len(files)} files")
            print(f"  sample: {sorted(files)[:3]}")

    else:
        raise SystemExit(
            f"unknown STAGE: {STAGE!r} (expected 'probe', 'week1' or 'image')"
        )

    print("\n" + "=" * 70)
    print("ARTIFACTS IN /kaggle/working/outputs")
    print("=" * 70)
    for path in sorted(OUTPUTS.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(OUTPUTS)}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
