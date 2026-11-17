# DEX-Thyroid

Disentangled **EX**ogenous/endogenous clinical fusion for thyroid ultrasound.

## What this project claims

Not a higher AUC. The best published model on ThyroidXL already reports 0.9939
patient-level AUC, so there is no headroom to chase. This project measures,
honestly, **what the clinical modality actually contributes** — and the answer
may well be "less than the literature implies."

Four gaps in the existing work, each mapped to an experiment:

| Gap | What nobody has done | Experiment |
|---|---|---|
| 1 | Used ThyroidXL's age and sex at all | M1, M4 |
| 2 | Reported a clinical-only baseline | **M0, M2** |
| 3 | Separated image-derived TI-RADS from exogenous demographics | M4 vs M5 |
| 4 | Checked performance by sex / age band | subgroup tables |
| 5 | Tested whether a *fusion* model transports across centres | external validation |
| 6 | Asked whether reported SOTA is inflated by mask shortcuts | controls A–D |

## Why the design is deliberately plain

A reviewer's first question is "is your gain from fusion, or from a fancier
backbone?" So: standard backbone, tiny clinical branches, and all the rigour in
the protocol. Every configuration differs from its neighbour by exactly one
input.

## Experiment grid

| ID | Inputs | Answers |
|---|---|---|
| M0 | TI-RADS only | The baseline nobody ran. If ~0.90, half the field's claims collapse |
| M1 | age + sex | Do bare demographics carry signal? |
| M2 | age + sex + TI-RADS | **Clinical-only baseline → headline** |
| M3 | image | Standard baseline, reproduces published 0.921–0.957 |
| M4 | image + age + sex | **Cleanest fusion claim in this literature** |
| M5 | image + TI-RADS | Replicates the Bijlani preprint (0.965) |
| M6 | image + age + sex + TI-RADS | Full model |

The two comparisons that carry the argument: **M4 vs M3** (uncontaminated
exogenous gain) and **M6 vs M2** (does the image beat the clinic?).

## Protocol rules, enforced in code

- **Patient-level splits only.** With ~2.8 images per patient, an image-level
  split leaks the same nodule across the boundary and inflates AUC by several
  points. `splits.assert_no_patient_overlap` is called after every split.
- **Test split touched once.** Thresholds come from out-of-fold *training*
  predictions, never from test.
- **Bootstraps resample patients, not images.** Resampling images treats 2.8
  correlated views as independent and produces CIs that are too narrow.
- **Feature statistics fitted on train only** (age mean/std, TI-RADS encoding),
  reused unchanged on every external cohort.
- **TI-RADS uses thermometer encoding** — respects the ordering, unlike one-hot;
  stays non-linear in risk, unlike a raw scalar.
- **Masks are never a model input.** They are used only in the control
  experiments below.

## Anti-shortcut controls

The current ThyroidXL SOTA (0.9939) feeds a *segmentation-masked image* as its
second branch, while RCAF lists "mask-related shortcut analysis" as a gap it set
out to address. Those two facts together are worth testing:

- **A — mask-only model.** Train on the mask alone, zero pixel content. A high
  AUC here means masks leak the label.
- **B — mask geometry probe.** Logistic regression on area, perimeter,
  circularity, centroid. Identifies *which* property leaks.
- **C — caliper audit.** TN3K and DDTI carry burned-in diameter marks. Quantify
  prevalence by class; inpaint and re-evaluate.
- **D — label permutation.** Shuffle labels, retrain, confirm AUC → 0.5.

## Cohorts

| Role | Cohort | Size | Clinical fields |
|---|---|---|---|
| Train / internal test | ThyroidXL | 11,545 img / 4,093 pts | age, sex, TI-RADS, pathology |
| External 1 | DDTI | ~300–400 cases | age, sex, TI-RADS, + Bethesda |
| External 2 | Stanford AIMI Cine-clip | 167 pts (**only 17 malignant**) | age, sex, ACR TI-RADS, histopath |
| External 3 | Sci Data 2024 (figshare) | 8,508 img / 842 cases | metadata CSV — **fields unverified** |

The clinical vector is fixed at `{age, sex, TI-RADS}` because that is the
**intersection** of fields available in training and in every external cohort.
Reaching for anything richer breaks external validation.

Official ThyroidXL split: train 3,275 pts / 9,541 img · test 739 pts / 2,094 img.

## Layout

```
src/schema.py      canonical columns, sex/TI-RADS normalisation, validation
src/data.py        cohort loaders, column sniffing, coverage reports
src/splits.py      patient-level StratifiedGroupKFold + overlap assertions
src/metrics.py     bootstrap AUC CIs, DeLong test, ECE, reliability curves
src/baselines.py   M0 / M1 / M2 clinical-only models
scripts/make_synthetic.py   ThyroidXL-shaped synthetic cohort
scripts/run_week1.py        the M0/M1/M2 experiment
```

## Running it

```bash
pip install -r requirements.txt

# Pipeline check on synthetic data (real release is gated on Hugging Face)
python scripts/make_synthetic.py
python scripts/run_week1.py --data data/thyroidxl_synthetic --n-boot 500

# Protocol guards -- run after any change to src/
python scripts/test_guards.py

# Once the real release is downloaded
python scripts/run_week1.py --data data/thyroidxl
```

`scripts/test_guards.py` asserts the 15 properties that stop this project
reproducing the field's mistakes: sex encoding is not silently inverted, CV folds
are patient-disjoint, the overlap guard actually fires, a patient with two
labels is rejected, each config sees only its allowed fields, DeLong on
identical scores gives p=1, and permuted labels give mean AUC 0.5.

> **On control D:** label permutation must be averaged over many permutations,
> not run once. AUC is rank-based and scale-invariant, so after permutation the
> fitted coefficients are near zero but the *ordering* of predictions is still
> fixed by their signs. A single shuffle draws from a wide distribution — we
> measured 0.255 to 0.783 across 25 permutations, mean 0.510. Only the mean is
> pinned to 0.5. Papers that report one shuffle are reporting noise.

> **On the synthetic cohort:** it reproduces ThyroidXL's published *marginals*
> (patient and image counts, 89.2% female, 47.8% prevalence), but the
> *associations* between variables are assumptions written by hand in
> `make_synthetic.py`. Synthetic runs validate the code, never the science.
> Outputs are stamped `SYNTHETIC` so they cannot be mistaken for findings.

## Adapting the loader to the real release

The ThyroidXL column names have not been inspected (gated repo). On first
download:

```python
import pandas as pd
from src.data import sniff_columns
raw = pd.read_csv("data/thyroidxl/train.csv")
print(sniff_columns(raw))     # proposes canonical -> source mapping
```

Eyeball the proposal, then hard-code it. Two things to verify on day one:

1. Whether the "text" modality on the dataset card is report text or just the
   metadata CSV. **Real report text would upgrade this project substantially** —
   add a vision-language branch and it becomes comparable to ThyroFusion/TTM-Net.
2. Whether nodule size is present. If yes, add it — but confirm DDTI and
   Stanford have it too, or external validation breaks.

## Hardware and Kaggle GPU

Local `torch` is CPU-only. M0–M2 are tabular and run in seconds. M3–M6 need an
EfficientNet-B4 over ~9.5k images, so they run on Kaggle's GPU runtime.

```bash
python scripts/kaggle_push.py             # push and poll until done
python scripts/kaggle_push.py --status    # status only
python scripts/kaggle_push.py --pull      # fetch output to outputs/kaggle/
```

Kernel: `choudhary15/dex-thyroid-train` (private, GPU + internet enabled).

`kaggle/run_kaggle.py` clones this repo rather than carrying pasted notebook
cells, so every run is reproducible from a commit SHA. Set `DEX_COMMIT` to pin
one. It writes `outputs/run_manifest.json` recording the GPU model, VRAM and
torch/CUDA versions — a result without its runtime is not reproducible, and a
GPU kernel that silently fell back to CPU otherwise just looks slow.

Stages: `week1` (tabular, verifies the whole chain on Kaggle) and `image`
(M3–M6, **refuses to run without a visible GPU**).

### Getting ThyroidXL onto Kaggle

The kernel prefers an attached Kaggle Dataset and falls back to the synthetic
cohort. Two ways to supply the real data:

1. **Upload as a private Kaggle Dataset** (recommended). Download from Hugging
   Face locally, then:
   ```bash
   kaggle datasets init -p data/thyroidxl
   # edit data/thyroidxl/dataset-metadata.json -> slug "thyroidxl"
   kaggle datasets create -p data/thyroidxl --dir-mode zip
   ```
   Then add `"choudhary15/thyroidxl"` to `dataset_sources` in
   `kaggle/kernel-metadata.json`.

2. **Download in-kernel from Hugging Face.** ThyroidXL is gated, so store your
   HF token as a Kaggle Secret (Add-ons → Secrets) rather than putting it in
   code, and read it with `UserSecretsClient`.

Keep the dataset **private** either way: you accepted ThyroidXL's terms as an
individual, and re-publishing it would likely breach them.

> Kaggle free GPU quota is ~30 h/week, and a session is capped at 12 h. Budget
> M3–M6 × 5 seeds accordingly, and checkpoint so a timeout is resumable.

## Comparison targets

| Paper | Venue | ThyroidXL result |
|---|---|---|
| BioMedCLIP dual-branch | IEEE UBMK 2026 | **AUC 0.9939**, F1 0.9699 |
| TAZD-Net | Discover AI 2026 | patient AUROC 0.9576, Dice 0.8825 |
| RCAF | Scientific Reports 2026 | protocol template (leakage-free, calibrated) |
| Bijlani (preprint) | Zenodo 2026 | image 0.921 → +TI-RADS 0.965 |
