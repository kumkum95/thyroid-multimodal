# DEX-Thyroid

Disentangled **EX**ogenous/endogenous clinical fusion for thyroid ultrasound.

## What this project claims

Not a higher AUC. The best published model on ThyroidXL already reports 0.9939
patient-level AUC, so there is no headroom to chase. This project measures,
honestly, **what the clinical modality actually contributes** — and the answer
may well be "less than the literature implies."

Six gaps in the existing work, each mapped to an experiment:

| Gap | What nobody has done | Experiment |
|---|---|---|
| 1 | Used ThyroidXL's age and sex at all | M1, M4 |
| 2 | Reported a clinical-only baseline | **M0, M2** |
| 3 | Separated image-derived TI-RADS from exogenous demographics | M4 vs M5 |
| 4 | Checked performance by sex / age band | subgroup tables |
| 5 | Tested whether a *fusion* model transports across centres | external validation |
| 6 | Asked whether reported SOTA is inflated by mask shortcuts | controls A–D |

## Results so far

Patient level, ThyroidXL official test split (739 patients, 47.8% malignant),
every model fitted on the 3354 train patients. Published comparators in italics.

| Model | Inputs | Test AUC | Reads pixels? |
|---|---|---|---|
| M1 | age + sex | 0.6186 [0.584, 0.659] | no |
| *published* | *image only, EfficientNet-B0* | *0.921* | yes |
| M0 | TI-RADS only | 0.9287 [0.912, 0.945] | no |
| M2 | age + sex + TI-RADS | 0.9404 [0.924, 0.955] | no |
| M6 | image + age + sex + TI-RADS | 0.9368 (ECE 0.1912) | yes |
| M2 | — see above — | 0.9404 | no |
| M3 | image only | 0.9421 (ECE 0.0541) | yes |
| Control B | mask geometry only | 0.9602 [0.947, 0.973] | no |
| — | mask geometry + TI-RADS | **0.9858** | **no** |
| *published* | *BioMedCLIP, image + mask* | *0.9939* | yes + mask |

Three things follow, none of which required training a vision model:

1. TI-RADS alone (0.9287) beats the published image-only model (0.921).
2. Mask contour alone (0.9602) beats TI-RADS, significantly: DeLong z = +3.06,
   p = 0.0022. Area alone manages 0.6286, so it is shape, not nodule size.
3. Contour plus TI-RADS reaches 0.9858 against a published 0.9939 that uses a
   vision-language model over images and masks. The entire learned-vision
   apparatus is worth roughly +0.008 over two human-derived annotations.

Fusion did not help. M6, with every input available, scored 0.9368 against
M3's 0.9421 on image alone, and its calibration error more than tripled, from
0.0541 to 0.1912. The AUC difference is 0.005 on 739 patients from a single
seed, which is inside the noise -- so the claim is "no gain", not "harm", until
more seeds and a paired DeLong test are in. The calibration gap is large enough
to be worth reporting now, and is consistent with the 21.6-point train/test
prevalence shift being amplified by a model with more parameters to overfit.

Note also that M3 at 0.9421 already exceeds the published image-only 0.921,
using EfficientNet-B0 at 256px for 20 epochs rather than the published B4 at
384px -- a deliberately reduced setting chosen to fit Kaggle's 9-hour limit, so
this is a floor on image-only performance, not a ceiling.

Mask geometry also recovers the radiologist's TI-RADS category at QWK 0.6545
and Spearman +0.672, so the contour encodes much of the human's assessment --
but not all of it: TI-RADS still adds +0.0256 over geometry (p = 3.8e-06), so
the two are overlapping, not interchangeable. Calling their fusion circular
would be wrong.

## Image and fusion models

Four configurations, three seeds each, EfficientNet-B0 at 256px for 20 epochs on
2x T4. Patient level on the official test split, 739 patients.

| Config | Inputs | seed 0 | seed 1 | seed 2 | mean | sd | ensemble | mean ECE |
|---|---|---|---|---|---|---|---|---|
| M3 | image | 0.9421 | 0.9287 | 0.9373 | 0.9360 | 0.0068 | 0.9464 | 0.0814 |
| M4 | image + age + sex | 0.5811 | 0.9513 | 0.6276 | 0.7200 | **0.2017** | 0.9254 | 0.1728 |
| M5 | image + TI-RADS | 0.9353 | 0.9379 | 0.9349 | 0.9361 | 0.0016 | 0.9379 | 0.0945 |
| M6 | image + age + sex + TI-RADS | 0.9368 | 0.9245 | 0.9288 | 0.9301 | 0.0062 | 0.9368 | 0.1741 |

Paired DeLong on the seed-averaged ensembles, same 739 patients:

| Comparison | Difference | z | p |
|---|---|---|---|
| M3 vs M4 | +0.0210 | +3.262 | **0.00111** |
| M3 vs M5 | +0.0084 | +0.898 | 0.369 |
| M3 vs M6 | +0.0096 | +1.054 | 0.292 |
| M5 vs M6 | +0.0012 | +0.201 | 0.841 |

**RETRACTED. These runs did not fuse anything.** The saved gates show every
multimodal configuration drove one modality to 1.0000 and the rest to 0.0000:

| Config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| M4 | exo 1.0000 | image 1.0000 | exo 1.0000 |
| M5 | endo 1.0000 | endo 0.9434 | endo 1.0000 |
| M6 | endo 0.9871 | endo 1.0000 | endo 1.0000 |

So M5 and M6 were TI-RADS-only models with the image branch switched off, and
M4 was a coin flip between age-and-sex-only (0.5811, 0.6276, against M1's
0.6186) and image-only (0.9513). The earlier reading of this table -- that M5
and M6 being indistinguishable from M3 shows fusion adds nothing -- does not
follow: two unrelated single-modality models happening to score alike says
nothing about fusion. Gap 3 is reopened.

One incidental observation survives, and it is informative. Given a free choice
between the image and TI-RADS, the gate discarded the image in three of three
seeds. With a saturating softmax that is a hard switch rather than a weighing,
but the optimiser consistently preferred the radiologist's category to the
pixels. Note also that M4's gate chose the *worse* modality in two of three
seeds, so the choice tracks early-training dynamics rather than final utility.

The architecture has since been fixed -- softmax temperature 2.0, a five-epoch
warmup blending gates toward uniform, and a normalised-entropy penalty of 0.05
in the loss. On a short real-image check the gates now mix: M4 at image 0.559 /
exo 0.441, M6 at image 0.594 / exo 0.152 / endo 0.254. The twelve runs need
repeating before Gap 3 can be answered.

**M4's instability is our architecture's fault, not a result about
demographics.** Two of three seeds landed at 0.5811 and 0.6276, within noise of
M1's age-and-sex-only 0.6186, and their validation AUCs were 0.6137 and 0.5936 --
so those runs failed to train rather than overfitting. The likely mechanism is
the softmax gate collapsing onto the exogenous branch early and starving the
image branch of gradient, which is exactly what the gates did in a smoke test
with uninformative images (image 0.0004, exogenous 0.9965). This should be
reported as a fusion-stability finding and fixed, not presented as evidence
that demographics are useless -- M1 already says that more cleanly.

**Fusion roughly doubles calibration error**, consistently across seeds: mean
ECE 0.0814 for M3 against 0.1741 for M6, against a 21.6-point train/test
prevalence shift that a larger model has more capacity to overfit.

## Control C: burned-in annotation

All 11,635 images scored for near-saturated straight runs, aggregated to
patients by mean.

| Split | Benign | Malignant | p | Annotation-only AUC |
|---|---|---|---|---|
| train (3354) | 0.012814 | 0.008202 | 2.5e-16 | 0.5927 |
| test (739) | 0.012347 | 0.007958 | 8.0e-06 | 0.5936 |

Burned-in annotation does differ by class, highly significantly, and in the
direction opposite to the obvious guess: **benign** scans carry more overlay.
But it is a weak predictor at 0.593, so it cannot account for M3's 0.9360. The
image models are not mainly reading calipers.

## Gap status

| Gap | Status | Evidence |
|---|---|---|
| 1 age and sex used at all | half | M1 = 0.6186; M4 needs the re-run after the gate fix |
| 2 clinical-only baseline | **closed** | M0 = 0.9287, M2 = 0.9404 |
| 3 endogenous vs exogenous split | reopened | gates were saturated; re-run pending |
| 4 performance by sex / age band | **closed for M0-M2** | subgroup tables in outputs/week1_real |
| 5 does fusion transport across centres | **partial** | DDTI and TN3K both negative; needs a labelled cohort for fusion |
| 6 is SOTA inflated by mask shortcuts | **closed** | B 0.9602, C 0.593, D passes |

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
