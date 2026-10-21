# CoLLM V-B2 Warm/Cold Evaluation on ML-1M

**Status report for thesis / pre-defense**

- Protocol: Section V-B2 of *CoLLM* (arXiv 2310.19488v3).
- Dataset: ML-1M, author-provided `test_ood2.pkl`.
- Branch: `feat/vb2-warm-cold-infra`.
- Implemented commits: `89b942c`, `ed2c4ff`, `7708a72`.
- Results reported here: MF, CoLLM-LoRA, and CoLLM-MICA. All values were
  verified from the downloaded result files and independently recomputed from
  the exported prediction CSVs; see Section 5.7.
- All three acceptance gates passed in the exported run and on recomputation.
- The prediction run recorded
  `git_commit=87afe6960d858e2da5bf703354678072e8d0bd1b`; the report was updated
  at commit `8b98345`.

Numbers in this report are taken from repository files, the supplied Kaggle run
log, or the downloaded result files. No value has been estimated or filled in.

---

## 1. Goal and scope

The goal is to evaluate the paper's Section V-B2 warm/cold protocol on ML-1M
using the existing CoLLM checkpoints already available in this project. The
evaluation is inference-only: no model is retrained, no fine-tuning is repeated,
and no new split is generated. The test file is `test_ood2.pkl` (7,331 rows).

The intended model set is:

| Model | Checkpoint | Status |
|---|---|---|
| MF | `weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth` | **Done** (verified from result files; recomputed) |
| CoLLM-LoRA | merged stage-2 LoRA checkpoint (`COLLM_STAGE2_CKPT`) | **Done** (verified from result files; recomputed) |
| CoLLM-MICA | frozen merged checkpoint `merged_mica_s12.pth` (`COLLM_MICA_STAGE2_CKPT`) | **Done** (verified from result files; recomputed) |

Scope limits:

- Only ML-1M and only the full `test_ood2.pkl` split.
- The MF result is the reference point for comparing CoLLM.
- No TALLRec result is produced or reported.
- The paper table is not reproduced exactly; this work uses the repository's
  `*_ood2.pkl` split and this project's checkpoints. Comparisons are therefore
  trend comparisons, not direct paper-table reproductions.

### Why TALLRec is excluded

TALLRec is outside this phase's scope. It is a separate text-only baseline and
would require its own training run and checkpoint; this phase is explicitly
restricted to evaluating already-trained CoLLM/MF assets without retraining.
TALLRec also does not use CoLLM's collaborative-ID path, so it would not answer
the CIE question addressed here. The compatibility wrapper still accepts the
old `--tallrec-pred` flag, but no TALLRec checkpoint is run in this work.

---

## 2. Warm/cold definition

### 2.1 Operational rule

For each test interaction, the training-set counts are looked up:

```text
warm = (count_train(uid) >= THRESHOLD) AND (count_train(iid) >= THRESHOLD)
cold = exact complement of warm
```

`THRESHOLD = 3`.

### 2.2 Paper and code agreement

- **Paper:** `docs/CoLLM_paper.pdf`, p. 2336, Section V-B2. The paper states that
  the warm subset contains interactions whose user and item each appeared **at
  least three times** in the training data, and the cold subset is the remainder.
- **Live code:** `warm_cold/data.py:18` defines `DEFAULT_THRESHOLD = 3`;
  `warm_cold/data.py:107` implements
  `warm = (user_counts >= threshold) & (item_counts >= threshold)`, and
  `warm_cold/data.py:108` implements `cold = ~warm`.
- **Historical source note:** `warm_cold/data.py:20-24` records
  `eval_warm_cold_vb2.py:240`, the pre-refactor threshold location. After commit
  `ed2c4ff`, `eval_warm_cold_vb2.py` is only a 67-line compatibility wrapper, so
  the live implementation is now in `warm_cold/data.py`.

The paper's "at least three" (`>= 3`) and the code's `>= 3` agree.

### 2.3 Legacy pickle flags are deliberately ignored

The pickles contain legacy flags (`not_cold`, and in the separate
`test_warm_cold_ood2.pkl`, `warm`/`cold`). They are not used for this evaluation
because they implement different criteria:

- `not_cold` marks a user/item pair whose user and item both appear **at least
  once** in training. This is a presence check, not the paper's count-`>= 3`
  rule (`dataset/ml-1m/processing_ood.ipynb`, cell 42).
- The legacy `warm` flag uses `count > 3` (strictly greater than three) for both
  user and item, not `>= 3` (`dataset/ml-1m/read_info.ipynb`, cells 4, 8, 11).
  This is an off-by-one difference from the paper.
- The legacy `cold` flag requires **both** the user and the item to be unseen,
  rather than defining cold as the exact complement of warm. Mixed cases
  (seen user/unseen item or vice versa) are neither legacy-warm nor legacy-cold.

Using these flags would therefore produce a different split, with possible
coverage gaps and an off-by-one threshold. `warm_cold/data.py` instead reads only
`uid`, `iid`, and `label`, builds the masks from training counts, and sets cold
to the exact complement. The split-integrity tests check disjointness and full
coverage.

### 2.4 Split sizes

From the ML-1M run (`test_ood2.pkl`):

| Subset | Rows | Percentage of test |
|---|---:|---:|
| Test | 7,331 | 100.00% |
| Warm | 3,682 | 50.23% |
| Cold | 3,649 | 49.77% |

Warm and cold are exact complements: 3,682 + 3,649 = 7,331.

---

## 3. Metrics

### 3.1 AUC

For a subset, AUC is `sklearn.metrics.roc_auc_score(label, logit)`. The logit is
the model score; no threshold or rounding is applied before scoring.

### 3.2 Primary UAUC (the thesis value)

The primary UAUC is implemented in `warm_cold/metrics.py:71-113`
(`per_user_aucs`) and summarized in `warm_cold/metrics.py:116-135`
(`uauc_summary`).

For each user in the subset:

1. Skip the user if there is only one interaction; AUC is undefined.
2. Skip the user if all interactions have the same label; AUC is undefined.
3. Otherwise compute the user's ROC AUC.
4. Average the finite per-user AUCs with the mean.

This is the main table value. It is therefore a **NaN-skipping mean over users
with both classes**, not the official NaN-propagating function.

Per-subset user counts for the current run (all three models use the same test
masks):

| Subset | Unique users | Computed users | Users skipped for single class | Users with one interaction |
|---|---:|---:|---:|---:|
| Warm | 245 | 158 | 53 | 34 |
| Cold | 166 | 82 | 39 | 45 |

The counts partition each subset's users: 158 + 53 + 34 = 245, and
82 + 39 + 45 = 166. In short: warm UAUC uses 158 of 245 users, cold UAUC uses
82 of 166 users.

**Caveat:** cold UAUC is based on only 82 computed users. It is a noisy estimate
and should not be treated as a stable ranking of cold-start quality.

### 3.3 Official `uAUC_me` (secondary diagnostic)

The repository's original `uAUC_me` (`minigpt4/tasks/rec_base_task.py:25-67`) is
vendored with its numeric behavior in `warm_cold/metrics.py:25-68`. It computes
per-user AUCs and then calls `np.mean` on the resulting array. A single `NaN`
per-user value therefore poisons the whole subset value. On this ML-1M run the
official value is `NaN` for both warm and cold, so it is reported only as the
secondary diagnostic column `UAUC (official, NaN-propagating)`.

The primary UAUC is not affected because it explicitly skips single-class users
and non-finite per-user values.

---

## 4. Infrastructure

### 4.1 Layout

```text
warm_cold/
├── __init__.py          package version
├── data.py              split loading, threshold, warm/cold masks
├── metrics.py           AUC, primary UAUC, vendored official uAUC_me
├── predictions.py       prediction CSV contract and validation
├── provenance.py        hashes, git commit, environment metadata
├── predict.py           unified prediction CLI
├── evaluate.py          scoring, results tables, acceptance gates
├── METRICS.md           metric definitions and threshold provenance
├── KAGGLE.md            ordered Kaggle runbook
├── warm_cold.md         this report
├── models/
│   ├── __init__.py
│   ├── mf.py            CPU MF dot-product predictor
│   └── collm.py         CoLLM-LoRA/MICA export, resume, MICA checks
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_evaluate.py
    ├── test_metrics.py
    ├── test_mf.py
    ├── test_predictions.py
    └── test_collm.py
```

At repository root:

- `run_all.sh` — runs MF, CoLLM-LoRA, CoLLM-MICA, then the combined evaluation.
- `eval_warm_cold_vb2.py` — compatibility wrapper around `warm_cold.evaluate`.

### 4.2 Module responsibilities

| Module | Responsibility |
|---|---|
| `warm_cold/data.py` | Load and validate pickles; read only `uid`/`iid`/`label`; build the threshold masks; expose row indices and split statistics. |
| `warm_cold/metrics.py` | Compute subset AUC; compute primary per-user UAUC; preserve the official NaN-propagating `uAUC_me` as a diagnostic. |
| `warm_cold/predictions.py` | Enforce the five-column prediction contract, atomic writes, and full-coverage/ID/label validation. |
| `warm_cold/provenance.py` | Record checkpoint hashes, data hashes, config hash, environment versions, and git commit in metadata JSON. |
| `warm_cold/models/mf.py` | Load the MF state dict directly and score the exact user/item dot product on CPU; no LAVIS/transformers import required. |
| `warm_cold/models/collm.py` | Load LoRA/MICA configs, build the model, enforce single-process export, write resumable predictions, and validate MICA's SVD-rebuilt `lora_B`. |
| `warm_cold/predict.py` | Single CLI for `mf`, `collm-lora`, and `collm-mica`; handles common arguments and error reporting. |
| `warm_cold/evaluate.py` | Join prediction CSVs to the test masks, compute AUC/UAUC, write `results.csv`/`results.json`, and print PASS/FAIL gates. |
| `warm_cold/tests/` | Unit and integration tests, including a fake-model partial/resume test. |
| `warm_cold/METRICS.md` | Written definition of the threshold, primary UAUC, and official diagnostic. |
| `warm_cold/KAGGLE.md` | Copy-paste Kaggle setup, env vars, test cell, export cells, and evaluation cell. |
| `run_all.sh` | End-to-end orchestration with required-env checks. |

### 4.3 Prediction CSV contract

Every model writes the same five columns:

```text
row_idx, uid, iid, label, logit
```

`row_idx` is the zero-based row position in the test pickle. The contract is
implemented in `warm_cold/predictions.py:24-44` (write) and
`warm_cold/predictions.py:47-107` (read):

- Required columns must be present.
- `row_idx` must be integer, in range, and unique.
- Full-coverage mode requires exactly one prediction for every test row.
- `uid`, `iid`, and `label` must match the test row at that `row_idx`.
- Labels must be binary 0/1 and logits must be finite.
- Writes are atomic (`.tmp` then replace).
- Reads sort by `row_idx`, so unordered CSVs are aligned safely before scoring.

This is what allows MF and CoLLM prediction files to be scored with exactly the
same masks and metrics.

### 4.4 Acceptance gates

`warm_cold/evaluate.py:25-33` defines the expected anchors. The tolerance is
`1e-6` and a gate failure exits non-zero. Unknown labels are scored but do not
get a gate.

| Model | Metrics checked | Anchor |
|---|---|---|
| MF | overall, warm, cold | overall `0.649366087184`; warm `0.727298048078`; cold `0.543353024549` |
| CoLLM-LoRA | overall | `0.7434193394` |
| CoLLM-MICA | overall | `0.7419987742` |

The MF anchors were verified from the result files. The CoLLM anchors are
pre-registered overall-AUC gates; all three passed in the exported run and were
reproduced exactly in the local recomputation described in Section 5.7.

### 4.5 CoLLM export: determinism, single process, resume

`warm_cold/models/collm.py` implements the export path:

- Single process only. `validate_single_process` rejects
  `run.distributed=True`, `run.world_size != 1`, and distributed worker
  environment variables (`WORLD_SIZE>1`, `RANK>0`, `LOCAL_RANK>0`);
  see `warm_cold/models/collm.py:95-132`.
- Deterministic order: `DataLoader(..., shuffle=False, drop_last=False)`;
  see `warm_cold/models/collm.py:592-600`.
- Partial prediction CSV and metadata are written every
  `checkpoint_every` batches (default 20), not only at the end.
- A resume reads the partial file, requires a contiguous prefix of test rows, and
  resumes only when the partial row count ends on a completed batch boundary
  (`resume_plan`, `warm_cold/models/collm.py:327-341`;
  `_read_partial`, `warm_cold/models/collm.py:352-361`).
- The final CSV is written only after all rows are present, then re-read with
  full-coverage validation. A completed partial is promoted to the final CSV.
- `--overwrite` removes the final CSV, partial CSV, and both metadata files.

MICA-specific safeguards:

- Config must use `arch == mini_gpt4rec_mica` and `lora_config.mica == true`
  (`warm_cold/models/collm.py:141-151`).
- The checkpoint basename must be exactly `merged_mica_s12.pth`
  (`warm_cold/models/collm.py:264-271`); this prevents accidental use of an
  unfrozen or unrelated stage checkpoint.
- Every `lora_B` parameter must be nonzero after the SVD rebuild, and the number
  of patched modules must match the number of `lora_B` parameters
  (`warm_cold/models/collm.py:172-194`).
- The patched-module count is printed during model build.

### 4.6 Adding a new recommender backbone or fine-tuning method

The intended extension path is:

1. Add the trained checkpoint and an eval config under `train_configs/`.
2. Add a `ModelSpec` entry in `warm_cold/models/collm.py` (label, `arch`,
   config path, MICA flag), or add a new predictor module under
   `warm_cold/models/` for a non-LLM model.
3. Add the CLI choice in `warm_cold/predict.py`.
4. Emit the same five-column prediction CSV through
   `warm_cold/predictions.write_predictions`.
5. Score it with `warm_cold.evaluate` as `LABEL=path.csv`; add an acceptance
   anchor only after a verified run, never before.
6. Add tests for the new loader/forward path and update `KAGGLE.md`.

This keeps the scorer, masks, and CSV contract shared across backbones and
fine-tuning methods.

### 4.7 Compatibility wrapper

`eval_warm_cold_vb2.py` is now a thin wrapper. It preserves the old flags
(`--mf-pred`, `--tallrec-pred`, `--collm-pred`) and forwards them to
`warm_cold.evaluate` (`eval_warm_cold_vb2.py:13`, `:24-26`, `:51-63`). It does not contain
a second scoring implementation. `--tallrec-pred` is retained only for CLI
compatibility; no TALLRec run is part of this work.

### 4.8 The only change inside `minigpt4/`

Commit `7708a72` adds exactly two lines to
`minigpt4/models/minigpt4rec_mica.py`:

- `minigpt4/models/minigpt4rec_mica.py:148` initializes
  `self.mica_patched_modules = 0`.
- `minigpt4/models/minigpt4rec_mica.py:154` stores the count returned by the
  existing SVD patch routine.

This is diagnostic state used by the exporter to verify that the number of
patched modules matches the number of `lora_B` parameters. It does not change
the forward pass, initialization values, SVD patch routine, or training
behavior.

### 4.9 Repository history

| Commit | Purpose |
|---|---|
| `89b942c` | Add the reusable V-B2 warm/cold infrastructure, MF predictor, scorer, metric tests, and validation. |
| `ed2c4ff` | Replace the old scorer body with the `eval_warm_cold_vb2.py` compatibility wrapper. |
| `7708a72` | Add resumable CoLLM-LoRA/MICA export, MICA checks, CLI integration, tests, `KAGGLE.md`, and `run_all.sh`. |

Branch: `feat/vb2-warm-cold-infra`.

---

## 5. Results so far

### 5.1 Environment and tests

The reported run is a Kaggle session using the Python 3.12 pinned image. The
export metadata records Python `3.12.13`, torch `2.10.0+cu128`, transformers
`4.28.0`, peft `0.3.0`, numpy `2.0.2`, pandas `2.3.3`, and scikit-learn
`1.6.1` for all three models. The unit-test summary was:

```text
Ran 31 tests in 16.225s
OK
```

No test was skipped in that run; the official `uAUC_me` parity test executed.
The test command used is shown in Section 6. The 31 tests include the
warm/cold mask tests, metric tests, MF predictor tests, prediction-contract
tests, MICA config/SVD checks, and a fake-model partial-export/resume test.

### 5.2 MF results

Source: `/kaggle/working/collm_logs/vb2_out/results.csv`, the session log, and
the downloaded copy in `raw_logs/warm_cold_logs/`.

| Subset | Rows | AUC | Primary UAUC | Official `uAUC_me` | Unique users | Computed users |
|---|---:|---:|---:|---:|---:|---:|
| Warm | 3,682 | 0.7273 | 0.6690 | NaN | 245 | 158 |
| Cold | 3,649 | 0.5434 | 0.5578 | NaN | 166 | 82 |
| Overall | 7,331 | 0.6494 | — | — | — | — |

Raw values from `results.csv`:

```text
warm AUC      = 0.7272980480780473
warm UAUC     = 0.6689595892862342
cold AUC      = 0.5433530245487074
cold UAUC     = 0.5578108021042136
overall AUC   = 0.649366087184006
```

Acceptance gate result:

```text
PASS acceptance gate: MF
```

Exact MF gate anchors (tolerance `1e-6`):

```text
warm    0.727298048078
cold    0.543353024549
overall 0.649366087184
```

The session log also reports the expected split and threshold:

```text
Total test interactions: 7331
Warm interactions: 3682 (50.23%)
Cold interactions: 3649 (49.77%)
Threshold: train count >= 3 for BOTH user and item
```

### 5.3 Interpretation of the MF numbers

MF's AUC falls from `0.7273` on warm rows to `0.5434` on cold rows, which is
close to random ranking (`0.5`) on the cold subset. Its warm UAUC (`0.6690`) is
also higher than its cold UAUC (`0.5578`), but the cold UAUC is based on only 82
users and is noisy.

For MF alone, the numbers support only this statement: **MF performs
substantially better on warm interactions than on cold interactions in this
split.** The CoLLM comparison is reported below.

### 5.4 CoLLM results

Source: the downloaded `results.csv`/`results.json`, both CoLLM prediction
CSVs, and the local recomputation in Section 5.7. Full-precision values are in
`results.csv`; the table below keeps four decimal places.

| Model | Warm AUC | Warm UAUC | Cold AUC | Cold UAUC | Overall AUC |
|---|---:|---:|---:|---:|---:|
| MF | 0.7273 | 0.6690 | 0.5434 | 0.5578 | 0.6494 |
| CoLLM-LoRA | 0.7782 | 0.7147 | 0.7031 | 0.6548 | 0.7434 |
| CoLLM-MICA | 0.7707 | 0.7155 | 0.7121 | 0.6656 | 0.7420 |

The MICA row uses the frozen merged checkpoint `merged_mica_s12.pth`.
Acceptance gates (tolerance `1e-6`):

| Model | Anchor(s) | Result |
|---|---|---|
| MF | overall `0.649366087184`; warm `0.727298048078`; cold `0.543353024549` | **PASS** |
| CoLLM-LoRA | overall `0.7434193394` | **PASS** |
| CoLLM-MICA | overall `0.7419987742` | **PASS** |

All three gates passed in the exported run and again on recomputation. The
CoLLM rows above are verified from the downloaded files, not console-only
values.

### 5.5 Interpretation of the warm/cold results

- MF drops about `0.184` AUC from warm to cold and is close to random on cold.
- CoLLM-LoRA drops about `0.075`; CoLLM-MICA drops about `0.059`.
- On cold, both CoLLM variants are about `0.16`–`0.17` AUC above MF.
- LoRA is slightly higher on warm AUC and overall AUC; MICA is slightly higher
  on cold AUC.

These differences are small. **No significance test or confidence interval has
been computed**, so this report does not claim that either CoLLM variant is
better than the other.

### 5.6 Comparability caveat

These numbers are not direct reproductions of the paper's table. The run uses a
different split implementation and this project's checkpoints, which may differ
from the paper's assets. The valid comparison here is the internal
warm-versus-cold trend, not a claim that a number matches the published table.

### 5.7 Verification against the downloaded files

Verification date: 2026-10-21. Inputs: the local copy of the Kaggle output in
`raw_logs/warm_cold_logs/`.

**Inventory.** All 11 expected files are present and no extra files were found:

```text
results.csv                  results.json
pred_mf.csv                  pred_mf.meta.json
pred_collm_lora.csv          pred_collm_lora.meta.json
pred_collm_mica.csv          pred_collm_mica.meta.json
warm_row_idx.npy             cold_row_idx.npy
warm_cold_tests_cell_output.txt
```

`results.csv` SHA-256 is
`6db25d40a4922125cd9d00052853c613326057aae61bc16f07c0260b2ea73131`;
`results.json` is
`ae2eb98e236c6b2e427dacc37b815279edfa28f15e01f8631ce150ab06ac0757`.
The prediction CSVs are 209–265 KB; the test-cell log is 357,649 bytes.

**Data and prediction contract.** `collm-datasets/ml-1m.zip` contains
`train_ood2.pkl` (SHA-256
`a0c4cb50a61a27453aa8c14a81cb1b41b1fdc874362e46960956b0bc720c21fe`) and
`test_ood2.pkl` (SHA-256
`a35ec4a7d93a76f2a81a0c0c926a95ad5fa5fe4ea8190f02dc1630c58a0e167a`);
both hashes match all three metadata files. The test split has 7,331 rows. For
each model, the CSV has the five required columns, exactly 7,331 rows,
`row_idx` equal to `0..7330` with no duplicates, and finite logits. `uid`,
`iid`, and `label` match `test_ood2.pkl` after `reset_index(drop=True)` for every
row. All three models cover the same rows.

**Masks.** The regenerated masks have 3,682 warm rows and 3,649 cold rows, are
disjoint, and cover all 7,331 rows. `warm_row_idx.npy` and `cold_row_idx.npy`
match the regenerated arrays exactly. User-count checks match the exported
table: warm has 245 unique users, 158 computed, 53 single-class, and 34
single-interaction; cold has 166 unique, 82 computed, 39 single-class, and 45
single-interaction. These counts are identical for all three models.

**Metrics.** Re-running `python -m warm_cold.evaluate` on the three exported
CSVs with the locally extracted data produced a `results.csv` identical in every
field to the exported `results.csv`. `results.json` also matched on all model
metrics, split statistics, and acceptance results. Full-precision core values:

| Model | Warm AUC | Warm UAUC | Cold AUC | Cold UAUC | Overall AUC |
|---|---:|---:|---:|---:|---:|
| MF | 0.7272980480780473 | 0.6689595892862342 | 0.5433530245487074 | 0.5578108021042136 | 0.649366087184006 |
| CoLLM-LoRA | 0.7782099266974793 | 0.7146614481079735 | 0.7031433308063144 | 0.6548131593738812 | 0.7434193394326927 |
| CoLLM-MICA | 0.7706538324357364 | 0.7155236105401004 | 0.712070605153898 | 0.6655873718468952 | 0.7419987746648602 |

The recomputed gates all passed: MF overall/warm/cold deltas were at most
`2.93e-13`; LoRA overall delta `3.3e-11`; MICA overall delta `4.6e-10`, against
tolerance `1e-6`.

**Metadata.** All three models record threshold `3`, seed `42`, and
`git_commit=87afe6960d858e2da5bf703354678072e8d0bd1b`; all data hashes are
identical. Environment: Python `3.12.13`, torch `2.10.0+cu128`, transformers
`4.28.0`, peft `0.3.0`, numpy `2.0.2`, pandas `2.3.3`, scikit-learn `1.6.1`.
Checkpoints:

| Model | Checkpoint | SHA-256 | Size |
|---|---|---|---:|
| MF | `/kaggle/working/CoLLM-thesis/weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth` | `08f3574e4fa39d81bfd54cad6fdc610c9af486d01f1d3e9b2c42de97d75fe1ce` | 4,195,749 B |
| CoLLM-LoRA | `/kaggle/input/models/rahelehvafajoo/collm-merged-lora/pytorch/default/1/merged_lora_s12.pth` | `6fbe4456017b699e1c0bd866f9e6d108f7725ad2f584faeef0a966ee62e4446d` | 61,409,517 B |
| CoLLM-MICA | `/kaggle/input/models/rahelehvafajoo/collm-merged-mica/pytorch/default/1/merged_mica_s12.pth` | `700bbc34aed06c02981f3bf7802d0e929b18e9255140f5a4b2b06c9c2ebf95fb` | 53,006,183 B |

LoRA used config `collm_eval_mf_ood.yaml` (SHA-256
`a97f6826344ed4ee446a6a0911761b2f4d12f000caeb4a2de6105a1afb0fd8fa`) and MICA
used `collm_mica_eval_mf_ood.yaml` (SHA-256
`a358f949d90a6e280f83bed7e185d255255bc86da410b1e23d0d452600fb7d73`). Both used
the same frozen MF checkpoint and merged Vicuna directory
`/kaggle/working/vicuna-v0/pretrained_ckpt/vicuna_ckpt/7b_v0`. MICA recorded
`patched=64`, `nonzero=64`, `total=64`. The exported `git_commit` predates the
report commit `8b98345`; this is expected because the predictions were produced
at `87afe696` and the report was updated later.

**Discrepancies.** None. No expected file was missing, no unexpected file was
present, and no metric or count differed at the printed precision.

**Limit of this verification.** The prediction CSVs were not regenerated: a
local rerun of the LLM export is not possible in this environment (no
Vicuna/transformers/peft installation and no GPU). The check verifies the
exported logits, CSV contract, masks, counts, metrics, gates, and metadata. The
recompute used Python `3.13.0`, numpy `2.3.5`, pandas `2.2.3`, and scikit-learn
`1.8.0`; despite that difference, every metric and count matched exactly.

---

## 6. Reproduction

### 6.1 Kaggle environment variables

Run this in a Python notebook cell first. Replace the input paths with the
actual Kaggle mounts for this session. `VB2_OUT` is derived from
`COLLM_OUTPUT_DIR` so later shell cells can use `$VB2_OUT` directly.

```python
import os

REPO = "/kaggle/working/CoLLM"

os.environ["REPO"] = REPO
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

os.environ["COLLM_LLAMA_DIR"] = "/kaggle/input/vicuna-v0/7b_v0"
os.environ["COLLM_ML1M_DIR"] = "/kaggle/working/ml-1m"
os.environ["COLLM_OUTPUT_DIR"] = "/kaggle/working/collm_logs"
os.environ["COLLM_MF_REC_PTH"] = "/kaggle/input/collm-weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"
os.environ["COLLM_STAGE2_CKPT"] = "/kaggle/input/collm-weights/merged_lora_s12.pth"
os.environ["COLLM_MICA_STAGE2_CKPT"] = "/kaggle/input/collm-weights/merged_mica_s12.pth"

os.environ["VB2_OUT"] = os.path.join(os.environ["COLLM_OUTPUT_DIR"], "vb2_out")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.makedirs(os.environ["VB2_OUT"], exist_ok=True)
```

The MICA checkpoint filename must remain `merged_mica_s12.pth`; the exporter
rejects other basenames.

### 6.2 Ordered steps

**1. Unit tests.** This must run the official parity test, not skip it.

```bash
!cd "$REPO" && COLLM_REQUIRE_OFFICIAL_PARITY=1 python -m unittest discover -s warm_cold/tests -v
```

**2. MF prediction.**

```bash
!cd "$REPO" && python -m warm_cold.predict \
  --model mf \
  --data-dir "$COLLM_ML1M_DIR" \
  --mf-ckpt "$COLLM_MF_REC_PTH" \
  --out "$VB2_OUT/pred_mf.csv"
```

**3. MF-only evaluation and gate.**

```bash
!cd "$REPO" && python -m warm_cold.evaluate \
  --data-dir "$COLLM_ML1M_DIR" \
  --pred "MF=$VB2_OUT/pred_mf.csv" \
  --out-dir "$VB2_OUT" \
  --dump-masks
```

This prints the split counts and `PASS acceptance gate: MF`.

**4. CoLLM-LoRA export.** Single process; do not use `torchrun`.

```bash
!cd "$REPO" && python -m warm_cold.predict \
  --model collm-lora \
  --cfg-path train_configs/collm_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$VB2_OUT/pred_collm_lora.csv" \
  --checkpoint-every 20 \
  --num-workers 0 \
  --device cuda:0
```

**5. CoLLM-MICA export.** Uses only the frozen merged
`merged_mica_s12.pth`; the exporter validates the architecture, `mica` flag,
checkpoint basename, and nonzero SVD-rebuilt `lora_B`.

```bash
!cd "$REPO" && python -m warm_cold.predict \
  --model collm-mica \
  --cfg-path train_configs/collm_mica_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$VB2_OUT/pred_collm_mica.csv" \
  --checkpoint-every 20 \
  --num-workers 0 \
  --device cuda:0
```

Both CoLLM commands are resumable: if the session stops, rerun the same cell and
the exporter continues from the last completed batch. Use `--overwrite` only
when intentionally restarting.

**6. Final evaluation of all three models.** Run after the LoRA and MICA CSVs
exist. Each CSV is checked for full coverage before scoring.

```bash
!cd "$REPO" && python -m warm_cold.evaluate \
  --data-dir "$COLLM_ML1M_DIR" \
  --pred "MF=$VB2_OUT/pred_mf.csv" \
  --pred "CoLLM-LoRA=$VB2_OUT/pred_collm_lora.csv" \
  --pred "CoLLM-MICA=$VB2_OUT/pred_collm_mica.csv" \
  --out-dir "$VB2_OUT" \
  --dump-masks
```

The single-command alternative, using the same environment variables, is:

```bash
!cd "$REPO" && bash run_all.sh
```

Use `$VB2_OUT` in shell lines, never the literal `{OUT}`; `{OUT}` is not
expanded by the shell.

### 6.3 Known pitfalls hit so far

1. **Kaggle's default image may be Python 3.13.** Use the Python 3.12 pinned
   image by selecting **Copy & Edit** on an older notebook that already has the
   working 3.12 environment. The repository's pinned stack (`transformers
   4.28.0`, `peft 0.3`, `bitsandbytes 0.37.0`) is not compatible with a generic
   Python 3.13 image.
2. **`{OUT}` is not expanded in `!` lines.** Set `VB2_OUT` in `os.environ` and
   use `$VB2_OUT` in shell commands, as shown above.
3. **Vicuna must be the merged weights directory.** `COLLM_LLAMA_DIR` must point
   to the directory containing `config.json`, `tokenizer.model`, and
   `pytorch_model-*.bin` — not to the gated LLaMA base or to the delta weights.
   Use the merge procedure in `PrepareVicuna.md`, or the pre-merged
   `mvsoom/pandagpt-vicuna-v0-7b` asset under
   `pretrained_ckpt/vicuna_ckpt/7b_v0/` (see `AGENTS.md`).
4. **CoLLM export must be one process.** Do not launch it with `torchrun`; the
   exporter rejects `run.distributed=True`, `world_size != 1`, and distributed
   rank environment variables.
5. **MICA checkpoint name matters.** `COLLM_MICA_STAGE2_CKPT` must resolve to a
   file named `merged_mica_s12.pth`; an unfrozen or renamed checkpoint is
   rejected before inference.
6. **Dataset filenames are hard-appended by the dataset class.** The loader
   expects `train_ood2.pkl`, `valid_ood2.pkl`, and `test_ood2.pkl` in
   `COLLM_ML1M_DIR`; do not point it at a path whose stem already includes
   `_ood2.pkl`.
7. **A session change can lose `pred_mf.csv`.** It was missing when the session
   changed and was regenerated; MF is deterministic and takes only seconds.
8. **Merged checkpoints are uploaded as Kaggle Models.** The LoRA and MICA
   merged `.pth` files are attached as model inputs, not produced in the
   notebook. Keep the MICA file named exactly `merged_mica_s12.pth`.
9. **The `Wrote ...` line was observed twice for MICA.** This is duplicate
   logging, not a second export: `predict_collm` prints at
   `warm_cold/models/collm.py:678-679`, then the CLI calls `_write_message` at
   `warm_cold/predict.py:88`, which prints again at `warm_cold/predict.py:33-39`.
   The code does this for both CoLLM-LoRA and CoLLM-MICA; it is not
   MICA-specific.

---

## Appendix A: Result files produced by the current run

The Kaggle output directory contains:

```text
results.csv
results.json
pred_mf.csv
pred_mf.meta.json
pred_collm_lora.csv
pred_collm_lora.meta.json
pred_collm_mica.csv
pred_collm_mica.meta.json
warm_row_idx.npy
cold_row_idx.npy
warm_cold_tests_cell_output.txt
```

`results.json` records the threshold, threshold source, data paths, split
statistics, per-user AUC arrays, acceptance details, and provenance. The masks
are exactly the warm/cold row indices used by the scorer. All values in Sections
5.2 and 5.4 have been checked against these files; Section 5.7 records the
checks. The downloaded copy used for verification is
`raw_logs/warm_cold_logs/`.

## Appendix B: Remaining limitations

- The LLM export itself was not rerun locally; only the exported logits were
  rescored.
- No significance test or confidence interval has been computed.
- Cold UAUC is based on 82 computed users and is noisy.
- The numbers are not a direct reproduction of the paper table (Section 5.6).
