# CoLLM Innovation Candidates — Research Ideation

**Date:** 2026-08-24 · **Status:** ideation document (separate from MEMORY.md / IMPLEMENTATION_REPORT.md)
**Baseline to beat:** stage-2 best_valid_auc **0.72567 @28** (Phase 3); test AUC **0.7434** / NDCG **0.8663** (Phase 4).
**All citations verified this session** via arXiv API / OpenAlex / DOI resolution; anything not verifiable is explicitly flagged.

**Execution environment (binding for every estimate):** Kaggle notebooks only; single NVIDIA T4 16GB (DDP gives no memory savings for a replicated 7B — always single-GPU); 30 GPU-h/week shared quota; ~9 h hard session cap; `/kaggle/working` wiped between sessions (manual backup / Kaggle Models/Datasets for persistence). Resume infrastructure available and confirmed: atomic saves + `save_ckpt_keep` retention + persisted early-stop state (commit `a712fc7`) + `resume_ckpt_path`.

---

## 0. Failure-mode analysis of Option C (what any new idea must explain)

| Attempt | Change | Result | Lesson |
|---|---|---|---|
| Control (`proj_token_num` 1→4) | +capacity | +0.00003 (null) | Parameter/token count is not the bottleneck |
| CIE-G (affine + gated MLP) | +capacity, geometry-preserving path | −0.0043 | Preserving dot-product geometry *alongside* the MLP doesn't help → ReLU is not the bottleneck either |
| CIE-X (joint MHA fusion, 256-d) | +architecture (attention) | −0.025, peak @48 vs @28 | Extra fusion machinery under-fits on ML-1M scale and delays convergence |

Synthesis: at ML-1M scale (~839 users, ~3.7k items), the baseline 2-layer MLP CIE is already sufficient to translate MF→LLM tokens. Every attempt to make the *mapping* richer lost. Therefore improvements must come from **(a) better training signal, (b) better initialization/conditioning of the same mapping, or (c) better PEFT on the LLM side — not from more expressive fusion**. This mirrors the broader finding that learned nonlinear combiners often lose to well-trained simple ones on small data:

- Steffen Rendle, Walid Krichene, Li Zhang, John Anderson, *Neural Collaborative Filtering vs. Matrix Factorization Revisited*, arXiv 2005.09683 (2020).
- Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach, *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*, RecSys 2019, arXiv 1907.06902.

---

## 1. RECOMMENDED — "Anchored & Regularized CIE" (two orthogonal, cheap interventions, one experiment program)

### 1.1 CIE-R: relational-consistency regularization (signal, not capacity)

- **Mechanism:** Keep the baseline `llama_proj` exactly. During stage-2 training only, collect the batch's raw MF embeddings `[user; history…; target]` (256-d) and their projected LLM-space counterparts (4096-d, L2-normalized), and penalize relational structure divergence à la Relational Knowledge Distillation: `L_rel = λd·‖D_raw − D_proj‖² + λa·(angle-wise cosine-pair loss)` where D are pairwise distance/similarity matrices within the batch. Total loss = BCE + `L_rel`.
- **Key repo hook:** the authors left a dormant, disabled consistency loss in the code — `consitence_loss()` (`minigpt4/models/minigpt4rec_v2.py:27-42`) whose call site is commented out at `minigpt4/models/minigpt4rec_v2.py:747` (`# + 1e-7 * samples_encode['loss_c']`). We revive and formalize it with proper distance-wise + angle-wise forms and study *why* it was disabled.
- **Citations:**
  - Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho, *Relational Knowledge Distillation*, CVPR 2019, arXiv 1904.05068 (verified).
  - Related-but-different: DMRec, *Towards Distribution Matching between Collaborative and Language Spaces for Generative Recommendation*, ACM 2025, DOI 10.1145/3726302.3730098 (generative setting, meta-network + cross-space distribution matching).
  - EasyRec, Xubin Ren & Chao Huang, EMNLP 2025 main, arXiv 2408.08821 (text-behavior contrastive alignment).
  - CL³M-Rec, ICDM 2025, DOI 10.1109/icdm65498.2025.00106 (contrastive ID enhancement).
- **Why it succeeds where Option C failed:** Option C added parameters/architecture; this adds *supervisory signal* to the existing parameterization. Small-data regimes reward denser signal over capacity (the exact lesson of Rendle 2020 + our Control result). The regularizer shapes the same MLP the baseline already proved optimal, so it cannot trigger CIE-X's under-fitting pathology (no new modules, no late convergence shift).
- **Feasibility:** new model class (e.g. `minigpt4/models/minigpt4rec_cie_r.py`, subclass overriding `forward_v2` to return BCE + λ·L_rel), one YAML. Runtime overhead negligible (pairwise sims on ≤ B×(max_items+2) tokens). Budget: pick λ ∈ {0.01, 0.1, 1} via three short proxy runs (max_epoch≈12, eval_freq 2, ≈1.5–2 h each), then one full run ≈7 h + 0.8 h eval. Total ≈ 13–15 GPU-h; multi-session-safe via existing resume infra.
- **Risks:** (a) λ mis-tuning wastes quota (mitigated by proxy-run screening); (b) MF geometry may be partially task-irrelevant — forcing preservation could conflict with BCE; that tension is itself the research question and yields an honest ablation either way; (c) novelty is moderate — RKD is established; the contribution is the instantiation + the "signal-not-capacity" evidence chain against our three documented negatives.

### 1.2 CIE-A: closed-form anchored initialization (conditioning, not capacity)

- **Mechanism:** Offline, embed every item title once with frozen Vicuna (mean-pooled last-layer states; ~3.7k items, minutes on a T4). Solve ridge regression `W* = argmin ‖MF_emb·W − title_emb‖² + α‖W‖²` (closed form, CPU seconds). Initialize the CIE output layer with `W*` (or add `W*` as a frozen affine path with the learned MLP as residual, reusing CIE-G's gating pattern but with a *meaningful* init instead of random). Stage-2 proceeds unchanged.
- **Grounding:** AlphaRec — Leheng Sheng, An Zhang, Yi Zhang, Yuxin Chen, Xiang Wang, Tat-Seng Chua, *Language Representations Can be What Recommenders Need: Findings and Potentials*, ICLR 2025 Oral, arXiv 2407.05441 (verified): item representations *linearly mapped* from LM representation space yield strong recommendation performance ⇒ language and recommendation spaces are near-linearly homomorphic; a linear anchor is principled, not a hack. Contrast with PiSSA-family (SVD init *inside* LLM weights, arXiv 2404.02948) — we anchor the *modality bridge* instead.
- **Why it succeeds where Option C failed:** Our observed pathology is slow/late convergence of randomly-initialized projections under tiny data (CIE-X peaked @48). Initialization quality is strictly orthogonal to architecture expressiveness — it adds zero inference parameters and cannot cause capacity-driven under-fitting. Even if the ridge fit is poor, worst case ≈ baseline (training continues from a slightly different point).
- **Feasibility:** one offline script + init-loading in a new model class + one YAML. Ridge solve trivially cheap. One full stage-2 run ≈7 h + eval ≈0.8 h ⇒ **~8 GPU-h total**. Single-session completion after setup/Vicuna download (~30 min) is possible but tight vs the 9 h cap — rely on resume if it dies mid-run; manually back up `checkpoint_best.pth` (~44 MB) regardless.
- **Kill-switch (Step 0, ~zero GPU):** load item-embedding matrix from `weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth`; compute Vicuna mean-pooled title embeddings (fp16 forward pass over ~3.7k titles); fit 5-fold out-of-fold ridge MF→text; report (a) OOF R² and (b) Spearman ρ between MF-vs-text pairwise similarity structures. **Decision protocol (agreed): no automatic threshold gate — report numbers + interpretation/recommendation to the user; the user makes the final call before any GPU training spend** (fixed thresholds deemed arbitrary for a one-time consequential decision).
- **Risks:** (a) MF ID-embeddings may correlate weakly with title semantics on ML-1M (check R² offline *before* spending GPU time); (b) norm-scale mismatch between pooled Vicuna sentence embeddings and token embeddings — handle via standardization; (c) novelty: no verified prior doing closed-form anchored init of CF→LLM projectors (closest are A-LLMRec arXiv 2404.11343 KDD 2024 — *trains* a fusion projector with alignment data — and EasyRec's contrastive alignment; both training-based, none closed-form/init-based).

### 1.3 Combined design & budget (Kaggle-audited)

| Item | GPU-h | Constraint check |
|---|---|---|
| Ridge R² kill-switch (CIE-A Step 0) | ~0.3–0.5 | ✅ trivial; MF `.pth` already in repo |
| CIE-A stage-2 full run + eval | ~8 | ✅ < 30/wk with large margin; ⚠ tight vs 9 h session cap → resume infra covers death; manual backup discipline applies |
| CIE-R proxies (3 × ~2 h) + full run | ~13–15 | ✅ fits; multi-session-safe via same resume path |
| 2×2 ablation (init × reg), minimal 3-run version | ~24 | ✅ ≈ 80% of weekly quota — schedule across sessions deliberately |

Deliverable narrative for the thesis: *"You don't need a fancier mapping; you need a better-conditioned optimization of the simple mapping"* — backed by our three negative controls, the dormant-loss archaeology (`consitence_loss`), and the AlphaRec/Rendle literature line.

### 1.4 Cheap add-on (optional third arm): curriculum hybrid prompting

LLaRA (Jiayi Liao et al., arXiv 2312.02445; commonly attributed to SIGIR 2024 — verify venue for final bib) ramps text-only→hybrid prompts via curriculum learning. Our code already mixes prompt templates with static weights (`self.prompt_with_p([5,5,5,1])`, `minigpt4/models/minigpt4rec_v2.py:693`); converting this to an epoch-dependent schedule is a small change and directly implements the "history-aware/training-schedule" alternative flagged in IMPLEMENTATION_REPORT §6.7.3. One run ≈7 h. Low novelty alone; valuable as part of the conditioning story.

---

## 2. Evaluated but deprioritized candidates

### 2.1 MiCA for Stage 1 — keep deferred; blocker removed, objections remain

**Verified real:** *MiCA Learns More Knowledge Than LoRA and Full Fine-Tuning*, Sten Rüdiger & Sebastian Raschka, arXiv 2604.01694 (submitted 2 Apr 2026). SVD-based PEFT constraining updates to minor singular directions; up to 5.9× knowledge acquisition at 6–60% of LoRA's params.

**Update (2026-08-24): code availability RESOLVED.** Official peft integration exists (`peft/examples/mica_finetuning`). Mechanism as implemented is minimal: it is an initialization mode of plain LoRA —

```python
LoraConfig(init_lora_weights="mica", r=16, lora_alpha=16,
           target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
# B = U[:, -r:] (minor left singular vectors of W, FROZEN); A = 0 (TRAINABLE)
# ΔW = (α/r)·B·A ; adapter contributes exactly 0 at step 0
```

Supports `nn.Linear`/`nn.Embedding`; one-time full SVD per target weight ("seconds" at 7B scale); not combinable with DoRA/variants; docs recommend continued-pretraining-from-base workflow, then merge.

**Compatibility with our pinned stack (checked):**

| Path | Verdict | Detail |
|---|---|---|
| peft == 0.3.0 as-is | ❌ | `init_lora_weights="mica"` doesn't exist in 0.3 (only True/False) |
| Upgrade peft (main ≈ 0.20.x) **with deps** | ❌ Forbidden | Pulls latest transformers → reintroduces the rotary-NaN bug (IMPLEMENTATION_REPORT §2.2, our hardest bug) + tokenizers version gate |
| Upgrade peft `--no-deps` | ⚠ Fragile | modern peft requires Python ≥3.10 (repo env nominally py3.9; Kaggle image newer — must verify in-session); untested peft-0.20-on-transformers-4.28 combo; state-dict key format drift could break `merge_stage_ckpts.py` / resume compat |
| **Vendor ~30-line MiCA init ourselves (RECOMMENDED)** | ✅ | New-file discipline: build standard peft-0.3 LoRA, overwrite `lora_B ← U[:,-r:]` (SVD computed fp32 once, frozen), zero `lora_A` (trainable). peft 0.3 scales α/r and backprops through constant B. Zero changes to pins; checkpoints stay format-compatible |

**Remaining risks (unchanged):**
1. *Wrong bottleneck* — Stage 1 is demonstrably not our limiter: text-only 0.662 vs fused 0.7434; stage 2 dominates; LoRA (r=8, q/v) converged in 10 epochs. A better Stage-1 PEFT has little headroom to act on.
2. *Contested science* — intermediate components beat first (PiSSA)/last (MiLoRA) for performance–forgetting trade-offs (Quercia et al., *Least but not Last*, arXiv 2602.03493, Feb 2026); PiSSA/MiLoRA can *underperform* vanilla LoRA off-SFT (Zhang et al., ICML 2026, arXiv 2606.31813). MiCA's claims are on language-knowledge-acquisition benchmarks, not ranking/BCE tasks. Lineage: SoMA (CVPR 2025, arXiv 2412.04077) already did minor-component adaptation for vision DG; MiLoRA (NAACL 2025, arXiv 2406.09044) is the direct LLM predecessor.
3. *Workflow mismatch* — peft docs frame MiCA for continued pretraining from base; our Stage 1 is supervised task tuning, exactly the regime where structurally-initialized variants have failed elsewhere.
4. Cost ≈ 13 GPU-h (stage-1 rerun ~5 h + downstream stage-2 rerun ~7 h + eval ~0.8 h), spanning up to 2 sessions; resume infra covers it.

**Verdict: revisit only after CIE-A concludes, as a deliberate high-risk second contribution. Expected AUC gain low; novelty-narrative value moderate.**

### 2.2 Other low-risk PEFT tweaks — cheap but weak thesis material

DoRA (ICML 2024 Oral, arXiv 2402.09353), rsLoRA (arXiv 2312.03732), LoRA+ (arXiv 2402.12354), VeRA (ICLR 2024, arXiv 2310.11454). Any could drop into Stage 1 as a one-run ablation; none attacks our actual failure modes. Discussion context: *LoRA Learns Less and Forgets Less* (TMLR 2024, arXiv 2405.09673); *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (arXiv 2410.21228 — "intruder dimensions").

### 2.3 Different injection paradigms — interesting, wrong cost/benefit for us

- **CoRA** (Yuting Liu et al., arXiv 2408.10645): injects CF embeddings into learnable queries converted to low-rank weight deltas merged into LLM weights — no input tokens at all. Directly diagnoses our setting ("collaborative features in prompts disrupt prompt semantics"; fine-tuning undermines world knowledge). Heavy to reimplement under peft 0.3 + 16 GB; novelty collision.
- **BinLLM** (Yang Zhang et al. — CoLLM's own group, ACL 2024, arXiv 2406.03210): encodes CF embeddings as binary text sequences. Paradigm shift requiring full retrains; differentiates rather than improves.
- **Laser** (Xinyu Zhang et al., arXiv 2409.01605): prefix/suffix virtual tokens + M-Former MoE query transformer over frozen SR embeddings. MoE-on-prefix on a 16 GB T4 risky; heavy infra.
- **E4SRec** (Xinhang Li et al., arXiv 2312.02443), **LC-Rec** (ICDE 2024, DOI 10.1109/icde60146.2024.00118), **TokenRec** (TKDE 2025, DOI 10.1109/tkde.2025.3599265): generative semantic-ID lines — incompatible with our discriminative Yes/No v2 objective.
- **A-LLMRec** (Sein Kim et al., KDD 2024, arXiv 2404.11343): fully frozen LLM + CF/text embedding fusion — an alternative baseline worth citing, not an extension of ours.
- Also useful related-work coverage: GAL-Rec (TKDE 2025, arXiv 2406.13235); RecInterpreter (*Large Language Model Can Interpret Latent Space of Sequential Recommender*, TOIS 2025, DOI 10.1145/3786201 — auxiliary narration task grounds the projector); TCA4Rec (*Token-level Collaborative Alignment…*, WWW 2026, DOI 10.1145/3774904.3792727 — soft CF labels inside NTP; discriminative analogue: distill MF score into the Yes-logit as fallback regularizer if RKD disappoints); *Do LLMs Understand Collaborative Signals? Diagnosis and Repair* (arXiv 2505.20730).
- **Dynamic-CoLLM** (Zhang Z. et al., 2026, Springer proc. *Advanced Intelligent Computing Technology and Applications*, pp. 490–501, DOI 10.1007/978-981-92-3391-5_40): ⚠ title indicates misalignment-aware gated fusion directly on CoLLM-style pipelines — i.e., **our CIE-G/X space is being published by others**. Abstract unretrievable (paywall/OpenAlex-null); exact mechanism unknown — treat accordingly. Strategic implication: pivoting away from pure fusion-architecture contributions (toward §1) is well-timed.

### 2.4 Free evaluation-rigor fix (do regardless)

UAUC=nan comes from per-user AUC arrays containing NaN poisoning the mean (287 users computed; IMPLEMENTATION_REPORT §6.4). Switch to `np.nanmean` + explicit exclusion reporting, and document the per-user protocol. Zero GPU cost; removes a thesis-defense liability.

---

## 3. Ranking

| Rank | Idea | Failure-mode fit | Feasibility (budget) | Novelty | Expected value |
|---|---|---|---|---|---|
| 1 | §1.1+§1.2 Anchored & Regularized CIE (+§1.4 optional) | Attacks conditioning/signal — axes Option C never touched | ~8 GPU-h first signal; ~24 GPU-h full program; stage-2 only; low eng risk | Moderate-good; strong *narrative* (negatives → principle) | Highest |
| 2 | §2.4 eval rigor (+§2.2 one cheap PEFT ablation) | n/a (rigor) | <8 GPU-h | Low | High per cost |
| 3 | §2.1 MiCA Stage-1 replacement | None (wrong bottleneck) | ~13 GPU-h, ≤2 sessions, HIGH eng risk (vendored-init path mitigates) | High (if it works) | Low-moderate |

---

## 4. Decision log

- **2026-08-24:** Next experimental cycle = **CIE-A (closed-form anchored initialization) ALONE first** (~8 GPU-h), cheapest-signal-first; NOT the combined 2×2 ablation yet.
- **2026-08-24:** Ridge kill-switch protocol: **no fixed thresholds** — report OOF R² + similarity-structure Spearman ρ + interpretation/recommendation to the user; **user makes the final call** before any GPU training spend.
- **2026-08-24:** MiCA risk assessment updated: official peft implementation exists → code availability no longer a blocker; preferred route on our stack = vendored ~30-line init surgery under peft 0.3 (not a peft upgrade). Wrong-bottleneck and contested-science objections stand; MiCA stays deferred until CIE-A concludes.
- **2026-08-24 (Step-0 EXECUTED):** Real Vicuna pass completed under validated env (transformers 4.28.0, vendored modeling_llama, 3255/3255 items, cache `title_embs_cache.npz`). **Result: OOF R² (var-weighted) = 0.0079, Spearman ρ = 0.0272, Pearson = 0.0275, recall@10 = 0.0142 (chance ≈ 0.0067), best α = 1e4 (grid max).** Calibration reference: null-mock gave R² −0.005/ρ 0.001; weak-signal mock (0.3) gave R² 0.261/ρ 0.737. Measured values sit ~10× closer to the null than to weak signal ⇒ **pre-registered CIE-A (linear anchor to last-layer title embeddings) measures as DEAD**. Status: awaiting user go/no-go between (a) close CIE-A & pivot to CIE-R, (b) one ~0.3–0.5 GPU-h salvage probe (input-token-space + mid-layer L12/L24 target spaces) before closing, (c) proceed anyway (not recommended).
- **Pending user decision:** proceed/abort after kill-switch numbers are reported.
