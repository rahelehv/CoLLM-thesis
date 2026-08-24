"""
Step 0 kill-switch for CIE-A (closed-form anchored initialization of the CIE).

Question answered here: do MF item embeddings correlate meaningfully with
Vicuna's title-text embeddings? If not, anchoring the CIE init with a ridge
map MF->text is pointless and we skip the GPU spend.

What it does:
  1. Builds iid -> title map from the ML-1M ood2 pickles (train/valid/test).
  2. Loads MF item embeddings (item_embedding.weight) from the Phase-1 .pth,
     drops padding / untitled items.
  3. Computes one text embedding per title.
       --text-source vicuna : frozen Vicuna last-layer mean-pool (Kaggle, needs
                              --llama_model pointing at the 7B v0 weights dir;
                              imports minigpt4.models.modeling_llama exactly like
                              the training code to avoid any version shadowing)
       --text-source mock   : deterministic pseudo-embeddings for offline
                              plumbing tests; with --mock-signal S>0 it embeds
                              Y = X @ W_true + sigma*noise so the metrics can be
                              validated against known ground truth
  4. 5-fold OOF ridge regression (StandardScaler + Ridge, alpha grid selected by
     mean fold R^2), reports pooled OOF R^2 (uniform + variance-weighted).
  5. Similarity-structure agreement: Pearson/Spearman between pairwise cosine
     sims in MF space vs text space (+ top-k neighbor overlap).

Usage (local smoke test):
  python step0_ridge_killswitch.py --ml1m_zip collm-datasets/ml-1m.zip \
      --mf_pth weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth \
      --text-source mock --mock-signal 0     # expect R^2 ~= 0
  python step0_ridge_killswitch.py ... --text-source mock --mock-signal 0.9

Usage (Kaggle, after the usual setup ritual):
  python step0_ridge_killswitch.py \
      --mf_pth /kaggle/working/CoLLM/weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth \
      --ml1m_zip <path>/ml-1m.zip --text-source vicuna \
      --llama_model $COLLM_LLAMA_DIR --cache_file title_embs_cache.npz
"""

import argparse
import hashlib
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SPLITS = ["train", "valid", "test", "valid_small"]


def find_split_dir(ml1m_arg):
    cands = [ml1m_arg, os.path.join(ml1m_arg, "ml-1m")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "train_ood2.pkl")):
            return c
    raise FileNotFoundError(f"no train_ood2.pkl under {ml1m_arg} (or {ml1m_arg}/ml-1m)")


def load_splits(split_dir):
    frames = []
    for s in SPLITS:
        p = os.path.join(split_dir, f"{s}_ood2.pkl")
        if os.path.isfile(p):
            df = pd.read_pickle(p)
            print(f"[data] {os.path.basename(p)}: {df.shape}")
            frames.append(df[["uid", "iid", "his", "title", "his_title"]])
        else:
            print(f"[data] {s}_ood2.pkl missing, skipped")
    return pd.concat(frames, ignore_index=True)


def build_item_title_map(df):
    m = {}
    tgt = df[["iid", "title"]].dropna()
    for iid, t in zip(tgt["iid"].astype(int), tgt["title"].astype(str)):
        m.setdefault(iid, str(t))
    his_iids = np.concatenate([np.asarray(h, dtype=int) for h in df["his"]])
    his_titles = []
    for hl in df["his_title"]:
        for x in hl:
            his_titles.append(str(x))
    for iid, t in zip(his_iids, his_titles):
        if t and t != "":
            m.setdefault(int(iid), t)
    return m


def mock_text_embeddings(titles, mf_emb, signal):
    rng_master = np.random.default_rng(1234)
    d_out = mf_emb.shape[1] * 4
    if signal <= 0:
        return np.stack([
            np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).standard_normal(d_out)
            for t in titles
        ])
    W_true = rng_master.standard_normal((mf_emb.shape[1], d_out)) / np.sqrt(mf_emb.shape[1])
    S = mf_emb @ W_true
    snr = signal / max(1.0 - signal, 1e-6)
    noise = rng_master.standard_normal(S.shape) * (S.std() / np.sqrt(snr))
    return S + noise


def vicuna_text_embeddings(titles, llama_model, batch_size=32, max_len=48):
    import torch
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from minigpt4.models.modeling_llama import LlamaForCausalLM
        print("[emb] using vendored minigpt4.models.modeling_llama")
    except Exception as e:
        from transformers import LlamaForCausalLM
        print(f"[emb] vendored import failed ({e}); falling back to transformers")
    from transformers import LlamaTokenizer

    tok = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(
        llama_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[emb] model on {device}")

    outs = []
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            chunk = titles[i:i + batch_size]
            enc = tok(list(chunk), return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"]).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outs.append(pooled.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"[emb] {i + len(chunk)}/{len(titles)}")
    return np.concatenate(outs, axis=0)


def oof_ridge(X, Y, n_folds=5, seed=42):
    alphas = np.logspace(-3, 4, 15)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_alpha, best_score = None, -np.inf
    for a in alphas:
        scores = []
        for tr, te in kf.split(X):
            sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
            r = Ridge(alpha=a).fit(sx.transform(X[tr]), sy.transform(Y[tr]))
            pred = r.predict(sx.transform(X[te]))
            true = sy.transform(Y[te])
            ss_res = ((true - pred) ** 2).sum()
            ss_tot = ((true - true.mean(axis=0)) ** 2).sum()
            scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
        m = float(np.mean(scores))
        if m > best_score:
            best_score, best_alpha = m, a
    preds, trues = [], []
    for tr, te in kf.split(X):
        sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        r = Ridge(alpha=best_alpha).fit(sx.transform(X[tr]), sy.transform(Y[tr]))
        preds.append(r.predict(sx.transform(X[te])))
        trues.append(sy.transform(Y[te]))
    P, T = np.concatenate(preds), np.concatenate(trues)

    def r2_multi(true, pred):
        out = []
        for j in range(true.shape[1]):
            tj, pj = true[:, j], pred[:, j]
            denom = ((tj - tj.mean()) ** 2).sum()
            out.append(1.0 - ((tj - pj) ** 2).sum() / denom if denom > 1e-12 else 0.0)
        return float(np.mean(out))

    return {
        "alpha": best_alpha,
        "mean_fold_r2": best_score,
        "oof_r2_uniform": r2_multi(T, P),
        "ss_res": ((T - P) ** 2).sum(),
        "ss_tot": ((T - T.mean(axis=0)) ** 2).sum(),
        "_pooled_var_weighted": None,
        "P": P, "T": T,
    }


def structure_metrics(A, B, n_sample=1500, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=min(n_sample, len(A)), replace=False)
    A, B = A[idx], B[idx]

    def cos(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return Mn @ Mn.T

    SA, SB = cos(A), cos(B)
    iu = np.triu_indices(len(A), k=1)
    a, b = SA[iu], SB[iu]
    rho_s, _ = spearmanr(a, b)
    rho_p, _ = pearsonr(a, b)

    def topk_mask(S, kk):
        np.fill_diagonal(S, -np.inf)
        m = np.zeros_like(S, dtype=bool)
        rows = np.arange(S.shape[0])[:, None]
        cols = np.argpartition(-S, kk, axis=1)[:, :kk]
        m[rows, cols] = True
        return m

    ma, mb = topk_mask(SA.copy(), k), topk_mask(SB.copy(), k)
    recall = float((ma & mb).sum() / (ma.sum()))
    return {"spearman": float(rho_s), "pearson": float(rho_p), f"recall@{k}": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml1m_zip", default=None)
    ap.add_argument("--ml1m_dir", default=os.environ.get("COLLM_ML1M_DIR"))
    ap.add_argument("--mf_pth",
                    default="weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth")
    ap.add_argument("--text_source", choices=["vicuna", "mock"], default="vicuna")
    ap.add_argument("--llama_model", default=os.environ.get("COLLM_LLAMA_DIR"))
    ap.add_argument("--mock_signal", type=float, default=0.0)
    ap.add_argument("--cache_file", default=None)
    args = ap.parse_args()

    if args.ml1m_dir:
        split_dir = find_split_dir(args.ml1m_dir)
    elif args.ml1m_zip:
        tmp = tempfile.mkdtemp(prefix="ml1m_")
        with zipfile.ZipFile(args.ml1m_zip) as z:
            z.extractall(tmp)
        split_dir = find_split_dir(tmp)
    else:
        raise SystemExit("provide --ml1m_zip or --ml1m_dir")

    import torch
    sd = torch.load(args.mf_pth, map_location="cpu", weights_only=True)
    E = sd["item_embedding.weight"].numpy().astype(np.float64)
    print(f"[mf] item_embedding: {E.shape}")

    df = load_splits(split_dir)
    id2title = build_item_title_map(df)
    print(f"[map] titled items: {len(id2title)} (item_num incl. pad = {E.shape[0]})")

    iids = sorted(i for i in id2title if 0 < i < E.shape[0])
    X = E[np.asarray(iids)]
    titles = [id2title[i] for i in iids]
    print(f"[eval] items used: {len(iids)}, dim MF={X.shape[1]}")

    if args.cache_file and os.path.isfile(args.cache_file):
        c = np.load(args.cache_file, allow_pickle=False)
        assert list(c["iids"]) == list(np.asarray(iids)), "cache iids mismatch"
        Y_raw = c["embs"]
        print(f"[cache] loaded text embs {Y_raw.shape} from {args.cache_file}")
    else:
        if args.text_source == "vicuna":
            if not args.llama_model:
                raise SystemExit("--text-source vicuna requires --llama_model/COLLM_LLAMA_DIR")
            Y_raw = vicuna_text_embeddings(titles, args.llama_model)
        else:
            Y_raw = mock_text_embeddings(titles, X, args.mock_signal)
            print(f"[mock] signal={args.mock_signal}")
        if args.cache_file:
            np.savez_compressed(args.cache_file, iids=np.asarray(iids), embs=Y_raw)
            print(f"[cache] saved {args.cache_file}")

    print("\n=== STEP 0 REPORT ===")
    print(f"items evaluated        : {len(iids)} / {E.shape[0] - 1} non-pad")
    print(f"text dim               : {Y_raw.shape[1]}")

    res = oof_ridge(X, Y_raw)
    vw = 1.0 - res["ss_res"] / max(res["ss_tot"], 1e-12)
    print("---- ridge MF -> text (5-fold OOF) ----")
    print(f"best alpha             : {res['alpha']:.4g}")
    print(f"mean fold R2           : {res['mean_fold_r2']:.4f}")
    print(f"OOF R2 (uniform avg)   : {res['oof_r2_uniform']:.4f}")
    print(f"OOF R2 (var weighted)  : {vw:.4f}")

    sm = structure_metrics(X, Y_raw)
    print("---- similarity-structure agreement ----")
    for k_, v_ in sm.items():
        print(f"{k_:<23}: {v_:.4f}")
    print("======================")


if __name__ == "__main__":
    main()
