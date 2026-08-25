"""
Parse CIE-R proxy run logs (log.txt JSONL) into a clean side-by-side table.

Scans COLLM_OUTPUT_DIR for collm-cie-r-* run directories, extracts the
valid AUC / NDCG trajectory from each log.txt, aligns them by epoch, prints a
comparison table vs the lowest-lambda arm (control), and writes a CSV next to it.

Usage (Kaggle, after runs finish):
  !python parse_cie_r_proxies.py --logs_root $COLLM_OUTPUT_DIR

Local test with synthetic fixture:
  python parse_cie_r_proxies.py --selftest
"""

import argparse
import csv
import glob
import json
import os
import re


def parse_log(path):
    """Return (config_dict|None, [(epoch, auc, ndcg), ...]) from a LAVIS-style log.txt."""
    cfg = None
    rows = []
    ep = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "run" in obj and "model" in obj:
                cfg = obj
                continue
            if "valid_agg_metrics" in obj:
                rows.append((ep, float(obj["valid_agg_metrics"]),
                             obj.get("valid_ndcg")))
                ep += int(cfg["run"]["eval_freq"]) if cfg else 2
    return cfg, rows


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default=os.environ.get("COLLM_OUTPUT_DIR", "."))
    ap.add_argument("--control", default=None,
                    help="name of control run dir; default: lexicographically first")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        print("[selftest] OK")
        return

    pattern = os.path.join(args.logs_root, "collm-cie-r-*", "**", "log.txt")
    found = sorted(glob.glob(pattern, recursive=True))
    if not found:
        raise SystemExit(f"no collm-cie-r-*/log.txt under {args.logs_root}")

    runs = {}
    for path in found:
        name = path.replace("\\", "/").split("/")[-3]
        cfg, rows = parse_log(path)
        lam = None
        if cfg:
            lam = cfg.get("model", {}).get("cie_reg_lambda")
        runs[name] = {"lambda": lam, "rows": rows,
                      "auc": {e: a for e, a, _ in rows},
                      "ndcg": {e: n for e, _, n in rows if n is not None}}

    ctrl_name = args.control or min(runs)
    ctrl = runs[ctrl_name]
    max_ep = max((e for r in runs.values() for e, _, _ in r["rows"]), default=0)

    hdr = f"{'epoch':>5} |" + "|".join(
        f" {n} (lam={fmt(r['lambda'])}) " for n, r in runs.items())
    print(hdr)
    print("-" * len(hdr))
    table_rows = []
    for e in range(0, max_ep + 1, 2):
        cells = []
        for n in runs:
            cells.append(fmt(runs[n]["auc"][e]) if e in runs[n]["auc"] else "-")
        deltas = []
        for n in runs:
            if n == ctrl_name:
                continue
            if e in ctrl["auc"] and e in runs[n]["auc"]:
                deltas.append(f"{runs[n]['auc'][e] - ctrl['auc'][e]:+.4f}")
        line = f"{e:>5} |" + "|".join(f" {c:>18} " for c in cells)
        if deltas:
            line += "   delta: " + " ".join(deltas)
        print(line)
        table_rows.append([e] + [runs[n]["auc"].get(e) for n in runs])

    print("-" * len(hdr))
    best_line = []
    for n, r in runs.items():
        vals = [a for _, a, _ in r["rows"]]
        bi = int(max(range(len(vals)), key=lambda i: vals[i]))
        be, bv = r["rows"][bi][0], vals[bi]
        bnd = r["rows"][bi][2]
        best_line.append(f"{n}: best auc {bv:.4f} @ep{be}" + (f" ndcg {bnd:.4f}" if bnd else ""))
    print("BEST: " + " | ".join(best_line))

    out_csv = args.out_csv or os.path.join(
        args.logs_root, f"cie_r_proxy_comparison_{ctrl_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(runs))
        w.writerows(table_rows)
    print(f"[csv] wrote {out_csv}")


def _make_tmp_log(dirname, eval_freq, lam, series, ndcgs=None):
    os.makedirs(os.path.join(dirname, "20260101"), exist_ok=True)
    p = os.path.join(dirname, "20260101", "log.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "run": {"eval_freq": eval_freq},
            "model": {"cie_reg_lambda": lam},
        }) + "\n")
        for i, a in enumerate(series):
            row = {"valid_agg_metrics": a}
            if ndcgs:
                row["valid_ndcg"] = ndcgs[i]
            f.write(json.dumps(row) + "\n")


def _selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="cie_r_parse_test_")
    try:
        _make_tmp_log(os.path.join(root, "collm-cie-r-l001"), 2, 0.01,
                      [0.66, 0.67, 0.6713])
        _make_tmp_log(os.path.join(root, "collm-cie-r-l01"), 2, 0.1,
                      [0.6659, 0.6894, 0.7033])
        args_logs = root
        sys_argv_bak = None
        # reuse main() by invoking with args
        import sys
        argv_bak = sys.argv
        sys.argv = ["parse_cie_r_proxies.py", "--logs_root", args_logs]
        try:
            main()
        finally:
            sys.argv = argv_bak
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
