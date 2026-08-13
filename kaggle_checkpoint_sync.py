"""Periodically push the newest CoLLM checkpoint to a private Kaggle Dataset.

Runs as a background process alongside training. Every `--poll` seconds it
checks whether a newer ``checkpoint_*.pth`` appeared anywhere under
`--ckpt-root` (which receives a per-run timestamped subdirectory). When one
has and at least `--min-interval` seconds have passed since the last upload,
it copies the newest checkpoint into `--sync-dir` (validating it loads) and
calls ``kaggle datasets create|version`` so the weights survive the
/Kaggle /kaggle/working reset when the session ends.

Next session, retrieve with:

    kaggle datasets download -d <user>/<dataset-name> -p /kaggle/working --unzip

and set ``run.resume_ckpt_path`` to the downloaded ``checkpoint_latest.pth``.
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

STATE_DIR = "/kaggle/working"


def get_username():
    for candidate in (
        os.environ.get("KAGGLE_USERNAME"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
        "/root/.kaggle/kaggle.json",
    ):
        if candidate and candidate.endswith(".json") and os.path.isfile(candidate):
            try:
                with open(candidate) as f:
                    return json.load(f).get("username")
            except Exception:
                pass
        elif candidate:
            return candidate
    raise RuntimeError("cannot determine Kaggle username")


def newest_checkpoint(ckpt_root):
    hits = glob.glob(os.path.join(ckpt_root, "**", "checkpoint_*.pth"), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def write_metadata(sync_dir, username, dataset_name):
    with open(os.path.join(sync_dir, "dataset-metadata.json"), "w") as f:
        json.dump(
            {
                "id": "{}/{}".format(username, dataset_name),
                "title": dataset_name.replace("-", " ").title(),
                "licenses": [{"name": "other"}],
            },
            f,
        )


def load_state(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_mtime": 0.0, "last_upload": 0.0, "created": False}


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f)


def upload(sync_dir, dataset_name, state):
    username = get_username()
    write_metadata(sync_dir, username, dataset_name)
    if not state["created"]:
        cmd = ["kaggle", "datasets", "create", "-p", sync_dir, "-m", "created"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            state["created"] = True
            logging.info("created dataset %s/%s", username, dataset_name)
            return True
        # dataset may already exist from a previous session -> fall through
        logging.info("kaggle create rc=%s (may already exist): %s", res.returncode,
                     res.stderr.strip().splitlines()[-1:] if res.stderr else res.stdout.strip().splitlines()[-1:])
    cmd = ["kaggle", "datasets", "version", "-p", sync_dir, "-m", "snapshot"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        state["created"] = True
        logging.info("uploaded snapshot of %s to %s/%s",
                     os.path.basename(sync_dir), username, dataset_name)
        return True
    logging.warning("kaggle version failed rc=%s: %s", res.returncode,
                    (res.stderr or res.stdout).strip().splitlines()[-2:])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--dataset-name", default="collm-stage1-checkpoints")
    ap.add_argument("--sync-dir", default="/kaggle/working/ckpt_sync")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--min-interval", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.sync_dir, exist_ok=True)
    state_path = os.path.join(
        STATE_DIR, ".{}.state.json".format(args.dataset_name)
    )
    state = load_state(state_path)

    while True:
        try:
            ckpt = newest_checkpoint(args.ckpt_root)
            if ckpt is None:
                logging.info("no checkpoint found yet under %s", args.ckpt_root)
            else:
                mtime = os.path.getmtime(ckpt)
                now = time.time()
                if (mtime > state["last_mtime"] and now - state["last_upload"]
                        >= args.min_interval):
                    dest = os.path.join(args.sync_dir, "checkpoint_latest.pth")
                    tmp = dest + ".tmp"
                    # copy to temp then rename so training's torch.save can't race us
                    subprocess.run(["cp", ckpt, tmp], check=True)
                    os.rename(tmp, dest)
                    # validate it actually loads
                    import torch
                    ck = torch.load(dest, map_location="cpu")
                    if "model" not in ck:
                        raise ValueError("checkpoint missing 'model' key")
                    with open(os.path.join(args.sync_dir, "checkpoint_info.txt"), "w") as f:
                        f.write("epoch={}\nsource={}\n".format(ck.get("epoch"), ckpt))
                    if upload(args.sync_dir, args.dataset_name, state):
                        state["last_mtime"] = mtime
                        state["last_upload"] = now
                    save_state(state_path, state)
        except Exception as exc:  # never let the sync crash training
            logging.error("sync iteration failed: %r", exc)
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)