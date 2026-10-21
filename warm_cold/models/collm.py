"""Export full-test CoLLM logits for the V-B2 warm/cold pipeline.

The exporter is deliberately single-process and deterministic: ``shuffle=False``,
``drop_last=False``, and partial CSV checkpoints after every N batches. It uses
the existing repo model/config stack, so Kaggle must have the full environment.

LoRA uses ``train_configs/collm_eval_mf_ood.yaml`` and
``COLLM_STAGE2_CKPT``. MiCA uses ``train_configs/collm_mica_eval_mf_ood.yaml``
and the frozen merged ``COLLM_MICA_STAGE2_CKPT`` (``merged_mica_s12.pth``).
"""

import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from warm_cold.data import DEFAULT_THRESHOLD, load_split
from warm_cold.predictions import read_predictions, write_predictions
from warm_cold.provenance import (
    checkpoint_info,
    config_info,
    environment_info,
    git_commit,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class CollmExportError(RuntimeError):
    """Raised when a CoLLM export cannot follow the V-B2 contract."""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    arch: str
    cfg_path: Path
    mica: bool


MODEL_SPECS = {
    "collm-lora": ModelSpec(
        label="CoLLM-LoRA",
        arch="mini_gpt4rec_v2",
        cfg_path=Path("train_configs/collm_eval_mf_ood.yaml"),
        mica=False,
    ),
    "collm-mica": ModelSpec(
        label="CoLLM-MICA",
        arch="mini_gpt4rec_mica",
        cfg_path=Path("train_configs/collm_mica_eval_mf_ood.yaml"),
        mica=True,
    ),
}


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    if key not in MODEL_SPECS:
        raise CollmExportError(
            f"unknown CoLLM model {name!r}; choose one of {sorted(MODEL_SPECS)}"
        )
    return key


def resolve_model_spec(
    model_name: str, cfg_path: Optional[str] = None, repo_root: Optional[Path] = None
):
    spec = MODEL_SPECS[normalize_model_name(model_name)]
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    cfg = Path(cfg_path) if cfg_path else spec.cfg_path
    if not cfg.is_absolute():
        cfg = root / cfg
    return spec, cfg.expanduser().resolve()


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    try:
        return node.get(key, default)
    except AttributeError:
        return getattr(node, key, default)


def validate_single_process(run_cfg, environ=None) -> None:
    env = os.environ if environ is None else environ
    if bool(_cfg_get(run_cfg, "distributed", False)):
        raise CollmExportError(
            "run.distributed=True is not supported for export: V-B2 prediction "
            "CSVs are row-aligned and must be produced by a single process. "
            "Set run.distributed=False and do not launch with torchrun."
        )
    try:
        world_size = int(_cfg_get(run_cfg, "world_size", 1))
    except (TypeError, ValueError):
        world_size = 1
    if world_size != 1:
        raise CollmExportError(
            f"run.world_size={world_size} is not supported for export; use one process."
        )

    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        value = env.get(key)
        if value in (None, ""):
            continue
        try:
            number = int(value)
        except ValueError:
            continue
        if key == "WORLD_SIZE" and number > 1:
            raise CollmExportError(
                f"{key}={number} indicates a distributed launch; export requires one process."
            )
        if key in ("RANK", "LOCAL_RANK") and number > 0:
            raise CollmExportError(
                f"{key}={number} indicates a distributed worker; export requires one process."
            )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise CollmExportError(
            "torch.distributed is initialized; export requires a single process."
        )


def validate_lora_config(model_cfg) -> None:
    lora = _cfg_get(model_cfg, "lora_config")
    if not bool(_cfg_get(lora, "use_lora", False)):
        raise CollmExportError("lora_config.use_lora must be true")


def validate_mica_config(model_cfg):
    arch = str(_cfg_get(model_cfg, "arch", ""))
    if arch != "mini_gpt4rec_mica":
        raise CollmExportError(
            f"MiCA export requires arch == 'mini_gpt4rec_mica', got {arch!r}"
        )
    validate_lora_config(model_cfg)
    lora = _cfg_get(model_cfg, "lora_config")
    if not bool(_cfg_get(lora, "mica", False)):
        raise CollmExportError("MiCA export requires lora_config.mica == true")
    return lora


def mica_lora_b_stats(named_parameters) -> dict:
    items = named_parameters.items() if hasattr(named_parameters, "items") else named_parameters
    total = 0
    nonzero = 0
    zeros = []
    for name, param in items:
        if "lora_B" not in name:
            continue
        if not torch.is_tensor(param) or param.numel() == 0:
            continue
        total += 1
        if int(torch.count_nonzero(param).item()) > 0:
            nonzero += 1
        else:
            zeros.append(name)
    return {"total": total, "nonzero": nonzero, "zero_names": zeros}


def validate_mica_lora_b(model, expected_patched: Optional[int] = None) -> dict:
    stats = mica_lora_b_stats(model.named_parameters())
    if stats["total"] == 0:
        raise CollmExportError("MiCA export found no lora_B parameters")
    if stats["nonzero"] != stats["total"]:
        raise CollmExportError(
            "MiCA export found zero-valued lora_B after SVD: "
            f"{stats['zero_names'][:3]} ({stats['nonzero']}/{stats['total']} nonzero)"
        )
    patched = getattr(model, "mica_patched_modules", None)
    if patched is None:
        raise CollmExportError(
            "MiCA model does not expose mica_patched_modules; cannot verify SVD patch count"
        )
    if expected_patched is not None and int(patched) != int(expected_patched):
        raise CollmExportError(
            f"MiCA patched-module count {patched} does not match expected {expected_patched}"
        )
    if int(patched) != stats["total"]:
        raise CollmExportError(
            f"MiCA patched {patched} modules but found {stats['total']} lora_B parameters"
        )
    return dict(stats, patched=int(patched))


def _as_repo_path(value, repo_root: Path) -> Path:
    if value in (None, ""):
        raise CollmExportError("required config path is empty")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_config_paths(cfg, repo_root: Path) -> None:
    model_cfg = cfg.model_cfg
    prompt = _as_repo_path(_cfg_get(model_cfg, "prompt_path"), repo_root)
    if not prompt.is_file():
        raise CollmExportError(f"prompt_path does not exist: {prompt}")
    model_cfg.prompt_path = str(prompt)

    llama = _as_repo_path(_cfg_get(model_cfg, "llama_model"), repo_root)
    if not llama.exists():
        raise CollmExportError(f"llama_model does not exist: {llama}")
    model_cfg.llama_model = str(llama)

    ckpt = _as_repo_path(_cfg_get(model_cfg, "ckpt"), repo_root)
    if not ckpt.is_file():
        raise CollmExportError(f"model.ckpt does not exist: {ckpt}")
    model_cfg.ckpt = str(ckpt)

    rec_cfg = _cfg_get(model_cfg, "rec_config")
    rec_path_value = _cfg_get(rec_cfg, "pretrained_path")
    if str(rec_path_value) != "not_have":
        rec_path = _as_repo_path(rec_path_value, repo_root)
        if not rec_path.is_file():
            raise CollmExportError(
                f"model.rec_config.pretrained_path does not exist: {rec_path}"
            )
        rec_cfg.pretrained_path = str(rec_path)


def load_collm_config(
    model_name: str,
    cfg_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
):
    spec, resolved_cfg = resolve_model_spec(model_name, cfg_path, repo_root)
    if not resolved_cfg.is_file():
        raise CollmExportError(f"config file not found: {resolved_cfg}")

    from minigpt4.common.config import Config

    if spec.mica:
        try:
            import minigpt4.models.minigpt4rec_mica  # noqa: F401
        except Exception as exc:
            raise CollmExportError(
                "failed to import minigpt4.models.minigpt4rec_mica; "
                "run in the full Kaggle environment"
            ) from exc

    cfg = Config(SimpleNamespace(cfg_path=str(resolved_cfg), options=None))
    arch = str(_cfg_get(cfg.model_cfg, "arch", ""))
    if arch != spec.arch:
        raise CollmExportError(
            f"{spec.label} export requires arch == {spec.arch!r}, got {arch!r}"
        )
    validate_single_process(cfg.run_cfg)
    mode = str(_cfg_get(cfg.run_cfg, "mode", "v2"))
    if mode != "v2":
        raise CollmExportError(f"run.mode must be 'v2', got {mode!r}")
    if spec.mica:
        validate_mica_config(cfg.model_cfg)
        ckpt_name = Path(str(_cfg_get(cfg.model_cfg, "ckpt", ""))).name
        if ckpt_name != "merged_mica_s12.pth":
            raise CollmExportError(
                "MiCA export requires the frozen merged checkpoint "
                f"'merged_mica_s12.pth', got {ckpt_name!r}"
            )
    else:
        validate_lora_config(cfg.model_cfg)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    resolve_config_paths(cfg, root.resolve())
    return cfg, spec, resolved_cfg


def infer_id_counts(
    data_dir: str,
    train_file: str = "train_ood2.pkl",
    test_file: str = "test_ood2.pkl",
    valid_file: str = "valid_ood2.pkl",
) -> tuple:
    root = Path(data_dir)
    frames = [load_split(data_dir, train_file), load_split(data_dir, test_file)]
    if (root / valid_file).is_file():
        frames.append(load_split(data_dir, valid_file))
    user_num = int(max(int(frame["uid"].max()) for frame in frames)) + 1
    item_num = int(max(int(frame["iid"].max()) for frame in frames)) + 1
    return user_num, item_num


def build_collm_model(
    model_name: str,
    data_dir: str,
    cfg_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
    loaded=None,
):
    if loaded is None:
        cfg, spec, resolved_cfg = load_collm_config(model_name, cfg_path, repo_root)
        user_num, item_num = infer_id_counts(data_dir)
        cfg.model_cfg.rec_config.user_num = user_num
        cfg.model_cfg.rec_config.item_num = item_num
    else:
        cfg, spec, resolved_cfg = loaded

    from minigpt4.common.registry import registry

    model_cls = registry.get_model_class(spec.arch)
    if model_cls is None:
        raise CollmExportError(f"model architecture is not registered: {spec.arch}")
    model = model_cls.from_config(cfg.model_cfg)
    model.set_mode("v2")

    mica_stats = None
    if spec.mica:
        mica_stats = validate_mica_lora_b(model)
        print(
            f"MiCA patched modules: {mica_stats['patched']} | "
            f"lora_B nonzero: {mica_stats['nonzero']}/{mica_stats['total']}"
        )
    return model, cfg, spec, resolved_cfg, mica_stats


def resume_plan(n_done: int, n_total: int, batch_size: int) -> tuple:
    if batch_size <= 0:
        raise CollmExportError(f"batch_size must be positive, got {batch_size}")
    if n_done < 0 or n_done > n_total:
        raise CollmExportError(f"partial row count {n_done} is outside [0, {n_total}]")
    if n_done == 0:
        return 0, False
    if n_done == n_total:
        return int(math.ceil(n_done / batch_size)), True
    if n_done % batch_size != 0:
        raise CollmExportError(
            f"partial has {n_done} rows, which is not a completed batch boundary "
            f"for batch_size={batch_size}; refusing to resume"
        )
    return n_done // batch_size, False


def _partial_path(out_path: Path) -> Path:
    return out_path.with_suffix(".partial.csv")


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _read_partial(path: Path, test):
    pred = read_predictions(str(path), test, require_full=False)
    expected = np.arange(len(pred.row_idx), dtype=np.int64)
    if not np.array_equal(pred.row_idx, expected):
        raise CollmExportError(
            "partial predictions must be a contiguous prefix of test rows; "
            "delete the partial file to restart"
        )
    iid = test["iid"].to_numpy()[pred.row_idx]
    return pred.row_idx, pred.users, iid, pred.labels, pred.logits


def _move_samples(samples, device):
    for key, value in samples.items():
        if torch.is_tensor(value):
            samples[key] = value.to(device, non_blocking=True)
    return samples


def _extract_logits(output, batch_len: int) -> np.ndarray:
    if not isinstance(output, dict) or "logits" not in output:
        raise CollmExportError("model output does not contain 'logits'")
    logits = output["logits"]
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    logits = logits.detach().float().cpu().numpy().reshape(-1)
    if logits.size != batch_len:
        raise CollmExportError(
            f"model returned {logits.size} logits for a batch of {batch_len}"
        )
    return logits.astype(np.float64)


def _build_meta(
    spec,
    cfg,
    cfg_path: Path,
    out_path: Path,
    data_dir: str,
    train_file: str,
    test_file: str,
    n_rows: int,
    threshold: int,
    seed: int,
    batch_size: int,
    checkpoint_every: int,
    num_workers: int,
    device: str,
    status: str,
    mica_stats: Optional[dict],
    hash_assets: bool = True,
) -> dict:
    model_cfg = cfg.model_cfg
    rec_cfg = _cfg_get(model_cfg, "rec_config")
    data_root = Path(data_dir)
    stage_ckpt = str(_cfg_get(model_cfg, "ckpt"))
    rec_ckpt = str(_cfg_get(rec_cfg, "pretrained_path"))
    llama = Path(str(_cfg_get(model_cfg, "llama_model")))
    stage_info = checkpoint_info(stage_ckpt) if hash_assets else {
        "path": stage_ckpt, "exists": Path(stage_ckpt).is_file(), "sha256": None,
    }
    rec_info = checkpoint_info(rec_ckpt) if hash_assets else {
        "path": rec_ckpt, "exists": Path(rec_ckpt).is_file(), "sha256": None,
    }
    cfg_info = config_info(str(cfg_path)) if hash_assets else {
        "path": str(cfg_path), "exists": cfg_path.is_file(), "sha256": None,
    }
    data_info = {
        "data_dir": str(data_root),
        "train_file": str(data_root / train_file),
        "test_file": str(data_root / test_file),
    }
    if hash_assets:
        data_info["train_sha256"] = sha256_file(str(data_root / train_file))
        data_info["test_sha256"] = sha256_file(str(data_root / test_file))
    return {
        "model": spec.label,
        "status": status,
        "prediction_file": str(out_path),
        "n_rows": int(n_rows),
        "arch": str(_cfg_get(model_cfg, "arch")),
        "checkpoint": stage_info,
        "mf_checkpoint": rec_info,
        "config": cfg_info,
        "llama_model": {
            "path": str(llama),
            "exists": llama.exists(),
            "is_dir": llama.is_dir(),
        },
        "threshold": int(threshold),
        "seed": int(seed),
        "export": {
            "batch_size": int(batch_size),
            "checkpoint_every_batches": int(checkpoint_every),
            "num_workers": int(num_workers),
            "device": str(device),
            "shuffle": False,
            "drop_last": False,
            "distributed": False,
        },
        "mica": mica_stats,
        "git_commit": git_commit(),
        "environment": environment_info(),
        "data": data_info,
    }


def _read_or_spec_meta(path: Path):
    if not path.is_file():
        return None
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def predict_collm(
    model_name: str,
    data_dir: str,
    out_path: str,
    cfg_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    checkpoint_every: int = 20,
    device: Optional[str] = None,
    overwrite: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
    seed: int = 42,
    train_file: str = "train_ood2.pkl",
    test_file: str = "test_ood2.pkl",
    repo_root: Optional[Path] = None,
) -> dict:
    if checkpoint_every <= 0:
        raise CollmExportError("checkpoint_every must be positive")
    if num_workers < 0:
        raise CollmExportError("num_workers must be non-negative")

    spec, resolved_cfg = resolve_model_spec(model_name, cfg_path, repo_root)
    test = load_split(data_dir, test_file)
    load_split(data_dir, train_file)
    n_total = len(test)
    if n_total == 0:
        raise CollmExportError("test split is empty")

    out = Path(out_path)
    partial = _partial_path(out)
    final_meta_path = _meta_path(out)
    partial_meta_path = _meta_path(partial)
    out.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in (out, final_meta_path, partial, partial_meta_path):
            if path.exists():
                path.unlink()

    if out.is_file():
        read_predictions(str(out), test, require_full=True)
        print(f"Already complete: {out}")
        existing = _read_or_spec_meta(final_meta_path)
        if existing is not None:
            return existing
        return {"prediction_file": str(out), "n_rows": n_total, "status": "complete"}

    if partial.is_file():
        row_idx, users, iids, labels, logits = _read_partial(partial, test)
        n_done = int(row_idx.size)
    else:
        row_idx = np.array([], dtype=np.int64)
        users = np.array([], dtype=test["uid"].dtype)
        iids = np.array([], dtype=test["iid"].dtype)
        labels = np.array([], dtype=test["label"].dtype)
        logits = np.array([], dtype=np.float64)
        n_done = 0

    cfg, spec, resolved_cfg = load_collm_config(model_name, cfg_path, repo_root)
    user_num, item_num = infer_id_counts(data_dir)
    cfg.model_cfg.rec_config.user_num = user_num
    cfg.model_cfg.rec_config.item_num = item_num
    selected_batch = batch_size
    if selected_batch is None:
        selected_batch = int(_cfg_get(cfg.run_cfg, "batch_size_eval", 4))
    if selected_batch <= 0:
        raise CollmExportError(f"batch_size must be positive, got {selected_batch}")

    start_batch, already_complete = resume_plan(n_done, n_total, selected_batch)

    device_name = device or os.environ.get("COLLM_DEVICE") or str(_cfg_get(cfg.run_cfg, "device", "cuda"))
    torch_device = torch.device(device_name)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise CollmExportError("CUDA is not available; CoLLM export requires the Kaggle GPU runtime")
    if num_workers > 0 and torch_device.type != "cuda":
        num_workers = 0

    print(
        f"{spec.label}: {n_total} test rows | batch_size={selected_batch} | "
        f"resume_from={n_done} rows ({start_batch} batches) | device={torch_device}"
    )

    if already_complete:
        write_predictions(str(out), row_idx, users, iids, labels, logits)
        read_predictions(str(out), test, require_full=True)
        for path in (partial, partial_meta_path):
            if path.exists():
                path.unlink()
        meta = _build_meta(
            spec, cfg, resolved_cfg, out, data_dir, train_file, test_file,
            n_total, threshold, seed, selected_batch, checkpoint_every,
            num_workers, str(torch_device), "complete", None,
        )
        write_json(str(final_meta_path), meta)
        print(f"Recovered complete prediction CSV from {partial}: {out}")
        return meta

    model, cfg, spec, resolved_cfg, mica_stats = build_collm_model(
        model_name,
        data_dir,
        cfg_path,
        repo_root,
        loaded=(cfg, spec, resolved_cfg),
    )
    model.to(torch_device)
    model.eval()

    from minigpt4.datasets.datasets.rec_datasets import MoiveOOData

    test_name = Path(test_file).name
    if not test_name.endswith("_ood2.pkl"):
        raise CollmExportError(
            f"CoLLM export expects a test pickle named *_ood2.pkl, got {test_file!r}"
        )
    dataset = MoiveOOData(
        ann_paths=[str(Path(data_dir) / test_name[: -len("_ood2.pkl")])]
    )
    if len(dataset) != n_total:
        raise CollmExportError(
            f"dataset length {len(dataset)} does not match test split length {n_total}"
        )
    loader = DataLoader(
        dataset,
        batch_size=selected_batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=dataset.collater,
        pin_memory=(torch_device.type == "cuda"),
    )
    expected_batches = int(math.ceil(n_total / selected_batch))
    if len(loader) != expected_batches:
        raise CollmExportError(
            f"DataLoader has {len(loader)} batches, expected {expected_batches}"
        )

    uid_all = test["uid"].to_numpy()
    iid_all = test["iid"].to_numpy()
    label_all = test["label"].to_numpy()

    loader_iter = itertools.islice(loader, start_batch, None)
    for offset, batch in enumerate(loader_iter):
        batch_idx = start_batch + offset
        batch_len = len(batch["UserID"])
        row_start = batch_idx * selected_batch
        row_stop = row_start + batch_len
        if row_stop > n_total:
            raise CollmExportError(
                f"batch {batch_idx} ends at row {row_stop}, beyond n_total={n_total}"
            )
        samples = _move_samples(batch, torch_device)
        with torch.no_grad():
            output = model.generate_for_samples_v2(samples)
        batch_logits = _extract_logits(output, batch_len)
        batch_idx_rows = np.arange(row_start, row_stop, dtype=np.int64)
        row_idx = np.concatenate([row_idx, batch_idx_rows])
        users = np.concatenate([users, uid_all[batch_idx_rows]])
        iids = np.concatenate([iids, iid_all[batch_idx_rows]])
        labels = np.concatenate([labels, label_all[batch_idx_rows]])
        logits = np.concatenate([logits, batch_logits])
        n_done = int(row_idx.size)
        print(
            f"{spec.label}: batch {batch_idx + 1}/{expected_batches} | "
            f"rows {n_done}/{n_total}"
        )

        if (batch_idx + 1) % checkpoint_every == 0 and n_done < n_total:
            write_predictions(str(partial), row_idx, users, iids, labels, logits)
            write_json(
                str(partial_meta_path),
                _build_meta(
                    spec, cfg, resolved_cfg, partial, data_dir, train_file, test_file,
                    n_done, threshold, seed, selected_batch, checkpoint_every,
                    num_workers, str(torch_device), "partial", mica_stats,
                    hash_assets=False,
                ),
            )
            print(f"Checkpointed {n_done}/{n_total} rows -> {partial}")
        if n_done == n_total:
            break

    if n_done != n_total:
        write_predictions(str(partial), row_idx, users, iids, labels, logits)
        write_json(
            str(partial_meta_path),
            _build_meta(
                spec, cfg, resolved_cfg, partial, data_dir, train_file, test_file,
                n_done, threshold, seed, selected_batch, checkpoint_every,
                num_workers, str(torch_device), "partial", mica_stats,
                hash_assets=False,
            ),
        )
        raise CollmExportError(
            f"export stopped at {n_done}/{n_total} rows; rerun to resume from {partial}"
        )

    write_predictions(str(out), row_idx, users, iids, labels, logits)
    read_predictions(str(out), test, require_full=True)
    for path in (partial, partial_meta_path):
        if path.exists():
            path.unlink()
    meta = _build_meta(
        spec, cfg, resolved_cfg, out, data_dir, train_file, test_file,
        n_total, threshold, seed, selected_batch, checkpoint_every,
        num_workers, str(torch_device), "complete", mica_stats,
    )
    write_json(str(final_meta_path), meta)
    print(f"Wrote {out} ({n_total} rows)")
    print(f"Wrote {final_meta_path}")
    return meta
