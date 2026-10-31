"""Small provenance helpers shared by the V-B2 scripts."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    cwd = repo_root or str(Path(__file__).resolve().parents[1])
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
    }


def checkpoint_info(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return {"path": str(candidate), "exists": False, "sha256": None, "size_bytes": None}
    return {
        "path": str(candidate),
        "exists": True,
        "sha256": sha256_file(str(candidate)),
        "size_bytes": candidate.stat().st_size,
    }


def config_info(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return {"path": str(candidate), "exists": False, "sha256": None}
    return {
        "path": str(candidate),
        "exists": True,
        "sha256": sha256_file(str(candidate)),
    }


def write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def copy_freeze_file(out_dir: str) -> Optional[str]:
    candidates = [
        Path.cwd() / "freeze_py312.txt",
        Path(__file__).resolve().parents[1] / "freeze_py312.txt",
    ]
    for src in candidates:
        if src.is_file():
            dst = Path(out_dir) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            return str(dst)
    return None
