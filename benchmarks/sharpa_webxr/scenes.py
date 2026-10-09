"""Materialize published WebXR-Teleop scenes from the live GCS store into a local folder.

Mirrors ``mujoco_webxr_teleop.cache.packages.materialize`` without the cache server/database:
``scenes/<id>/current.json`` -> revision manifest -> scene files + asset ZIPs (extracted under
``assets/<zip stem>/``) + ``scene.xml.poses.json`` from the manifest layout.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIVE = "gs://foundational-research/webxr-teleop/live"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "webxr_scenes"


def _cat(uri: str) -> bytes:
    return subprocess.run(["gsutil", "-q", "cat", uri], check=True, capture_output=True).stdout


def list_scenes() -> list[str]:
    out = subprocess.run(["gsutil", "ls", f"{LIVE}/scenes/"], check=True, capture_output=True, text=True).stdout
    return sorted(line.rstrip("/").split("/")[-1] for line in out.split())


def _within(base: Path, rel: str) -> Path:
    """``base / rel`` resolved, or ValueError if it escapes ``base`` (absolute paths, ``..``, symlinks)."""
    target = (base / rel).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"unsafe path {rel!r}: resolves outside {base}")
    return target


def _check_sha(data: bytes, want: str, what: str) -> None:
    if hashlib.sha256(data).hexdigest() != want:
        raise ValueError(f"{what}: sha256 mismatch")


def fetch_scene(scene_id: str, root: Path = DEFAULT_ROOT) -> Path:
    """Download one scene (idempotent: a complete folder with a matching revision is reused)."""
    folder = root / scene_id
    current = json.loads(_cat(f"{LIVE}/scenes/{scene_id}/current.json"))
    stamp = folder / ".revision.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("revision") == current["revision"]:
        return folder
    rev = f"{LIVE}/scenes/{scene_id}/revisions/{current['revision']}"
    manifest_bytes = _cat(f"{rev}/manifest.json")
    _check_sha(manifest_bytes, current["manifest_sha256"], f"{scene_id} manifest")
    manifest = json.loads(manifest_bytes)
    folder.mkdir(parents=True, exist_ok=True)
    for name, ref in manifest["files"].items():
        target = _within(folder, name)  # manifest names come from the store: never write outside the scene dir
        data = _cat(f"{rev}/{name}")
        _check_sha(data, ref["sha256"], f"{scene_id}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for ref in manifest.get("asset_packages", []):
        blob = _cat(f"{LIVE}/{ref['key']}")
        _check_sha(blob, ref["sha256"], ref["key"])
        dest = _within(folder / "assets", Path(ref["key"]).stem)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                target = _within(dest, member.filename)  # zip-slip: unconditional check (not an assert)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    if manifest.get("layout") is not None:
        (folder / "scene.xml.poses.json").write_text(json.dumps(manifest["layout"]))
    (folder / "scene_manifest.json").write_text(json.dumps(manifest, indent=1))
    stamp.write_text(json.dumps(current))
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenes", nargs="*", help="scene ids (default: all live scenes)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    ids = a.scenes or list_scenes()
    with ThreadPoolExecutor(a.workers) as ex:
        for sid, path in zip(ids, ex.map(lambda s: fetch_scene(s, a.root), ids)):
            print(sid, "->", path)


if __name__ == "__main__":
    main()
