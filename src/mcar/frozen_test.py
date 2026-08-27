"""Hash-locked manifest and atomic registry helpers for frozen test runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_version": manifest["model_version"],
        "e_final": manifest["e_final"],
        "scheduler_horizon_cycles": manifest["scheduler_horizon_cycles"],
        "ensemble": manifest["ensemble"],
        "members": [
            {
                "seed": member["seed"],
                "config_sha256": member["config_sha256"],
                "checkpoint_sha256": member["checkpoint_sha256"],
            }
            for member in manifest["members"]
        ],
    }


def identity_sha256(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_identity_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _resolve_inside_root(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest resource escapes repository root: {relative}") from error
    return path


def iter_resources(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    resources: list[Mapping[str, Any]] = []
    for member in manifest["members"]:
        resources.extend(
            (
                {"path": member["config"], "sha256": member["config_sha256"]},
                {
                    "path": member["checkpoint"],
                    "sha256": member["checkpoint_sha256"],
                },
            )
        )
    resources.extend(manifest["dataset"]["resources"])
    resources.extend(manifest["evaluator"]["resources"])
    return resources


def load_verified_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("Manifest status must be 'frozen'")
    actual_identity = identity_sha256(manifest)
    if manifest.get("identity_sha256") != actual_identity:
        raise ValueError("Manifest identity SHA-256 mismatch")
    for resource in iter_resources(manifest):
        resource_path = _resolve_inside_root(root, str(resource["path"]))
        if not resource_path.is_file():
            raise FileNotFoundError(resource_path)
        actual = file_sha256(resource_path)
        if actual != str(resource["sha256"]).upper():
            raise ValueError(f"Resource SHA-256 mismatch: {resource['path']}")
    return manifest


def load_verified_registry(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    allowed_states: tuple[str, ...] = ("not_started",),
) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("manifest_identity_sha256") != manifest["identity_sha256"]:
        raise ValueError("Registry identity does not match the frozen manifest")
    if registry.get("state") not in allowed_states:
        raise ValueError(
            f"Registry state {registry.get('state')!r} is not one of {allowed_states}"
        )
    return registry


def atomic_update_registry(path: Path, registry: Mapping[str, Any], state: str, **extra: Any) -> None:
    if state not in {"not_started", "started", "completed", "failed"}:
        raise ValueError(f"Unsupported registry state {state!r}")
    updated = dict(registry)
    updated.update(extra)
    updated["state"] = state
    updated["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
