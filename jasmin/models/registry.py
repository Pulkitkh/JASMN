"""Model versioning, approval and rollback (design doc §17).

Each training run produces a new version directory under models/store/ with
the pickled bundle plus metrics. registry.json tracks every version; the
newest *approved* version is what predictions load. `approve`/`rollback`
flip which version is live without deleting anything.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jasmin.config import MODELS_DIR
from jasmin.utils.logging import get_logger

log = get_logger("registry")


class ModelRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or MODELS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    def _read_index(self) -> list[dict]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return []

    def _write_index(self, entries: list[dict]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2))

    def register(self, bundle: dict[str, Any], metrics: dict, approved: bool) -> str:
        entries = self._read_index()
        version = f"v{len(entries) + 1:04d}"
        version_dir = self.root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        with open(version_dir / "bundle.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
        entries.append(
            {
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "approved": approved,
            }
        )
        self._write_index(entries)
        log.info("registered model %s (approved=%s) metrics=%s", version, approved, metrics)
        return version

    def latest_approved(self) -> tuple[str, dict[str, Any]]:
        approved = [e for e in self._read_index() if e["approved"]]
        if not approved:
            raise FileNotFoundError("No approved model. Run `jasmin train` first.")
        entry = approved[-1]
        with open(self.root / entry["version"] / "bundle.pkl", "rb") as fh:
            return entry["version"], pickle.load(fh)

    def list_versions(self) -> list[dict]:
        return self._read_index()

    def set_approval(self, version: str, approved: bool) -> None:
        entries = self._read_index()
        for entry in entries:
            if entry["version"] == version:
                entry["approved"] = approved
                self._write_index(entries)
                log.info("model %s approval -> %s", version, approved)
                return
        raise KeyError(f"unknown model version {version}")

    def rollback(self) -> str:
        """Un-approve the newest approved version, reverting to the previous one."""
        approved = [e for e in self._read_index() if e["approved"]]
        if len(approved) < 2:
            raise RuntimeError("Need at least two approved versions to roll back.")
        self.set_approval(approved[-1]["version"], False)
        return approved[-2]["version"]
