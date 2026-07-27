"""Atomic cache for normalized Databento UTC daily gold bars."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from aupilot.core.hashing import canonical_json_sha256, sha256_file


CANONICAL_COLUMNS = ["ts_event_utc", "open", "high", "low", "close", "volume"]


class DatabentoMarketRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.legacy_cache_path = root / "databento_gold_ohlcv_1d.parquet"
        self.manifest_path = root / "databento_gold_ohlcv_1d.manifest.json"

    @property
    def cache_path(self) -> Path:
        """Return the active object for diagnostics and compatibility tests."""
        if not self.manifest_path.is_file():
            return self.legacy_cache_path
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return self._active_cache_path(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            return self.legacy_cache_path

    def _active_cache_path(self, manifest: dict) -> Path:
        filename = manifest.get("active_parquet")
        if filename is None and manifest.get("schema_version") == "aurumpilot.databento_gold_daily.v1":
            return self.legacy_cache_path
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("Databento cache manifest active object is invalid")
        candidate = (self.root / filename).resolve()
        root = self.root.resolve()
        if candidate.parent != root:
            raise ValueError("Databento cache manifest escapes its storage root")
        return candidate

    @staticmethod
    def validate(frame: pd.DataFrame, *, minimum_rows: int = 60) -> pd.DataFrame:
        missing = set(CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Databento bars are missing canonical fields: {sorted(missing)}")
        normalized = frame[CANONICAL_COLUMNS].copy()
        normalized["ts_event_utc"] = pd.to_datetime(normalized["ts_event_utc"], utc=True, errors="raise")
        for column in ("open", "high", "low", "close", "volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if len(normalized) < minimum_rows:
            raise ValueError(f"Databento daily history requires at least {minimum_rows} rows")
        if normalized["ts_event_utc"].duplicated().any() or not normalized["ts_event_utc"].is_monotonic_increasing:
            raise ValueError("Databento timestamps must be unique and strictly increasing")
        prices = normalized[["open", "high", "low", "close"]]
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
            raise ValueError("Databento OHLC prices must be finite and positive")
        if not np.isfinite(normalized["volume"].to_numpy(dtype=float)).all() or (normalized["volume"] < 0).any():
            raise ValueError("Databento volume must be finite and non-negative")
        if (normalized["high"] < normalized[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError("Databento high violates the OHLC relationship")
        if (normalized["low"] > normalized[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError("Databento low violates the OHLC relationship")
        return normalized.reset_index(drop=True)

    @staticmethod
    def content_sha256(frame: pd.DataFrame) -> str:
        records = json.loads(frame.to_json(orient="records", date_format="iso", date_unit="ns"))
        return canonical_json_sha256(records)

    def write(self, frame: pd.DataFrame, metadata: dict) -> dict:
        normalized = self.validate(frame)
        self.root.mkdir(parents=True, exist_ok=True)
        parquet_tmp: Path | None = None
        manifest_tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".parquet.tmp", delete=False) as handle:
                parquet_tmp = Path(handle.name)
            normalized.to_parquet(parquet_tmp, index=False)
            parquet_sha256 = sha256_file(parquet_tmp)
            active_filename = f"databento_gold_ohlcv_1d.{parquet_sha256[:20].lower()}.parquet"
            active_path = self.root / active_filename
            if active_path.is_file():
                if sha256_file(active_path) != parquet_sha256:
                    raise ValueError("Databento versioned cache object is corrupted")
                parquet_tmp.unlink()
                parquet_tmp = None
            else:
                os.replace(parquet_tmp, active_path)
                parquet_tmp = None
            manifest = {
                "schema_version": "aurumpilot.databento_gold_daily.v2",
                **metadata,
                "active_parquet": active_filename,
                "data_from_utc": normalized["ts_event_utc"].iloc[0].isoformat(),
                "data_until_utc": normalized["ts_event_utc"].iloc[-1].isoformat(),
                "row_count": len(normalized),
                "content_sha256": self.content_sha256(normalized),
                "parquet_sha256": parquet_sha256,
                "is_formal_data": True,
                "quality_status": metadata.get("quality_status", "VALID"),
            }
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.root, suffix=".json.tmp", delete=False
            ) as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                manifest_tmp = Path(handle.name)
            os.replace(manifest_tmp, self.manifest_path)
            manifest_tmp = None
            return manifest
        finally:
            for temporary in (parquet_tmp, manifest_tmp):
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    def read_manifest(self) -> dict:
        """Read the atomic cache pointer without loading the Parquet payload."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError("Verified Databento daily cache is missing")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Databento cache manifest is invalid") from exc
        if manifest.get("schema_version") not in {
            "aurumpilot.databento_gold_daily.v1",
            "aurumpilot.databento_gold_daily.v2",
        }:
            raise ValueError("Databento cache manifest version is unsupported")
        cache_path = self._active_cache_path(manifest)
        if not cache_path.is_file():
            raise FileNotFoundError("Verified Databento daily cache object is missing")
        retrieved = pd.Timestamp(manifest["retrieved_at_utc"])
        if retrieved.tzinfo is None:
            raise ValueError("Databento retrieval timestamp must be timezone-aware")
        manifest["retrieved_at_utc"] = retrieved.tz_convert(UTC).isoformat()
        return manifest

    def read(self) -> tuple[pd.DataFrame, dict]:
        manifest = self.read_manifest()
        cache_path = self._active_cache_path(manifest)
        if manifest.get("parquet_sha256") != sha256_file(cache_path):
            raise ValueError("Databento cache hash does not match its manifest")
        frame = self.validate(pd.read_parquet(cache_path))
        if manifest.get("content_sha256") != self.content_sha256(frame):
            raise ValueError("Databento normalized content hash does not match its manifest")
        if manifest.get("row_count") != len(frame):
            raise ValueError("Databento cache row count does not match its manifest")
        if pd.Timestamp(manifest.get("data_from_utc")) != frame["ts_event_utc"].iloc[0]:
            raise ValueError("Databento cache start does not match its manifest")
        if pd.Timestamp(manifest.get("data_until_utc")) != frame["ts_event_utc"].iloc[-1]:
            raise ValueError("Databento cache end does not match its manifest")
        return frame, manifest
