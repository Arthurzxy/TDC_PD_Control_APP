from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from app.models import AppConfig, PixelParamRecord, SessionMetadata


class ConfigRepository:
    @staticmethod
    def load(path: str | Path) -> AppConfig:
        with open(path, "r", encoding="utf-8") as handle:
            return AppConfig.from_dict(json.load(handle))

    @staticmethod
    def save(path: str | Path, config: AppConfig) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)


class PixelArrayRepository:
    @staticmethod
    def save_json(path: str | Path, records: List[PixelParamRecord]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([record.__dict__ for record in records], handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str | Path) -> List[PixelParamRecord]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [PixelParamRecord(**item) for item in payload]

    @staticmethod
    def save_csv(path: str | Path, records: List[PixelParamRecord]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["addr", "value36", "version", "comment"])
            for record in records:
                writer.writerow([record.addr, record.value36, record.version, record.comment])

    @staticmethod
    def load_csv(path: str | Path) -> List[PixelParamRecord]:
        records: List[PixelParamRecord] = []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    PixelParamRecord(
                        addr=int(row["addr"]),
                        value36=int(row["value36"]),
                        version=row.get("version", "v1"),
                        comment=row.get("comment", ""),
                    )
                )
        return records


class RawDataWriter:
    def __init__(self) -> None:
        self._handle = None
        self.path: Path | None = None

    def open(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")

    def write(self, data: bytes) -> None:
        if self._handle is None:
            return
        self._handle.write(data)

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None


class SessionRepository:
    @staticmethod
    def create_session(
        root_dir: str | Path,
        session_name: str,
        config_snapshot: dict,
        notes: str = "",
    ) -> tuple[Path, SessionMetadata]:
        root = Path(root_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = root / f"{timestamp}_{session_name}"
        session_dir.mkdir(parents=True, exist_ok=True)
        metadata = SessionMetadata(
            session_name=session_name,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config_snapshot=config_snapshot,
            notes=notes,
        )
        SessionRepository.save_metadata(session_dir / "session.json", metadata)
        return session_dir, metadata

    @staticmethod
    def save_metadata(path: str | Path, metadata: SessionMetadata) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_metadata(path: str | Path) -> SessionMetadata:
        with open(path, "r", encoding="utf-8") as handle:
            return SessionMetadata(**json.load(handle))


class AnalysisExportService:
    @staticmethod
    def export_histogram_csv(path: str | Path, histogram: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, histogram.astype(np.uint32), fmt="%d", delimiter=",")

    @staticmethod
    def export_npz(path: str | Path, histograms: dict, image_projection: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, image_projection=image_projection, **histograms)
