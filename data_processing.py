from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from app.models import AppConfig, HistogramSettings, HistogramSnapshot, MarkerMapping, SessionMetadata, StatusPacket, TdcEvent
from app.protocol import PacketParser
from app.storage import AnalysisExportService, RawDataWriter, SessionRepository
from app.usb_link import DeviceService


@dataclass
class ScanState:
    current_row: int = 0
    current_col: int = 0


class ScanStateMachine:
    def __init__(self, mapping: MarkerMapping) -> None:
        self.mapping = mapping
        self.state = ScanState()

    def reset(self) -> None:
        self.state = ScanState()

    def process_event(self, event: TdcEvent) -> Tuple[bool, int, int]:
        if event.event_class == self.mapping.line_event_class:
            self.state.current_row += 1
            self.state.current_col = 0
            return False, self.state.current_row, self.state.current_col
        if event.event_class == self.mapping.pixel_event_class:
            self.state.current_col += 1
            return False, self.state.current_row, self.state.current_col
        return True, self.state.current_row, self.state.current_col


class HistogramBuilder:
    def __init__(self, settings: HistogramSettings) -> None:
        self.settings = settings
        self.histograms: Dict[Tuple[int, int], np.ndarray] = {}
        self.pixel_counts: Dict[Tuple[int, int], int] = {}

    def clear(self) -> None:
        self.histograms.clear()
        self.pixel_counts.clear()

    def bin_index(self, tstop: int) -> int | None:
        idx = (int(tstop) - self.settings.bin_offset) // max(1, self.settings.bin_width_raw)
        if idx < 0 or idx >= self.settings.bin_count:
            return None
        return int(idx)

    def process_measurement(self, row: int, col: int, event: TdcEvent) -> None:
        bin_idx = self.bin_index(event.tstop)
        if bin_idx is None:
            return
        key = (row, col)
        if key not in self.histograms:
            self.histograms[key] = np.zeros(self.settings.bin_count, dtype=np.uint32)
            self.pixel_counts[key] = 0
        self.histograms[key][bin_idx] += 1
        self.pixel_counts[key] += 1

    def snapshot(self, current_row: int, current_col: int) -> HistogramSnapshot:
        if not self.pixel_counts:
            return HistogramSnapshot(current_row=current_row, current_col=current_col)
        max_row = max(row for row, _ in self.pixel_counts.keys())
        max_col = max(col for _, col in self.pixel_counts.keys())
        image = np.zeros((max_row + 1, max_col + 1), dtype=np.uint32)
        for (row, col), count in self.pixel_counts.items():
            image[row, col] = count
        hist_data = {f"{row},{col}": hist.astype(int).tolist() for (row, col), hist in self.histograms.items()}
        return HistogramSnapshot(
            histograms=hist_data,
            image_projection=image.astype(int).tolist(),
            current_row=current_row,
            current_col=current_col,
        )


class SessionReplayer:
    def __init__(self, mapping: MarkerMapping, histogram_settings: HistogramSettings) -> None:
        self.mapping = mapping
        self.histogram_settings = histogram_settings

    def replay(self, raw_file: str | Path) -> HistogramSnapshot:
        parser = PacketParser()
        state_machine = ScanStateMachine(self.mapping)
        hist_builder = HistogramBuilder(self.histogram_settings)
        with open(raw_file, "rb") as handle:
            while True:
                chunk = handle.read(256 * 1024)
                if not chunk:
                    break
                for packet in parser.feed(chunk):
                    for event in packet.tdc_events or []:
                        is_measurement, row, col = state_machine.process_event(event)
                        if is_measurement:
                            hist_builder.process_measurement(row, col, event)
        return hist_builder.snapshot(
            current_row=state_machine.state.current_row,
            current_col=state_machine.state.current_col,
        )


class AcquisitionService(QtCore.QObject):
    histogram_updated = QtCore.pyqtSignal(object)
    recording_state_changed = QtCore.pyqtSignal(bool, str)
    status_packet_updated = QtCore.pyqtSignal(object)

    def __init__(self, device_service: DeviceService, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.config = config
        self.raw_writer = RawDataWriter()
        self.scan_state = ScanStateMachine(config.marker_mapping)
        self.hist_builder = HistogramBuilder(config.histogram_settings)
        self.recording = False
        self.session_dir: Optional[Path] = None
        self.session_metadata: Optional[SessionMetadata] = None

        self.device_service.raw_bytes_received.connect(self._handle_raw_bytes)
        self.device_service.tdc_events_received.connect(self._handle_tdc_events)
        self.device_service.status_received.connect(self._handle_status)

    def start_recording(self, session_name: str = "capture", notes: str = "") -> Path:
        session_dir, metadata = SessionRepository.create_session(
            self.config.save_directory,
            session_name,
            self.config.to_dict(),
            notes,
        )
        raw_path = session_dir / metadata.raw_file_name
        self.raw_writer.open(raw_path)
        self.session_dir = session_dir
        self.session_metadata = metadata
        self.recording = True
        self.scan_state.reset()
        self.hist_builder.clear()
        self.recording_state_changed.emit(True, str(session_dir))
        return session_dir

    def stop_recording(self) -> None:
        self.recording = False
        self.raw_writer.close()
        if self.session_dir is not None and self.session_metadata is not None:
            snapshot = self.current_snapshot()
            npz_path = self.session_dir / self.session_metadata.analysis_file_name
            image = np.array(snapshot.image_projection, dtype=np.uint32)
            histograms = {
                key.replace(",", "_"): np.array(values, dtype=np.uint32)
                for key, values in snapshot.histograms.items()
            }
            AnalysisExportService.export_npz(npz_path, histograms, image)
            self.session_metadata.packet_count = self.device_service.runtime_stats.packet_count
            self.session_metadata.tdc_event_count = self.device_service.runtime_stats.tdc_event_count
            SessionRepository.save_metadata(self.session_dir / "session.json", self.session_metadata)
        self.recording_state_changed.emit(False, "")

    def current_snapshot(self) -> HistogramSnapshot:
        return self.hist_builder.snapshot(
            current_row=self.scan_state.state.current_row,
            current_col=self.scan_state.state.current_col,
        )

    def _handle_raw_bytes(self, data: bytes) -> None:
        if self.recording:
            self.raw_writer.write(data)

    def _handle_tdc_events(self, events: list[TdcEvent]) -> None:
        for event in events:
            is_measurement, row, col = self.scan_state.process_event(event)
            if is_measurement:
                self.hist_builder.process_measurement(row, col, event)
        self.histogram_updated.emit(self.current_snapshot())

    def _handle_status(self, status: StatusPacket) -> None:
        self.status_packet_updated.emit(status)


class AnalysisService(QtCore.QObject):
    replay_completed = QtCore.pyqtSignal(object)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config

    def replay_session(self, session_json_path: str | Path) -> HistogramSnapshot:
        session_json_path = Path(session_json_path)
        metadata = SessionRepository.load_metadata(session_json_path)
        raw_file = session_json_path.parent / metadata.raw_file_name
        replayer = SessionReplayer(self.config.marker_mapping, self.config.histogram_settings)
        snapshot = replayer.replay(raw_file)
        self.replay_completed.emit(snapshot)
        return snapshot
