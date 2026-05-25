from __future__ import annotations

import atexit
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from app.models import (
    AppConfig,
    HistogramSettings,
    HistogramSnapshot,
    MarkerMapping,
    PhotonEvent,
    SessionMetadata,
    StatusPacket,
    TdcHistogramChunk,
    TdcTestSettings,
    TdcEvent,
    TdcRawEventBatch,
)
from app.protocol import PacketParser
from app.storage import AnalysisExportService, RawDataWriter, SessionRepository
from app.usb_link import DeviceService


_ACQUISITION_SERVICES: "weakref.WeakSet[AcquisitionService]" = weakref.WeakSet()


def _shutdown_acquisition_services() -> None:
    for service in list(_ACQUISITION_SERVICES):
        try:
            service.shutdown()
        except Exception:
            pass


atexit.register(_shutdown_acquisition_services)


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
        self.add_bin(row, col, bin_idx)

    def process_photon(self, event: PhotonEvent) -> None:
        self.add_bin(event.line_id, event.pixel_id, event.bin_index)

    def add_bin(self, row: int, col: int, bin_idx: int) -> None:
        if bin_idx < 0 or bin_idx >= self.settings.bin_count:
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


@dataclass(frozen=True)
class TcspcPairingResult:
    dt_raw: np.ndarray
    sync_indices: np.ndarray
    photon_indices: np.ndarray
    paired_photons: int
    orphan_photons: int
    before_history: int
    after_history: int


@dataclass(frozen=True)
class TcspcPeakShape:
    peak_bin: int
    peak_count: int
    peak_ns: float
    fwhm_raw: int
    fwhm_ns: float
    rms_raw: float
    rms_ns: float
    background_mean: float
    background_std: float
    peak_to_bg: float
    second_peak_bin: int
    peak_separation_raw: int
    peak_separation_ns: float
    half_count: float = 0.0
    fwhm_left_raw: float = 0.0
    fwhm_right_raw: float = 0.0
    fwhm_left_ns: float = 0.0
    fwhm_right_ns: float = 0.0


@dataclass(frozen=True)
class TcspcSatellitePeak:
    bin_index: int
    dt_ns: float
    offset_ns: float
    count: int
    start_repeat_hits: int
    stop_repeat_hits: int


@dataclass(frozen=True)
class TcspcCommercialMetrics:
    shape: TcspcPeakShape
    background_mean: float
    background_std: float
    background_cv: float
    peak_to_bg: float
    deadtime_start_ns: float
    deadtime_end_ns: float
    deadtime_violation_count: int
    deadtime_violation_ratio: float
    satellite_decision: str
    satellite_peaks: tuple[TcspcSatellitePeak, ...]


@dataclass(frozen=True)
class TcspcShapeDiagnostic:
    name: str
    time_formula: str
    pairing_mode: str
    accepted_pairs: int
    paired_photons: int
    orphan_photons: int
    pairs_per_photon: float
    shape: TcspcPeakShape
    coarse_ref_delta: Dict[int, int]
    fine_tstop_delta: Dict[int, int]


@dataclass(frozen=True)
class TcspcReplayHistogram:
    name: str
    histogram: np.ndarray
    pairs: int
    paired_photons: int
    orphan_photons: int
    before_history: int
    after_history: int
    sync_count: int
    stop_count: int
    reference_filtered: int
    deadtime_filtered: int
    shape: TcspcPeakShape
    commercial: TcspcCommercialMetrics


@dataclass(frozen=True)
class TcspcRawReplayResult:
    settings: TdcTestSettings
    channel_counts: tuple[int, int, int, int]
    timestamp_regressions: int
    stream_prev_start: Optional[TcspcReplayHistogram]
    sorted_prev_start: TcspcReplayHistogram
    sorted_prev_start_deadtime: TcspcReplayHistogram
    sorted_all_start: TcspcReplayHistogram
    start_repeat_histogram: np.ndarray
    stop_repeat_histogram: np.ndarray
    shape_diagnostics: tuple[TcspcShapeDiagnostic, ...]
    stream_vs_sorted: Dict[str, Any]
    flags: tuple[str, ...]
    decision: str


class TcspcTimeReconstructor:
    TICK_NS = 0.008
    FORMULAS = ("forward", "reverse_same", "reverse_next")
    PAIRING_MODES = ("all_start", "prev_start")

    @staticmethod
    def timestamps_from_formula(
        extended_refids: np.ndarray,
        tstops: np.ndarray,
        refclk_divisions: int,
        formula: str = "forward",
    ) -> np.ndarray:
        refs = np.asarray(extended_refids, dtype=np.int64)
        fine = np.asarray(tstops, dtype=np.int64)
        div = max(1, int(refclk_divisions))
        if formula == "forward":
            return refs * div + fine
        if formula == "reverse_same":
            return refs * div - fine
        if formula == "reverse_next":
            return (refs + 1) * div - fine
        raise ValueError(f"unknown TCSPC time formula: {formula}")

    @staticmethod
    def pair_indices(
        sync_timestamps: np.ndarray,
        photon_timestamps: np.ndarray,
        min_dt_raw: int,
        max_dt_raw: int,
        pairing_mode: str = "prev_start",
    ) -> TcspcPairingResult:
        sync_ts = np.asarray(sync_timestamps, dtype=np.int64)
        photon_ts = np.asarray(photon_timestamps, dtype=np.int64)
        if sync_ts.size > 1 and np.any(sync_ts[1:] < sync_ts[:-1]):
            sync_order = np.argsort(sync_ts, kind="mergesort")
            sync_ts = sync_ts[sync_order]
        else:
            sync_order = np.arange(sync_ts.size, dtype=np.int64)
        if photon_ts.size > 1 and np.any(photon_ts[1:] < photon_ts[:-1]):
            photon_order = np.argsort(photon_ts, kind="mergesort")
            photon_ts = photon_ts[photon_order]
        else:
            photon_order = np.arange(photon_ts.size, dtype=np.int64)

        if sync_ts.size == 0 or photon_ts.size == 0:
            return TcspcPairingResult(
                dt_raw=np.array([], dtype=np.int64),
                sync_indices=np.array([], dtype=np.int64),
                photon_indices=np.array([], dtype=np.int64),
                paired_photons=0,
                orphan_photons=int(photon_ts.size),
                before_history=int(photon_ts.size if sync_ts.size == 0 and photon_ts.size else 0),
                after_history=0,
            )

        mode = pairing_mode if pairing_mode in TcspcTimeReconstructor.PAIRING_MODES else "prev_start"
        min_dt = max(0, int(min_dt_raw))
        max_dt = max(min_dt, int(max_dt_raw))
        dt_chunks: list[np.ndarray] = []
        sync_chunks: list[np.ndarray] = []
        photon_chunks: list[np.ndarray] = []
        paired_photons = 0
        orphan_photons = 0
        before_history = 0
        after_history = 0
        first_sync = int(sync_ts[0])
        last_sync = int(sync_ts[-1])

        for local_photon_idx, photon_timestamp in enumerate(photon_ts):
            photon_value = int(photon_timestamp)
            window_start = photon_value - max_dt
            window_end = photon_value - min_dt
            right = int(np.searchsorted(sync_ts, window_end, side="right"))
            if mode == "prev_start":
                left = max(0, right - 1)
            else:
                left = int(np.searchsorted(sync_ts, window_start, side="left"))
            usable = right > left
            if usable and mode == "prev_start" and int(sync_ts[left]) < window_start:
                usable = False
            if not usable:
                orphan_photons += 1
                if window_end < first_sync:
                    before_history += 1
                elif window_start > last_sync:
                    after_history += 1
                continue

            selected_sync = sync_ts[left:right]
            dt_chunks.append(photon_value - selected_sync)
            sync_chunks.append(sync_order[left:right].astype(np.int64, copy=False))
            photon_chunks.append(
                np.full(right - left, int(photon_order[local_photon_idx]), dtype=np.int64)
            )
            paired_photons += 1

        if dt_chunks:
            dt_raw = np.concatenate(dt_chunks).astype(np.int64, copy=False)
            sync_indices = np.concatenate(sync_chunks).astype(np.int64, copy=False)
            photon_indices = np.concatenate(photon_chunks).astype(np.int64, copy=False)
        else:
            dt_raw = np.array([], dtype=np.int64)
            sync_indices = np.array([], dtype=np.int64)
            photon_indices = np.array([], dtype=np.int64)
        return TcspcPairingResult(
            dt_raw=dt_raw,
            sync_indices=sync_indices,
            photon_indices=photon_indices,
            paired_photons=paired_photons,
            orphan_photons=orphan_photons,
            before_history=before_history,
            after_history=after_history,
        )

    @staticmethod
    def histogram_from_dt(dt_raw: np.ndarray, settings: TdcTestSettings) -> np.ndarray:
        hist = np.zeros(int(settings.bin_count), dtype=np.uint32)
        if hist.size == 0:
            return hist
        values = np.asarray(dt_raw, dtype=np.int64)
        if values.size == 0:
            return hist
        bins = (values - int(settings.bin_offset)) // max(1, int(settings.bin_width_raw))
        valid = (bins >= 0) & (bins < hist.size)
        if not np.any(valid):
            return hist
        counts = np.bincount(bins[valid].astype(np.int64, copy=False), minlength=hist.size)
        hist[:] = counts[: hist.size].astype(np.uint32, copy=False)
        return hist

    @staticmethod
    def _compact_counter(values: np.ndarray, limit: int = 6) -> Dict[int, int]:
        if values.size == 0:
            return {}
        unique, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
        order = np.argsort(counts)[::-1][:limit]
        return {int(unique[idx]): int(counts[idx]) for idx in order}

    @staticmethod
    def measure_histogram_shape(
        histogram: np.ndarray,
        settings: TdcTestSettings,
        *,
        min_bin: int = 0,
        expected_peak_separation_raw: Optional[int] = None,
    ) -> TcspcPeakShape:
        values = np.asarray(histogram, dtype=np.uint32)
        if values.size == 0 or int(values.sum()) == 0:
            return TcspcPeakShape(-1, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0, 0.0)

        start_bin = min(max(0, int(min_bin)), values.size - 1)
        view = values[start_bin:]
        peak_bin = start_bin + int(view.argmax())
        peak_count = int(values[peak_bin])
        bin_width = max(1, int(settings.bin_width_raw))
        bin_offset = int(settings.bin_offset)
        peak_ns = (peak_bin * bin_width + bin_offset) * TcspcTimeReconstructor.TICK_NS

        second_peak_bin = -1
        peak_separation_raw = 0
        rough_half = peak_count / 2.0
        rough_left = peak_bin
        while rough_left > start_bin and float(values[rough_left]) >= rough_half:
            rough_left -= 1
        if rough_left < peak_bin and float(values[rough_left]) < rough_half:
            rough_left += 1
        rough_right = peak_bin
        while rough_right + 1 < values.size and float(values[rough_right]) >= rough_half:
            rough_right += 1
        if rough_right > peak_bin and float(values[rough_right]) < rough_half:
            rough_right -= 1
        rough_fwhm_bins = max(1, rough_right - rough_left + 1)
        if expected_peak_separation_raw and expected_peak_separation_raw > 0:
            period_bins = max(1, int(round(expected_peak_separation_raw / bin_width)))
            search_half = max(3, int(round(period_bins * 0.10)))
            candidates: list[tuple[int, int]] = []
            for target in (peak_bin - period_bins, peak_bin + period_bins):
                lo = max(start_bin, target - search_half)
                hi = min(values.size, target + search_half + 1)
                if hi > lo:
                    local = lo + int(values[lo:hi].argmax())
                    candidates.append((int(values[local]), local))
            if candidates:
                _count, second_peak_bin = max(candidates, key=lambda item: item[0])
        if second_peak_bin < 0:
            suppress = max(5, rough_fwhm_bins * 2)
            masked = values.astype(np.int64, copy=True)
            masked[max(start_bin, peak_bin - suppress) : min(values.size, peak_bin + suppress + 1)] = -1
            if int(masked.max()) >= 0:
                second_peak_bin = int(masked.argmax())
        if second_peak_bin >= 0:
            peak_separation_raw = abs(second_peak_bin - peak_bin) * bin_width

        bg_mask = np.ones(values.size, dtype=bool)
        guard = max(5, rough_fwhm_bins * 2)
        bg_mask[max(start_bin, peak_bin - guard) : min(values.size, peak_bin + guard + 1)] = False
        if second_peak_bin >= 0:
            bg_mask[
                max(start_bin, second_peak_bin - guard) : min(values.size, second_peak_bin + guard + 1)
            ] = False
        bg_mask[:start_bin] = False
        background = values[bg_mask].astype(float, copy=False)
        bg_mean = float(background.mean()) if background.size else 0.0
        bg_std = float(background.std()) if background.size else 0.0
        peak_to_bg = float(peak_count) / bg_mean if bg_mean > 0.0 else float(peak_count)
        half_count = bg_mean + max(0.0, float(peak_count) - bg_mean) / 2.0

        centers_raw = (np.arange(values.size, dtype=float) * bin_width) + bin_offset

        def _interpolate_crossing(low_bin: int, high_bin: int) -> float:
            x0 = float(centers_raw[low_bin])
            y0 = float(values[low_bin])
            x1 = float(centers_raw[high_bin])
            y1 = float(values[high_bin])
            if y1 == y0:
                return x0
            ratio = (half_count - y0) / (y1 - y0)
            ratio = min(1.0, max(0.0, ratio))
            return x0 + ratio * (x1 - x0)

        left_index = peak_bin
        while left_index > start_bin and float(values[left_index]) >= half_count:
            left_index -= 1
        if left_index == start_bin and float(values[left_index]) >= half_count:
            fwhm_left_raw = float(centers_raw[left_index])
        elif left_index == peak_bin:
            fwhm_left_raw = float(centers_raw[peak_bin])
        else:
            fwhm_left_raw = _interpolate_crossing(left_index, left_index + 1)

        right_index = peak_bin
        while right_index < values.size - 1 and float(values[right_index]) >= half_count:
            right_index += 1
        if right_index == values.size - 1 and float(values[right_index]) >= half_count:
            fwhm_right_raw = float(centers_raw[right_index])
        elif right_index == peak_bin:
            fwhm_right_raw = float(centers_raw[peak_bin])
        else:
            fwhm_right_raw = _interpolate_crossing(right_index - 1, right_index)

        fwhm_raw_float = max(0.0, fwhm_right_raw - fwhm_left_raw)
        fwhm_raw = int(round(fwhm_raw_float))
        fwhm_ns = fwhm_raw_float * TcspcTimeReconstructor.TICK_NS

        region_left = max(start_bin, int(np.searchsorted(centers_raw, fwhm_left_raw, side="left")))
        region_right = min(values.size - 1, int(np.searchsorted(centers_raw, fwhm_right_raw, side="right")))
        if region_right < region_left:
            region_left = region_right = peak_bin
        region_bins = np.arange(region_left, region_right + 1, dtype=np.int64)
        region_weights = values[region_left : region_right + 1].astype(float, copy=False)
        if region_weights.sum() > 0:
            region_centers_raw = (region_bins * bin_width + bin_offset).astype(float, copy=False)
            peak_raw = peak_bin * bin_width + bin_offset
            rms_raw = float(np.sqrt(np.average((region_centers_raw - peak_raw) ** 2, weights=region_weights)))
        else:
            rms_raw = 0.0
        return TcspcPeakShape(
            peak_bin=peak_bin,
            peak_count=peak_count,
            peak_ns=peak_ns,
            fwhm_raw=fwhm_raw,
            fwhm_ns=fwhm_ns,
            rms_raw=rms_raw,
            rms_ns=rms_raw * TcspcTimeReconstructor.TICK_NS,
            background_mean=bg_mean,
            background_std=bg_std,
            peak_to_bg=peak_to_bg,
            second_peak_bin=second_peak_bin,
            peak_separation_raw=int(peak_separation_raw),
            peak_separation_ns=peak_separation_raw * TcspcTimeReconstructor.TICK_NS,
            half_count=half_count,
            fwhm_left_raw=fwhm_left_raw,
            fwhm_right_raw=fwhm_right_raw,
            fwhm_left_ns=fwhm_left_raw * TcspcTimeReconstructor.TICK_NS,
            fwhm_right_ns=fwhm_right_raw * TcspcTimeReconstructor.TICK_NS,
        )

    @staticmethod
    def estimate_sync_period_raw(sync_timestamps: np.ndarray) -> Optional[int]:
        values = np.asarray(sync_timestamps, dtype=np.int64)
        if values.size < 2:
            return None
        diffs = np.diff(values[-4096:])
        positive = diffs[diffs > 0]
        if positive.size == 0:
            return None
        return int(np.median(positive))

    @staticmethod
    def shape_diagnostics(
        channels: np.ndarray,
        extended_refids: np.ndarray,
        tstops: np.ndarray,
        settings: TdcTestSettings,
    ) -> list[TcspcShapeDiagnostic]:
        ch = np.asarray(channels, dtype=np.int64)
        refs = np.asarray(extended_refids, dtype=np.int64)
        fine = np.asarray(tstops, dtype=np.int64)
        start_mask = ch == int(settings.start_channel)
        stop_mask = ch == int(settings.stop_channel)
        diagnostics: list[TcspcShapeDiagnostic] = []
        if not np.any(start_mask) or not np.any(stop_mask):
            return diagnostics
        min_dt, max_dt = TdcTestHistogramBuilder.window_bounds_for_settings(settings)
        for formula in TcspcTimeReconstructor.FORMULAS:
            timestamps = TcspcTimeReconstructor.timestamps_from_formula(
                refs, fine, int(settings.refclk_divisions), formula
            )
            sync_order = np.argsort(timestamps[start_mask], kind="mergesort")
            photon_order = np.argsort(timestamps[stop_mask], kind="mergesort")
            sync_ts = timestamps[start_mask][sync_order]
            photon_ts = timestamps[stop_mask][photon_order]
            sync_refs = refs[start_mask][sync_order]
            stop_refs = refs[stop_mask][photon_order]
            sync_fine = fine[start_mask][sync_order]
            stop_fine = fine[stop_mask][photon_order]
            expected_period = TcspcTimeReconstructor.estimate_sync_period_raw(sync_ts)
            for mode in TcspcTimeReconstructor.PAIRING_MODES:
                pairs = TcspcTimeReconstructor.pair_indices(sync_ts, photon_ts, min_dt, max_dt, mode)
                hist = TcspcTimeReconstructor.histogram_from_dt(pairs.dt_raw, settings)
                shape = TcspcTimeReconstructor.measure_histogram_shape(
                    hist,
                    settings,
                    expected_peak_separation_raw=expected_period,
                )
                coarse = (
                    stop_refs[pairs.photon_indices] - sync_refs[pairs.sync_indices]
                    if pairs.dt_raw.size
                    else np.array([], dtype=np.int64)
                )
                fine_delta = (
                    stop_fine[pairs.photon_indices] - sync_fine[pairs.sync_indices]
                    if pairs.dt_raw.size
                    else np.array([], dtype=np.int64)
                )
                diagnostics.append(
                    TcspcShapeDiagnostic(
                        name=f"{mode}_{formula}",
                        time_formula=formula,
                        pairing_mode=mode,
                        accepted_pairs=int(pairs.dt_raw.size),
                        paired_photons=int(pairs.paired_photons),
                        orphan_photons=int(pairs.orphan_photons),
                        pairs_per_photon=float(pairs.dt_raw.size) / max(1, int(photon_ts.size)),
                        shape=shape,
                        coarse_ref_delta=TcspcTimeReconstructor._compact_counter(coarse),
                        fine_tstop_delta=TcspcTimeReconstructor._compact_counter(fine_delta),
                    )
                )
        return diagnostics


class TcspcCommercialAnalyzer:
    @staticmethod
    def _repeat_hits(histogram: np.ndarray | None, offset_bin: int) -> int:
        if histogram is None or histogram.size == 0:
            return 0
        lo = max(0, int(offset_bin) - 2)
        hi = min(int(histogram.size), int(offset_bin) + 3)
        return int(np.asarray(histogram[lo:hi], dtype=np.uint64).sum())

    @staticmethod
    def analyze(
        histogram: np.ndarray,
        settings: TdcTestSettings,
        *,
        min_bin: int = 0,
        start_repeat_histogram: np.ndarray | None = None,
        stop_repeat_histogram: np.ndarray | None = None,
        expected_peak_separation_raw: Optional[int] = None,
        detector_deadtime_ns: Optional[float] = None,
    ) -> TcspcCommercialMetrics:
        values = np.asarray(histogram, dtype=np.uint32)
        shape = TcspcTimeReconstructor.measure_histogram_shape(
            values,
            settings,
            min_bin=min_bin,
            expected_peak_separation_raw=expected_peak_separation_raw,
        )
        if values.size == 0 or shape.peak_bin < 0 or shape.peak_count <= 0:
            empty = TcspcPeakShape(-1, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0, 0.0)
            return TcspcCommercialMetrics(
                shape=empty,
                background_mean=0.0,
                background_std=0.0,
                background_cv=0.0,
                peak_to_bg=0.0,
                deadtime_start_ns=0.0,
                deadtime_end_ns=0.0,
                deadtime_violation_count=0,
                deadtime_violation_ratio=0.0,
                satellite_decision="no_peak",
                satellite_peaks=(),
            )

        bin_width = max(1, int(settings.bin_width_raw))
        bin_offset = int(settings.bin_offset)
        min_bin = min(max(0, int(min_bin)), max(0, values.size - 1))
        centers_raw = (np.arange(values.size, dtype=float) * bin_width) + bin_offset
        peak_raw = float(shape.peak_bin * bin_width + bin_offset)
        prompt_guard_bins = max(5, int(np.ceil(max(1.0, shape.fwhm_raw) / float(bin_width))) * 3)

        bg_mask = np.ones(values.size, dtype=bool)
        bg_mask[:min_bin] = False
        bg_mask[
            max(min_bin, int(shape.peak_bin) - prompt_guard_bins) : min(
                values.size,
                int(shape.peak_bin) + prompt_guard_bins + 1,
            )
        ] = False
        if shape.second_peak_bin >= 0:
            bg_mask[
                max(min_bin, int(shape.second_peak_bin) - prompt_guard_bins) : min(
                    values.size,
                    int(shape.second_peak_bin) + prompt_guard_bins + 1,
                )
            ] = False
        background = values[bg_mask].astype(float, copy=False)
        bg_mean = float(background.mean()) if background.size else 0.0
        bg_std = float(background.std()) if background.size else 0.0
        bg_cv = bg_std / bg_mean if bg_mean > 0.0 else 0.0
        peak_to_bg = float(shape.peak_count) / bg_mean if bg_mean > 0.0 else float(shape.peak_count)

        deadtime_ns = (
            max(0.0, float(detector_deadtime_ns))
            if detector_deadtime_ns is not None
            else max(0.0, float(getattr(settings, "detector_deadtime_ns", 100.0)))
        )
        deadtime_start_raw = max(float(shape.fwhm_right_raw), peak_raw)
        deadtime_end_raw = peak_raw + (deadtime_ns / TcspcTimeReconstructor.TICK_NS)
        deadtime_mask = (centers_raw > deadtime_start_raw) & (centers_raw <= deadtime_end_raw)
        deadtime_mask[:min_bin] = False
        deadtime_violation_count = int(values[deadtime_mask].sum()) if np.any(deadtime_mask) else 0
        deadtime_violation_ratio = float(deadtime_violation_count) / max(1.0, float(shape.peak_count))

        bin_width_ns = bin_width * TcspcTimeReconstructor.TICK_NS
        threshold = max(
            3,
            int(np.ceil(bg_mean + 6.0 * bg_std)),
            int(np.ceil(float(shape.peak_count) * 0.0001)),
        )
        exclude_bins = max(3, int(round(2.0 / max(bin_width_ns, 1e-12))))
        min_sep_bins = max(1, int(round(5.0 / max(bin_width_ns, 1e-12))))
        search_lo = min(values.size - 1, int(shape.peak_bin) + exclude_bins)
        search_hi = min(
            values.size - 1,
            int(shape.peak_bin) + max(exclude_bins + 1, int(round(700.0 / max(bin_width_ns, 1e-12)))),
        )
        satellites: list[TcspcSatellitePeak] = []
        if 0 < search_lo < search_hi:
            values_i = values.astype(np.int64, copy=False)
            candidate_region = np.arange(search_lo, search_hi, dtype=np.int64)
            is_peak = (
                (values_i[candidate_region] >= values_i[candidate_region - 1])
                & (values_i[candidate_region] >= values_i[candidate_region + 1])
                & (values_i[candidate_region] >= threshold)
            )
            candidates = candidate_region[is_peak]
            if candidates.size:
                candidates = candidates[np.argsort(values_i[candidates])[::-1]]
                selected: list[int] = []
                for idx in candidates:
                    if all(abs(int(idx) - existing) >= min_sep_bins for existing in selected):
                        selected.append(int(idx))
                    if len(selected) >= 6:
                        break
                selected.sort()
                for idx in selected:
                    offset_bin = max(0, int(idx) - int(shape.peak_bin))
                    offset_ns = offset_bin * bin_width_ns
                    dt_ns = (idx * bin_width + bin_offset) * TcspcTimeReconstructor.TICK_NS
                    satellites.append(
                        TcspcSatellitePeak(
                            bin_index=int(idx),
                            dt_ns=float(dt_ns),
                            offset_ns=float(offset_ns),
                            count=int(values_i[int(idx)]),
                            start_repeat_hits=TcspcCommercialAnalyzer._repeat_hits(
                                start_repeat_histogram,
                                offset_bin,
                            ),
                            stop_repeat_hits=TcspcCommercialAnalyzer._repeat_hits(
                                stop_repeat_histogram,
                                offset_bin,
                            ),
                        )
                    )

        start_hits = sum(item.start_repeat_hits for item in satellites)
        stop_hits = sum(item.stop_repeat_hits for item in satellites)
        if not satellites:
            decision = "no_significant_satellites"
        elif stop_hits > max(5, start_hits * 3):
            decision = "likely_stop_channel_retrigger_or_afterpulse"
        elif start_hits > max(5, stop_hits * 3):
            decision = "likely_start_channel_retrigger"
        else:
            decision = "satellites_present_source_inconclusive"

        return TcspcCommercialMetrics(
            shape=shape,
            background_mean=bg_mean,
            background_std=bg_std,
            background_cv=bg_cv,
            peak_to_bg=peak_to_bg,
            deadtime_start_ns=deadtime_start_raw * TcspcTimeReconstructor.TICK_NS,
            deadtime_end_ns=deadtime_end_raw * TcspcTimeReconstructor.TICK_NS,
            deadtime_violation_count=deadtime_violation_count,
            deadtime_violation_ratio=deadtime_violation_ratio,
            satellite_decision=decision,
            satellite_peaks=tuple(satellites),
        )


class TcspcRawReplayAnalyzer:
    """Offline analyzer for one captured GPX2 EXT raw event stream.

    The replay path deliberately computes several views from the same decoded
    event table so stream-order faults, timestamp formula faults, and real
    channel retriggers can be separated without relying on the GUI state.
    """

    @staticmethod
    def timestamp_regressions(timestamps: np.ndarray) -> int:
        values = np.asarray(timestamps, dtype=np.uint64)
        if values.size <= 1:
            return 0
        return int((values[1:] < values[:-1]).sum())

    @staticmethod
    def repeat_histogram(event_timestamps: np.ndarray, settings: TdcTestSettings) -> np.ndarray:
        hist = np.zeros(int(settings.bin_count), dtype=np.uint32)
        values = np.asarray(event_timestamps, dtype=np.int64)
        if values.size < 2:
            return hist
        if values.size > 1 and np.any(values[1:] < values[:-1]):
            values = np.sort(values, kind="mergesort")
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return hist
        bin_width = max(1, int(settings.bin_width_raw))
        bin_indices = diffs // bin_width
        valid = (bin_indices > 0) & (bin_indices < hist.size)
        if np.any(valid):
            counts = np.bincount(bin_indices[valid].astype(np.int64, copy=False), minlength=hist.size)
            nonzero = np.nonzero(counts)[0]
            hist[nonzero] = counts[nonzero].astype(hist.dtype, copy=False)
        return hist

    @staticmethod
    def _filter_deadtime_sorted(
        timestamps: np.ndarray,
        deadtime_raw: int,
    ) -> tuple[np.ndarray, int]:
        values = np.asarray(timestamps, dtype=np.int64)
        if values.size <= 1:
            return values.copy(), 0
        if np.any(values[1:] < values[:-1]):
            values = np.sort(values, kind="mergesort")
        if deadtime_raw <= 0:
            return values.copy(), 0
        keep = np.empty(values.size, dtype=bool)
        keep.fill(False)
        last_kept: Optional[int] = None
        for idx, value in enumerate(values):
            current = int(value)
            if last_kept is None or current - last_kept >= deadtime_raw:
                keep[idx] = True
                last_kept = current
        kept = values[keep]
        return kept, int(values.size - kept.size)

    @staticmethod
    def _shape_to_dict(shape: TcspcPeakShape) -> Dict[str, Any]:
        return {
            "peak_bin": int(shape.peak_bin),
            "peak_count": int(shape.peak_count),
            "peak_ns": float(shape.peak_ns),
            "fwhm_raw": float(shape.fwhm_raw),
            "fwhm_ns": float(shape.fwhm_ns),
            "rms_ns": float(shape.rms_ns),
            "background_mean": float(shape.background_mean),
            "background_std": float(shape.background_std),
            "peak_to_bg": float(shape.peak_to_bg),
            "second_peak_bin": int(shape.second_peak_bin),
            "peak_separation_ns": float(shape.peak_separation_ns),
            "half_count": float(shape.half_count),
            "fwhm_left_ns": float(shape.fwhm_left_ns),
            "fwhm_right_ns": float(shape.fwhm_right_ns),
        }

    @staticmethod
    def _commercial_to_dict(metrics: TcspcCommercialMetrics) -> Dict[str, Any]:
        return {
            "shape": TcspcRawReplayAnalyzer._shape_to_dict(metrics.shape),
            "background_mean": float(metrics.background_mean),
            "background_std": float(metrics.background_std),
            "background_cv": float(metrics.background_cv),
            "peak_to_bg": float(metrics.peak_to_bg),
            "deadtime_start_ns": float(metrics.deadtime_start_ns),
            "deadtime_end_ns": float(metrics.deadtime_end_ns),
            "deadtime_violation_count": int(metrics.deadtime_violation_count),
            "deadtime_violation_ratio": float(metrics.deadtime_violation_ratio),
            "satellite_decision": metrics.satellite_decision,
            "satellite_peaks": [
                {
                    "bin_index": int(item.bin_index),
                    "dt_ns": float(item.dt_ns),
                    "offset_ns": float(item.offset_ns),
                    "count": int(item.count),
                    "start_repeat_hits": int(item.start_repeat_hits),
                    "stop_repeat_hits": int(item.stop_repeat_hits),
                }
                for item in metrics.satellite_peaks
            ],
        }

    @staticmethod
    def replay_histogram_to_dict(item: Optional[TcspcReplayHistogram]) -> Dict[str, Any]:
        if item is None:
            return {}
        stop_count = max(1, int(item.stop_count))
        return {
            "name": item.name,
            "pairs": int(item.pairs),
            "paired_photons": int(item.paired_photons),
            "orphan_photons": int(item.orphan_photons),
            "before_history": int(item.before_history),
            "after_history": int(item.after_history),
            "sync_count": int(item.sync_count),
            "stop_count": int(item.stop_count),
            "pairs_per_photon": float(item.pairs) / stop_count,
            "no_pair_ratio": float(item.orphan_photons) / stop_count,
            "reference_filtered": int(item.reference_filtered),
            "deadtime_filtered": int(item.deadtime_filtered),
            "shape": TcspcRawReplayAnalyzer._shape_to_dict(item.shape),
            "commercial": TcspcRawReplayAnalyzer._commercial_to_dict(item.commercial),
        }

    @staticmethod
    def shape_diagnostic_to_dict(item: TcspcShapeDiagnostic) -> Dict[str, Any]:
        return {
            "name": item.name,
            "time_formula": item.time_formula,
            "pairing_mode": item.pairing_mode,
            "accepted_pairs": int(item.accepted_pairs),
            "paired_photons": int(item.paired_photons),
            "orphan_photons": int(item.orphan_photons),
            "pairs_per_photon": float(item.pairs_per_photon),
            "shape": TcspcRawReplayAnalyzer._shape_to_dict(item.shape),
            "coarse_ref_delta": {str(k): int(v) for k, v in item.coarse_ref_delta.items()},
            "fine_tstop_delta": {str(k): int(v) for k, v in item.fine_tstop_delta.items()},
        }

    @classmethod
    def _build_histogram(
        cls,
        name: str,
        channels: np.ndarray,
        extended_refs: np.ndarray,
        tstops: np.ndarray,
        settings: TdcTestSettings,
        *,
        formula: str,
        pairing_mode: str,
        reference_deadtime_ns: float,
        tdc_deadtime_ps: float,
        start_repeat_histogram: np.ndarray,
        stop_repeat_histogram: np.ndarray,
    ) -> TcspcReplayHistogram:
        channels_i = np.asarray(channels, dtype=np.int64)
        refs_i = np.asarray(extended_refs, dtype=np.int64)
        tstops_i = np.asarray(tstops, dtype=np.int64)
        timestamps = TcspcTimeReconstructor.timestamps_from_formula(
            refs_i,
            tstops_i,
            max(1, int(settings.refclk_divisions)),
            formula,
        )
        start_mask = channels_i == int(settings.start_channel)
        stop_mask = channels_i == int(settings.stop_channel)
        sync_ts = np.sort(timestamps[start_mask], kind="mergesort")
        stop_ts = np.sort(timestamps[stop_mask], kind="mergesort")
        raw_sync_count = int(sync_ts.size)
        raw_stop_count = int(stop_ts.size)

        tdc_deadtime_raw = int(
            round(max(0.0, float(tdc_deadtime_ps)) / (TcspcTimeReconstructor.TICK_NS * 1000.0))
        )
        sync_ts, sync_deadtime_filtered = cls._filter_deadtime_sorted(sync_ts, tdc_deadtime_raw)
        stop_ts, stop_deadtime_filtered = cls._filter_deadtime_sorted(stop_ts, tdc_deadtime_raw)
        reference_deadtime_raw = int(
            round(max(0.0, float(reference_deadtime_ns)) / TcspcTimeReconstructor.TICK_NS)
        )
        sync_ts, reference_filtered = cls._filter_deadtime_sorted(sync_ts, reference_deadtime_raw)

        min_dt_raw, max_dt_raw = TdcTestHistogramBuilder.window_bounds_for_settings(settings)
        pairs = TcspcTimeReconstructor.pair_indices(
            sync_ts,
            stop_ts,
            min_dt_raw,
            max_dt_raw,
            pairing_mode,
        )
        histogram = TcspcTimeReconstructor.histogram_from_dt(pairs.dt_raw, settings)
        expected_period = TcspcTimeReconstructor.estimate_sync_period_raw(sync_ts)
        commercial = TcspcCommercialAnalyzer.analyze(
            histogram,
            settings,
            start_repeat_histogram=start_repeat_histogram,
            stop_repeat_histogram=stop_repeat_histogram,
            expected_peak_separation_raw=expected_period,
            detector_deadtime_ns=getattr(settings, "detector_deadtime_ns", 100.0),
        )
        return TcspcReplayHistogram(
            name=name,
            histogram=histogram,
            pairs=int(pairs.dt_raw.size),
            paired_photons=int(pairs.paired_photons),
            orphan_photons=int(pairs.orphan_photons),
            before_history=int(pairs.before_history),
            after_history=int(pairs.after_history),
            sync_count=raw_sync_count,
            stop_count=raw_stop_count,
            reference_filtered=int(reference_filtered),
            deadtime_filtered=int(sync_deadtime_filtered + stop_deadtime_filtered),
            shape=commercial.shape,
            commercial=commercial,
        )

    @classmethod
    def _stream_histogram_result(
        cls,
        histogram: Optional[np.ndarray],
        settings: TdcTestSettings,
        counts: Optional[list[int] | tuple[int, ...] | np.ndarray],
        start_repeat_histogram: np.ndarray,
        stop_repeat_histogram: np.ndarray,
    ) -> Optional[TcspcReplayHistogram]:
        if histogram is None:
            return None
        values = np.asarray(histogram, dtype=np.uint32)
        image = list(counts) if counts is not None else []
        expected_period = int(image[22]) if len(image) > 22 else None
        commercial = TcspcCommercialAnalyzer.analyze(
            values,
            settings,
            start_repeat_histogram=start_repeat_histogram,
            stop_repeat_histogram=stop_repeat_histogram,
            expected_peak_separation_raw=expected_period,
            detector_deadtime_ns=getattr(settings, "detector_deadtime_ns", 100.0),
        )
        return TcspcReplayHistogram(
            name="stream_prev_start",
            histogram=values.copy(),
            pairs=int(image[0]) if len(image) > 0 else int(values.sum()),
            paired_photons=int(image[14]) if len(image) > 14 else 0,
            orphan_photons=int(image[13]) if len(image) > 13 else 0,
            before_history=int(image[18]) if len(image) > 18 else 0,
            after_history=int(image[19]) if len(image) > 19 else 0,
            sync_count=int(image[1]) if len(image) > 1 else 0,
            stop_count=int(image[2]) if len(image) > 2 else 0,
            reference_filtered=int(image[26]) if len(image) > 26 else 0,
            deadtime_filtered=int(image[23]) if len(image) > 23 else 0,
            shape=commercial.shape,
            commercial=commercial,
        )

    @classmethod
    def _compare_stream_sorted(
        cls,
        stream: Optional[TcspcReplayHistogram],
        sorted_prev: TcspcReplayHistogram,
    ) -> Dict[str, Any]:
        stream_dict = cls.replay_histogram_to_dict(stream)
        sorted_dict = cls.replay_histogram_to_dict(sorted_prev)
        if stream is None:
            return {
                "stream_available": False,
                "sorted": sorted_dict,
                "sorted_peak_to_bg_gain": 0.0,
                "sorted_pair_gain": 0.0,
                "sorted_no_pair_ratio_gain": 0.0,
            }
        stream_pbg = max(0.0, float(stream.commercial.peak_to_bg))
        sorted_pbg = max(0.0, float(sorted_prev.commercial.peak_to_bg))
        stream_pairs = max(1, int(stream.pairs))
        stream_no_pair = float(stream.orphan_photons) / max(1, int(stream.stop_count))
        sorted_no_pair = float(sorted_prev.orphan_photons) / max(1, int(sorted_prev.stop_count))
        return {
            "stream_available": True,
            "stream": stream_dict,
            "sorted": sorted_dict,
            "sorted_peak_to_bg_gain": float(sorted_pbg / max(stream_pbg, 1e-9)),
            "sorted_pair_gain": float(int(sorted_prev.pairs) / stream_pairs),
            "sorted_no_pair_ratio_gain": float(stream_no_pair / max(sorted_no_pair, 1e-9)),
        }

    @staticmethod
    def _time_formula_fault(diagnostics: tuple[TcspcShapeDiagnostic, ...]) -> Optional[str]:
        by_name = {item.name: item for item in diagnostics}
        baseline = by_name.get("prev_start_forward") or by_name.get("all_start_forward")
        if baseline is None or baseline.shape.fwhm_raw <= 0:
            return None
        viable = [
            item
            for item in diagnostics
            if item.time_formula != "forward"
            and item.accepted_pairs > 0
            and item.shape.fwhm_raw > 0
            and 1200.0 <= item.shape.peak_separation_ns <= 2000.0
        ]
        best = min(viable, key=lambda item: (item.shape.fwhm_raw, -item.shape.peak_to_bg), default=None)
        if best is not None and best.shape.fwhm_raw * 5 <= baseline.shape.fwhm_raw:
            return best.time_formula
        return None

    @classmethod
    def _classify(
        cls,
        *,
        timestamp_regressions: int,
        stream: Optional[TcspcReplayHistogram],
        sorted_prev: TcspcReplayHistogram,
        sorted_deadtime: TcspcReplayHistogram,
        diagnostics: tuple[TcspcShapeDiagnostic, ...],
    ) -> tuple[str, tuple[str, ...]]:
        flags: list[str] = []
        compare = cls._compare_stream_sorted(stream, sorted_prev)
        if sorted_prev.commercial.background_mean < 2.0:
            flags.append("low_statistics_background")
        stream_order_fault = False
        if timestamp_regressions > 0 and stream is not None:
            stream_order_fault = (
                compare["sorted_peak_to_bg_gain"] >= 2.0
                or compare["sorted_pair_gain"] >= 2.0
                or compare["sorted_no_pair_ratio_gain"] >= 2.0
            )
        if stream_order_fault:
            flags.append("stream_order_fault")
        formula_fault = cls._time_formula_fault(diagnostics)
        if formula_fault:
            flags.append(f"time_formula_fault:{formula_fault}")
        decision = sorted_deadtime.commercial.satellite_decision or sorted_prev.commercial.satellite_decision
        if "start" in decision:
            flags.append("start_retrigger")
        elif "stop" in decision:
            flags.append("stop_retrigger_or_afterpulse")
        elif "satellites_present" in decision:
            flags.append("satellites_present_source_inconclusive")
        if not any(
            flag.startswith(("stream_order_fault", "time_formula_fault", "start_retrigger", "stop_retrigger"))
            for flag in flags
        ):
            flags.append("input_or_gpx2_measurement_fault")
        priority = (
            "stream_order_fault",
            "time_formula_fault",
            "start_retrigger",
            "stop_retrigger_or_afterpulse",
            "satellites_present_source_inconclusive",
            "input_or_gpx2_measurement_fault",
            "low_statistics_background",
        )
        for item in priority:
            for flag in flags:
                if flag.startswith(item):
                    return flag, tuple(flags)
        return flags[0], tuple(flags)

    @classmethod
    def analyze_events(
        cls,
        channels: np.ndarray,
        timestamps: np.ndarray,
        refids: np.ndarray,
        tstops: np.ndarray,
        settings: TdcTestSettings,
        *,
        stream_histogram: Optional[np.ndarray] = None,
        stream_counts: Optional[list[int] | tuple[int, ...] | np.ndarray] = None,
        stream_start_repeat_histogram: Optional[np.ndarray] = None,
        stream_stop_repeat_histogram: Optional[np.ndarray] = None,
        tdc_deadtime_ps: Optional[float] = None,
        reference_deadtime_ns: Optional[float] = None,
    ) -> TcspcRawReplayResult:
        channels_i = np.asarray(channels, dtype=np.int64)
        timestamps_u = np.asarray(timestamps, dtype=np.uint64)
        tstops_i = np.asarray(tstops, dtype=np.int64)
        if timestamps_u.size != channels_i.size or tstops_i.size != channels_i.size:
            raise ValueError("channels, timestamps, and tstops must have the same length")
        refclk_divisions = max(1, int(settings.refclk_divisions))
        extended_refs = (timestamps_u // np.uint64(refclk_divisions)).astype(np.int64, copy=False)
        start_mask = channels_i == int(settings.start_channel)
        stop_mask = channels_i == int(settings.stop_channel)
        timestamp_i = timestamps_u.astype(np.int64, copy=False)
        start_repeat = cls.repeat_histogram(timestamp_i[start_mask], settings)
        stop_repeat = cls.repeat_histogram(timestamp_i[stop_mask], settings)
        reference_holdoff = (
            float(settings.reference_deadtime_ns)
            if reference_deadtime_ns is None
            else float(reference_deadtime_ns)
        )
        deadtime_ps = float(settings.tdc_deadtime_ps if tdc_deadtime_ps is None else tdc_deadtime_ps)
        base_settings = TdcTestSettings(
            enabled=True,
            start_channel=int(settings.start_channel),
            stop_channel=int(settings.stop_channel),
            source=settings.source,
            pairing_mode="prev_start",
            time_formula="forward",
            refclk_divisions=int(settings.refclk_divisions),
            bin_width_raw=int(settings.bin_width_raw),
            bin_offset=int(settings.bin_offset),
            bin_count=int(settings.bin_count),
            acquisition_time_s=float(settings.acquisition_time_s),
            tdc_deadtime_ps=deadtime_ps,
            reference_deadtime_ns=reference_holdoff,
            detector_deadtime_ns=float(getattr(settings, "detector_deadtime_ns", 100.0)),
            raw_epoch=int(getattr(settings, "raw_epoch", 0) or 0),
        )
        stream_start_repeat = (
            np.asarray(stream_start_repeat_histogram, dtype=np.uint32)
            if stream_start_repeat_histogram is not None
            else start_repeat
        )
        stream_stop_repeat = (
            np.asarray(stream_stop_repeat_histogram, dtype=np.uint32)
            if stream_stop_repeat_histogram is not None
            else stop_repeat
        )
        stream = cls._stream_histogram_result(
            stream_histogram,
            base_settings,
            stream_counts,
            stream_start_repeat,
            stream_stop_repeat,
        )
        sorted_prev = cls._build_histogram(
            "sorted_prev_start",
            channels_i,
            extended_refs,
            tstops_i,
            base_settings,
            formula="forward",
            pairing_mode="prev_start",
            reference_deadtime_ns=reference_holdoff,
            tdc_deadtime_ps=0.0,
            start_repeat_histogram=start_repeat,
            stop_repeat_histogram=stop_repeat,
        )
        sorted_deadtime = cls._build_histogram(
            "sorted_prev_start_deadtime",
            channels_i,
            extended_refs,
            tstops_i,
            base_settings,
            formula="forward",
            pairing_mode="prev_start",
            reference_deadtime_ns=reference_holdoff,
            tdc_deadtime_ps=deadtime_ps,
            start_repeat_histogram=start_repeat,
            stop_repeat_histogram=stop_repeat,
        )
        all_start_settings = TdcTestSettings(
            enabled=True,
            start_channel=base_settings.start_channel,
            stop_channel=base_settings.stop_channel,
            source=base_settings.source,
            pairing_mode="all_start",
            time_formula=base_settings.time_formula,
            refclk_divisions=base_settings.refclk_divisions,
            bin_width_raw=base_settings.bin_width_raw,
            bin_offset=base_settings.bin_offset,
            bin_count=base_settings.bin_count,
            acquisition_time_s=base_settings.acquisition_time_s,
            tdc_deadtime_ps=0.0,
            reference_deadtime_ns=reference_holdoff,
            detector_deadtime_ns=base_settings.detector_deadtime_ns,
            raw_epoch=base_settings.raw_epoch,
        )
        sorted_all = cls._build_histogram(
            "sorted_all_start",
            channels_i,
            extended_refs,
            tstops_i,
            all_start_settings,
            formula="forward",
            pairing_mode="all_start",
            reference_deadtime_ns=reference_holdoff,
            tdc_deadtime_ps=0.0,
            start_repeat_histogram=start_repeat,
            stop_repeat_histogram=stop_repeat,
        )
        diagnostics = tuple(
            TcspcTimeReconstructor.shape_diagnostics(
                channels_i,
                extended_refs,
                tstops_i,
                base_settings,
            )
        )
        channel_counts_np = np.bincount(
            channels_i[(channels_i >= 0) & (channels_i < 4)],
            minlength=4,
        )
        regression_count = cls.timestamp_regressions(timestamps_u)
        stream_vs_sorted = cls._compare_stream_sorted(stream, sorted_prev)
        decision, flags = cls._classify(
            timestamp_regressions=regression_count,
            stream=stream,
            sorted_prev=sorted_prev,
            sorted_deadtime=sorted_deadtime,
            diagnostics=diagnostics,
        )
        return TcspcRawReplayResult(
            settings=base_settings,
            channel_counts=tuple(int(channel_counts_np[idx]) for idx in range(4)),
            timestamp_regressions=regression_count,
            stream_prev_start=stream,
            sorted_prev_start=sorted_prev,
            sorted_prev_start_deadtime=sorted_deadtime,
            sorted_all_start=sorted_all,
            start_repeat_histogram=start_repeat,
            stop_repeat_histogram=stop_repeat,
            shape_diagnostics=diagnostics,
            stream_vs_sorted=stream_vs_sorted,
            flags=flags,
            decision=decision,
        )

    @classmethod
    def result_to_summary(cls, result: TcspcRawReplayResult) -> Dict[str, Any]:
        return {
            "decision": result.decision,
            "flags": list(result.flags),
            "timestamp_regressions": int(result.timestamp_regressions),
            "channel_counts": {
                f"ch{idx + 1}": int(value)
                for idx, value in enumerate(result.channel_counts)
            },
            "settings": {
                "start_channel": int(result.settings.start_channel) + 1,
                "stop_channel": int(result.settings.stop_channel) + 1,
                "refclk_divisions": int(result.settings.refclk_divisions),
                "bin_width_raw": int(result.settings.bin_width_raw),
                "bin_width_ps": float(result.settings.bin_width_raw) * TcspcTimeReconstructor.TICK_NS * 1000.0,
                "bin_offset": int(result.settings.bin_offset),
                "bin_count": int(result.settings.bin_count),
                "window_ns": (
                    int(result.settings.bin_width_raw)
                    * int(result.settings.bin_count)
                    * TcspcTimeReconstructor.TICK_NS
                ),
                "reference_deadtime_ns": float(result.settings.reference_deadtime_ns),
                "tdc_deadtime_ps": float(result.settings.tdc_deadtime_ps),
                "detector_deadtime_ns": float(result.settings.detector_deadtime_ns),
            },
            "stream_vs_sorted": result.stream_vs_sorted,
            "histograms": {
                "stream_prev_start": cls.replay_histogram_to_dict(result.stream_prev_start),
                "sorted_prev_start": cls.replay_histogram_to_dict(result.sorted_prev_start),
                "sorted_prev_start_deadtime": cls.replay_histogram_to_dict(
                    result.sorted_prev_start_deadtime
                ),
                "sorted_all_start": cls.replay_histogram_to_dict(result.sorted_all_start),
            },
            "shape_diagnostics": [
                cls.shape_diagnostic_to_dict(item) for item in result.shape_diagnostics
            ],
        }


class TdcTestHistogramBuilder:
    REFID_WRAP = 1 << 16
    REFID_MASK = REFID_WRAP - 1
    REFID_WRAP_HALF = REFID_WRAP >> 1
    WRAP_NEG_THRESH = 0xC000
    WRAP_POS_THRESH = 0x4000
    MAX_SYNC_TIMESTAMPS = 1_000_000
    REORDER_HISTORY_REFID_WRAPS = 32

    def __init__(self, settings: TdcTestSettings) -> None:
        self.settings = settings
        self.histogram = np.zeros(settings.bin_count, dtype=np.uint32)
        self.start_repeat_histogram = np.zeros(settings.bin_count, dtype=np.uint32)
        self.stop_repeat_histogram = np.zeros(settings.bin_count, dtype=np.uint32)
        self._last_refid_by_channel: list[Optional[int]] = [None, None, None, None]
        self._refid_epoch_by_channel = [0, 0, 0, 0]
        self._last_timestamp_by_channel: list[Optional[int]] = [None, None, None, None]
        self._latest_extended_ref: Optional[int] = None
        self._sync_timestamps = np.array([], dtype=np.int64)
        self._sync_timestamp_by_refid = np.full(self.REFID_WRAP, -1, dtype=np.int64)
        self._sync_extended_ref_by_refid = np.full(self.REFID_WRAP, -1, dtype=np.int64)
        self.latest_sync_timestamp: Optional[int] = None
        self._latest_seen_timestamp: Optional[int] = None
        self._last_stream_timestamp: Optional[int] = None
        self.channel_counts = [0, 0, 0, 0]
        self.sync_count = 0
        self.photon_count = 0
        self.accepted_count = 0
        self.paired_photon_count = 0
        self.last_dt_raw: Optional[int] = None
        self.min_dt_raw: Optional[int] = None
        self.max_dt_raw: Optional[int] = None
        self.zero_dt_count = 0
        self.negative_dt_count = 0
        self.out_of_window_count = 0
        self.orphan_photon_count = 0
        self.timestamp_regression_count = 0
        self.start_buffer_trim_count = 0
        self.legacy_fold_risk_count = 0
        self.photon_before_sync_history_count = 0
        self.photon_after_sync_history_count = 0
        self.deadtime_filtered_count = 0
        self.reference_filtered_count = 0
        self.sync_period_raw: Optional[int] = None
        self.stale_batch_count = 0
        self._last_reference_kept_timestamp: Optional[int] = None

    def configure(self, settings: TdcTestSettings) -> None:
        self.settings = settings
        self.clear()

    def clear(self) -> None:
        self.histogram = np.zeros(self.settings.bin_count, dtype=np.uint32)
        self.start_repeat_histogram = np.zeros(self.settings.bin_count, dtype=np.uint32)
        self.stop_repeat_histogram = np.zeros(self.settings.bin_count, dtype=np.uint32)
        self._last_refid_by_channel = [None, None, None, None]
        self._refid_epoch_by_channel = [0, 0, 0, 0]
        self._last_timestamp_by_channel = [None, None, None, None]
        self._latest_extended_ref = None
        self._sync_timestamps = np.array([], dtype=np.int64)
        self._sync_timestamp_by_refid = np.full(self.REFID_WRAP, -1, dtype=np.int64)
        self._sync_extended_ref_by_refid = np.full(self.REFID_WRAP, -1, dtype=np.int64)
        self.latest_sync_timestamp = None
        self._latest_seen_timestamp = None
        self._last_stream_timestamp = None
        self.channel_counts = [0, 0, 0, 0]
        self.sync_count = 0
        self.photon_count = 0
        self.accepted_count = 0
        self.paired_photon_count = 0
        self.last_dt_raw = None
        self.min_dt_raw = None
        self.max_dt_raw = None
        self.zero_dt_count = 0
        self.negative_dt_count = 0
        self.out_of_window_count = 0
        self.orphan_photon_count = 0
        self.timestamp_regression_count = 0
        self.start_buffer_trim_count = 0
        self.legacy_fold_risk_count = 0
        self.photon_before_sync_history_count = 0
        self.photon_after_sync_history_count = 0
        self.deadtime_filtered_count = 0
        self.reference_filtered_count = 0
        self.sync_period_raw = None
        self.stale_batch_count = 0
        self._last_reference_kept_timestamp = None

    def timestamp_raw(self, event: TdcEvent) -> int:
        return int(event.refid) * max(1, int(self.settings.refclk_divisions)) + int(event.tstop)

    def _initial_epoch_for_refid(self, refid: int) -> int:
        if self._latest_extended_ref is None:
            return 0
        base_epoch = (int(self._latest_extended_ref) // self.REFID_WRAP) * self.REFID_WRAP
        candidates = (
            max(0, base_epoch - self.REFID_WRAP),
            base_epoch,
            base_epoch + self.REFID_WRAP,
        )
        return min(candidates, key=lambda epoch: abs((epoch + int(refid)) - int(self._latest_extended_ref)))

    def _extend_refids(self, channels: np.ndarray, refids: np.ndarray) -> np.ndarray:
        channels_i = np.asarray(channels, dtype=np.int64)
        refids_i = np.asarray(refids, dtype=np.int64)
        if refids_i.size == 0:
            return refids_i
        extended = np.empty_like(refids_i)
        for idx, refid_value in enumerate(refids_i):
            channel = int(channels_i[idx]) if idx < channels_i.size else -1
            refid = int(refid_value) & self.REFID_MASK
            if 0 <= channel < 4:
                last_refid = self._last_refid_by_channel[channel]
                if last_refid is None:
                    epoch = self._initial_epoch_for_refid(refid)
                else:
                    epoch = int(self._refid_epoch_by_channel[channel])
                    if last_refid > self.WRAP_NEG_THRESH and refid < self.WRAP_POS_THRESH:
                        epoch += self.REFID_WRAP
                self._last_refid_by_channel[channel] = refid
                self._refid_epoch_by_channel[channel] = epoch
            else:
                epoch = self._initial_epoch_for_refid(refid)
            extended_refid = epoch + refid
            extended[idx] = extended_refid
            self._latest_extended_ref = (
                extended_refid
                if self._latest_extended_ref is None
                else max(int(self._latest_extended_ref), int(extended_refid))
            )
        return extended

    @staticmethod
    def window_bounds_for_settings(settings: TdcTestSettings) -> tuple[int, int]:
        bin_width = max(1, int(settings.bin_width_raw))
        bin_count = max(1, int(settings.bin_count))
        min_dt = max(0, int(settings.bin_offset))
        max_dt = int(settings.bin_offset) + (bin_width * bin_count) - 1
        return min_dt, max(min_dt, max_dt)

    def _window_bounds_raw(self) -> tuple[int, int]:
        return self.window_bounds_for_settings(self.settings)

    def _tdc_deadtime_raw(self) -> int:
        deadtime_ps = max(0.0, float(getattr(self.settings, "tdc_deadtime_ps", 0.0)))
        ps_per_tick = TcspcTimeReconstructor.TICK_NS * 1000.0
        return int(round(deadtime_ps / ps_per_tick)) if ps_per_tick > 0.0 else 0

    def _reference_deadtime_raw(self) -> int:
        deadtime_ns = max(0.0, float(getattr(self.settings, "reference_deadtime_ns", 0.0)))
        return int(round(deadtime_ns / TcspcTimeReconstructor.TICK_NS))

    @staticmethod
    def _filter_channel_deadtime(
        indices: np.ndarray,
        timestamps: np.ndarray,
        deadtime_raw: int,
    ) -> tuple[np.ndarray, int]:
        selected = np.asarray(indices, dtype=np.int64)
        if deadtime_raw <= 0 or selected.size <= 1:
            return selected, 0
        all_timestamps = np.asarray(timestamps, dtype=np.int64)
        order = np.argsort(all_timestamps[selected], kind="mergesort")
        sorted_indices = selected[order]
        keep: list[int] = []
        last_kept: Optional[int] = None
        for idx in sorted_indices:
            current = int(all_timestamps[int(idx)])
            if last_kept is None or current - last_kept >= deadtime_raw:
                keep.append(int(idx))
                last_kept = current
        kept = np.asarray(keep, dtype=np.int64)
        return kept, int(selected.size - kept.size)

    def _filter_reference_deadtime(
        self,
        indices: np.ndarray,
        timestamps: np.ndarray,
        deadtime_raw: int,
    ) -> tuple[np.ndarray, int]:
        selected = np.asarray(indices, dtype=np.int64)
        if deadtime_raw <= 0 or selected.size == 0:
            if selected.size:
                ts = np.asarray(timestamps, dtype=np.int64)
                self._last_reference_kept_timestamp = int(ts[selected[-1]])
            return selected, 0
        all_timestamps = np.asarray(timestamps, dtype=np.int64)
        order = np.argsort(all_timestamps[selected], kind="mergesort")
        sorted_indices = selected[order]
        keep: list[int] = []
        last_kept = self._last_reference_kept_timestamp
        for idx in sorted_indices:
            current = int(all_timestamps[int(idx)])
            if last_kept is None or current - int(last_kept) >= deadtime_raw:
                keep.append(int(idx))
                last_kept = current
        self._last_reference_kept_timestamp = last_kept
        kept = np.asarray(keep, dtype=np.int64)
        return kept, int(selected.size - kept.size)

    def _accumulate_repeat_intervals(
        self,
        repeat_histogram: np.ndarray,
        channel: int,
        timestamps: np.ndarray,
    ) -> None:
        values = np.asarray(timestamps, dtype=np.int64)
        if values.size == 0:
            return
        if values.size > 1 and np.any(values[1:] < values[:-1]):
            values = np.sort(values, kind="mergesort")
        channel_idx = int(channel)
        if 0 <= channel_idx < len(self._last_timestamp_by_channel):
            last_timestamp = self._last_timestamp_by_channel[channel_idx]
            if last_timestamp is not None:
                values_for_diff = np.concatenate((np.asarray([int(last_timestamp)], dtype=np.int64), values))
            else:
                values_for_diff = values
            self._last_timestamp_by_channel[channel_idx] = int(values[-1])
        else:
            values_for_diff = values
        if values_for_diff.size < 2:
            return
        diffs = np.diff(values_for_diff)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return
        bin_width = max(1, int(self.settings.bin_width_raw))
        bin_indices = diffs // bin_width
        valid = (bin_indices > 0) & (bin_indices < repeat_histogram.size)
        if not np.any(valid):
            return
        counts = np.bincount(bin_indices[valid].astype(np.int64, copy=False), minlength=repeat_histogram.size)
        nonzero_bins = np.nonzero(counts)[0]
        repeat_histogram[nonzero_bins] += counts[nonzero_bins].astype(repeat_histogram.dtype)

    def _sync_history_depth_raw(self) -> int:
        _min_dt, max_dt = self._window_bounds_raw()
        refclk_divisions = max(1, int(self.settings.refclk_divisions))
        reorder_slack = refclk_divisions * self.REFID_WRAP * self.REORDER_HISTORY_REFID_WRAPS
        return max_dt + max(max_dt, reorder_slack)

    def _remember_sync_timestamps(self, timestamps: np.ndarray, extended_refids: np.ndarray) -> None:
        sync_timestamps = np.asarray(timestamps, dtype=np.int64)
        if sync_timestamps.size == 0:
            return
        sync_extended_refids = np.asarray(extended_refids, dtype=np.int64)
        if sync_timestamps.size > 1 and np.any(sync_timestamps[1:] < sync_timestamps[:-1]):
            order = np.argsort(sync_timestamps)
            sync_timestamps = sync_timestamps[order]
            sync_extended_refids = sync_extended_refids[order]
        if self._sync_timestamps.size:
            combined = np.concatenate((self._sync_timestamps, sync_timestamps))
            if combined.size > 1 and np.any(combined[1:] < combined[:-1]):
                combined = np.sort(combined)
            self._sync_timestamps = combined
        else:
            self._sync_timestamps = sync_timestamps.copy()
        if self._sync_timestamps.size > self.MAX_SYNC_TIMESTAMPS:
            trimmed = int(self._sync_timestamps.size - self.MAX_SYNC_TIMESTAMPS)
            self._sync_timestamps = self._sync_timestamps[-self.MAX_SYNC_TIMESTAMPS :]
            self.start_buffer_trim_count += trimmed
        self.latest_sync_timestamp = int(self._sync_timestamps[-1])
        self.sync_period_raw = TcspcTimeReconstructor.estimate_sync_period_raw(self._sync_timestamps)
        sync_refs = (sync_extended_refids & self.REFID_MASK).astype(np.int64, copy=False)
        self._sync_timestamp_by_refid[sync_refs] = sync_timestamps
        self._sync_extended_ref_by_refid[sync_refs] = sync_extended_refids

    def _trim_sync_history(self) -> None:
        if self._sync_timestamps.size == 0 or self._latest_seen_timestamp is None:
            return
        keep_after = int(self._latest_seen_timestamp) - self._sync_history_depth_raw()
        trim_count = int(np.searchsorted(self._sync_timestamps, keep_after, side="left"))
        if trim_count <= 0:
            return
        self._sync_timestamps = self._sync_timestamps[trim_count:]
        self.start_buffer_trim_count += trim_count
        self.latest_sync_timestamp = (
            int(self._sync_timestamps[-1]) if self._sync_timestamps.size else None
        )
        self.sync_period_raw = TcspcTimeReconstructor.estimate_sync_period_raw(self._sync_timestamps)

    def bin_index(self, dt_raw: int) -> int | None:
        idx = (int(dt_raw) - self.settings.bin_offset) // max(1, self.settings.bin_width_raw)
        if idx < 0 or idx >= self.settings.bin_count:
            return None
        return int(idx)

    def process_event(self, event: TdcEvent) -> bool:
        return self.process_events([event])

    def process_batch(self, batch: TdcRawEventBatch) -> bool:
        if not self.settings.enabled:
            return False
        if len(batch) == 0:
            return False
        active_epoch = int(getattr(self.settings, "raw_epoch", 0) or 0)
        batch_epoch = int(getattr(batch, "raw_epoch", 0) or 0)
        if active_epoch and batch_epoch != active_epoch:
            self.stale_batch_count += 1
            return True
        before_counts = (
            self.sync_count,
            self.photon_count,
            self.accepted_count,
            self.orphan_photon_count,
            self.negative_dt_count,
            self.out_of_window_count,
            self.timestamp_regression_count,
            self.start_buffer_trim_count,
            self.legacy_fold_risk_count,
            self.deadtime_filtered_count,
            self.reference_filtered_count,
        )
        channels = np.asarray(batch.channels, dtype=np.int64)
        counts = np.bincount(channels[(channels >= 0) & (channels < 4)], minlength=4)
        for idx in range(4):
            self.channel_counts[idx] += int(counts[idx])

        refclk_divisions = max(1, int(self.settings.refclk_divisions))
        batch_timestamps = getattr(batch, "timestamps", None)
        time_formula = getattr(self.settings, "time_formula", "forward")
        if batch_timestamps is not None and np.asarray(batch_timestamps).size == len(batch):
            forward_timestamps = np.asarray(batch_timestamps, dtype=np.uint64).astype(np.int64, copy=False)
            extended_refids = forward_timestamps // refclk_divisions
            if time_formula == "forward":
                timestamps = forward_timestamps
            else:
                timestamps = TcspcTimeReconstructor.timestamps_from_formula(
                    extended_refids,
                    batch.tstops,
                    refclk_divisions,
                    time_formula,
                )
        else:
            extended_refids = self._extend_refids(channels, batch.refids)
            timestamps = TcspcTimeReconstructor.timestamps_from_formula(
                extended_refids,
                batch.tstops,
                refclk_divisions,
                time_formula,
            )
        if timestamps.size > 1:
            self.timestamp_regression_count += int((timestamps[1:] < timestamps[:-1]).sum())
        if timestamps.size and self._last_stream_timestamp is not None:
            self.timestamp_regression_count += int(timestamps[0] < int(self._last_stream_timestamp))
        if timestamps.size:
            self._last_stream_timestamp = int(timestamps[-1])
            batch_latest = int(timestamps.max())
            self._latest_seen_timestamp = (
                batch_latest
                if self._latest_seen_timestamp is None
                else max(int(self._latest_seen_timestamp), batch_latest)
            )

        sync_mask = channels == int(self.settings.start_channel)
        photon_mask = channels == int(self.settings.stop_channel)
        sync_indices = np.flatnonzero(sync_mask)
        photon_indices = np.flatnonzero(photon_mask)
        deadtime_raw = self._tdc_deadtime_raw()
        sync_indices, sync_filtered = self._filter_channel_deadtime(sync_indices, timestamps, deadtime_raw)
        photon_indices, photon_filtered = self._filter_channel_deadtime(
            photon_indices,
            timestamps,
            deadtime_raw,
        )
        self.deadtime_filtered_count += int(sync_filtered + photon_filtered)
        self.sync_count += int(sync_indices.size)
        self.photon_count += int(photon_indices.size)
        if sync_indices.size:
            self._accumulate_repeat_intervals(
                self.start_repeat_histogram,
                int(self.settings.start_channel),
                timestamps[sync_indices],
            )
        if photon_indices.size:
            self._accumulate_repeat_intervals(
                self.stop_repeat_histogram,
                int(self.settings.stop_channel),
                timestamps[photon_indices],
            )
        reference_deadtime_raw = self._reference_deadtime_raw()
        sync_pair_indices, reference_filtered = self._filter_reference_deadtime(
            sync_indices,
            timestamps,
            reference_deadtime_raw,
        )
        self.reference_filtered_count += int(reference_filtered)
        if sync_indices.size:
            self._remember_sync_timestamps(
                timestamps[sync_pair_indices],
                extended_refids[sync_pair_indices],
            )

        if photon_indices.size:
            photon_timestamps = timestamps[photon_indices]
            photon_extended_refs = extended_refids[photon_indices]
            photon_refs = (photon_extended_refs & self.REFID_MASK).astype(np.int64, copy=False)
            same_ref_sync = self._sync_timestamp_by_refid[photon_refs]
            same_ref_ext = self._sync_extended_ref_by_refid[photon_refs]
            same_ref_usable = same_ref_sync >= 0
            if np.any(same_ref_usable):
                same_ref_delta = photon_timestamps[same_ref_usable] - same_ref_sync[same_ref_usable]
                same_ref_ext_delta = photon_extended_refs[same_ref_usable] - same_ref_ext[same_ref_usable]
                legacy_fold_risk = (
                    (same_ref_delta < 0)
                    | (same_ref_delta >= refclk_divisions)
                    | (same_ref_ext_delta != 0)
                )
                self.legacy_fold_risk_count += int(legacy_fold_risk.sum())

            if self._sync_timestamps.size == 0:
                self.orphan_photon_count += int(photon_timestamps.size)
            else:
                min_dt_raw, max_dt_raw = self._window_bounds_raw()
                pairs = TcspcTimeReconstructor.pair_indices(
                    self._sync_timestamps,
                    photon_timestamps,
                    min_dt_raw,
                    max_dt_raw,
                    getattr(self.settings, "pairing_mode", "prev_start"),
                )
                self.orphan_photon_count += int(pairs.orphan_photons)
                self.photon_before_sync_history_count += int(pairs.before_history)
                self.photon_after_sync_history_count += int(pairs.after_history)
                accepted = self._accumulate_dt_values(pairs.dt_raw)
                if accepted:
                    self.paired_photon_count += int(pairs.paired_photons)
        self._trim_sync_history()
        after_counts = (
            self.sync_count,
            self.photon_count,
            self.accepted_count,
            self.orphan_photon_count,
            self.negative_dt_count,
            self.out_of_window_count,
            self.timestamp_regression_count,
            self.start_buffer_trim_count,
            self.legacy_fold_risk_count,
            self.deadtime_filtered_count,
            self.reference_filtered_count,
        )
        return after_counts != before_counts

    def _accumulate_dt_values(self, dt_raw: np.ndarray) -> int:
        if dt_raw.size == 0:
            return 0
        dt_raw = np.asarray(dt_raw, dtype=np.int64)
        negative = dt_raw < 0
        self.negative_dt_count += int(negative.sum())
        nonnegative_dt = dt_raw[~negative]
        if nonnegative_dt.size == 0:
            return 0
        bin_indices = (nonnegative_dt - int(self.settings.bin_offset)) // max(
            1, int(self.settings.bin_width_raw)
        )
        valid = (bin_indices >= 0) & (bin_indices < int(self.settings.bin_count))
        self.out_of_window_count += int((~valid).sum())
        accepted_dt = nonnegative_dt[valid]
        if accepted_dt.size == 0:
            return 0
        accepted_bins = bin_indices[valid].astype(np.int64, copy=False)
        bin_counts = np.bincount(accepted_bins)
        nonzero_bins = np.nonzero(bin_counts)[0]
        self.histogram[nonzero_bins] += bin_counts[nonzero_bins].astype(self.histogram.dtype)
        accepted_count = int(accepted_dt.size)
        self.accepted_count += accepted_count
        self.last_dt_raw = int(accepted_dt[-1])
        current_min = int(accepted_dt.min())
        current_max = int(accepted_dt.max())
        self.min_dt_raw = current_min if self.min_dt_raw is None else min(self.min_dt_raw, current_min)
        self.max_dt_raw = current_max if self.max_dt_raw is None else max(self.max_dt_raw, current_max)
        self.zero_dt_count += int((accepted_dt == 0).sum())
        return accepted_count

    def process_events(self, events: list[TdcEvent]) -> bool:
        if not events:
            return False
        header = events[0].header
        return self.process_batch(
            TdcRawEventBatch(
                header=header,
                channels=np.asarray([event.channel for event in events], dtype=np.uint8),
                refids=np.asarray([event.refid for event in events], dtype=np.uint32),
                tstops=np.asarray([event.tstop for event in events], dtype=np.uint32),
                rec_types=np.asarray([event.rec_type for event in events], dtype=np.uint8),
                event_classes=np.asarray([event.event_class for event in events], dtype=np.uint8),
                reserved=np.asarray([event.reserved for event in events], dtype=np.uint32),
            )
        )

    def snapshot(self) -> HistogramSnapshot:
        image = [[
            int(self.accepted_count),
            int(self.sync_count),
            int(self.photon_count),
            int(self.last_dt_raw if self.last_dt_raw is not None else -1),
            int(self.min_dt_raw if self.min_dt_raw is not None else -1),
            int(self.max_dt_raw if self.max_dt_raw is not None else -1),
            int(self.zero_dt_count),
            int(self.negative_dt_count),
            int(self.out_of_window_count),
            int(self.channel_counts[0]),
            int(self.channel_counts[1]),
            int(self.channel_counts[2]),
            int(self.channel_counts[3]),
            int(self.orphan_photon_count),
            int(self.paired_photon_count),
            int(self.timestamp_regression_count),
            int(self.start_buffer_trim_count),
            int(self.legacy_fold_risk_count),
            int(self.photon_before_sync_history_count),
            int(self.photon_after_sync_history_count),
            int(self._sync_timestamps.size),
            int(self._sync_history_depth_raw()),
            int(self.sync_period_raw if self.sync_period_raw is not None else 0),
            int(self.deadtime_filtered_count),
            int(self.stale_batch_count),
            int(getattr(self.settings, "raw_epoch", 0) or 0),
            int(self.reference_filtered_count),
        ]]
        return HistogramSnapshot(
            histograms={
                "tdc_test": self.histogram.copy(),
                "tdc_start_repeat": self.start_repeat_histogram.copy(),
                "tdc_stop_repeat": self.stop_repeat_histogram.copy(),
            },
            image_projection=image,
            current_row=0,
            current_col=0,
        )


class TdcTestAnalysisWorker(QtCore.QObject):
    snapshot_ready = QtCore.pyqtSignal(object)

    def __init__(self, settings: TdcTestSettings, snapshot_interval_s: float = 0.25, parent=None) -> None:
        super().__init__(parent)
        self.builder = TdcTestHistogramBuilder(settings)
        self.snapshot_interval_s = max(0.05, float(snapshot_interval_s))
        self._last_snapshot_s = 0.0

    @QtCore.pyqtSlot(object)
    def configure(self, settings: TdcTestSettings) -> None:
        self.builder.configure(settings)
        self._last_snapshot_s = time.monotonic()

    @QtCore.pyqtSlot()
    def clear(self) -> None:
        self.builder.clear()
        self._last_snapshot_s = time.monotonic()

    @QtCore.pyqtSlot()
    def emit_snapshot(self) -> None:
        self._last_snapshot_s = time.monotonic()
        self.snapshot_ready.emit(self.builder.snapshot())

    @QtCore.pyqtSlot(object)
    def process_batch(self, batch: TdcRawEventBatch) -> None:
        if not self.builder.settings.enabled:
            return
        if not self.builder.process_batch(batch):
            return
        now = time.monotonic()
        if (now - self._last_snapshot_s) >= self.snapshot_interval_s:
            self._last_snapshot_s = now
            self.snapshot_ready.emit(self.builder.snapshot())


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
                    for photon in packet.photon_events or []:
                        hist_builder.process_photon(photon)
                        state_machine.state.current_row = photon.line_id
                        state_machine.state.current_col = photon.pixel_id
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
    tdc_test_histogram_updated = QtCore.pyqtSignal(object)
    recording_state_changed = QtCore.pyqtSignal(bool, str)
    status_packet_updated = QtCore.pyqtSignal(object)
    _tdc_test_configure_requested = QtCore.pyqtSignal(object)
    _tdc_test_clear_requested = QtCore.pyqtSignal()
    _tdc_test_snapshot_requested = QtCore.pyqtSignal()

    def __init__(self, device_service: DeviceService, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.config = config
        self.raw_writer = RawDataWriter()
        self.scan_state = ScanStateMachine(config.marker_mapping)
        self.hist_builder = HistogramBuilder(config.histogram_settings)
        self._tdc_test_settings = config.tdc_test_settings
        self._last_tdc_test_snapshot = self._empty_tdc_test_snapshot(self._tdc_test_settings)
        self.recording = False
        self.session_dir: Optional[Path] = None
        self.session_metadata: Optional[SessionMetadata] = None
        self._histogram_snapshot_interval_s = 0.2
        self._last_histogram_snapshot_s = 0.0
        self._raw_bytes_connected = False
        self._fpga_hist_assemblies: dict[tuple[int, int], dict[str, Any]] = {}
        self._fpga_hist_latest_snapshot_id: dict[int, int] = {}
        self._shutdown = False

        self._tdc_test_thread = QtCore.QThread(self)
        self._tdc_test_worker = TdcTestAnalysisWorker(self._tdc_test_settings, snapshot_interval_s=0.25)
        self._tdc_test_worker.moveToThread(self._tdc_test_thread)
        self._tdc_test_thread.finished.connect(self._tdc_test_worker.deleteLater)
        self._tdc_test_configure_requested.connect(self._tdc_test_worker.configure)
        self._tdc_test_clear_requested.connect(self._tdc_test_worker.clear)
        self._tdc_test_snapshot_requested.connect(self._tdc_test_worker.emit_snapshot)
        self._tdc_test_worker.snapshot_ready.connect(self._handle_tdc_test_worker_snapshot)
        self.device_service.tdc_raw_batch_received.connect(self._tdc_test_worker.process_batch)
        self._tdc_test_thread.start()

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        _ACQUISITION_SERVICES.add(self)

        self.device_service.tdc_events_received.connect(self._handle_tdc_events)
        self.device_service.photon_events_received.connect(self._handle_photon_events)
        self.device_service.tdc_histogram_chunk_received.connect(self._handle_tdc_histogram_chunk)
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
        self._set_raw_recording_enabled(True)
        self.scan_state.reset()
        self.hist_builder.clear()
        self.recording_state_changed.emit(True, str(session_dir))
        return session_dir

    def stop_recording(self) -> None:
        self.recording = False
        self._set_raw_recording_enabled(False)
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
            self.session_metadata.photon_event_count = self.device_service.runtime_stats.photon_event_count
            SessionRepository.save_metadata(self.session_dir / "session.json", self.session_metadata)
        self.recording_state_changed.emit(False, "")

    def current_snapshot(self) -> HistogramSnapshot:
        return self.hist_builder.snapshot(
            current_row=self.scan_state.state.current_row,
            current_col=self.scan_state.state.current_col,
        )

    def current_tdc_test_snapshot(self) -> HistogramSnapshot:
        return self._last_tdc_test_snapshot

    def configure_tdc_test(self, settings: TdcTestSettings) -> HistogramSnapshot:
        self._tdc_test_settings = settings
        snapshot = self._empty_tdc_test_snapshot(settings)
        self._last_tdc_test_snapshot = snapshot
        self._fpga_hist_assemblies.clear()
        self._fpga_hist_latest_snapshot_id.clear()
        self._tdc_test_configure_requested.emit(settings)
        return snapshot

    def clear_tdc_test(self) -> HistogramSnapshot:
        snapshot = self._empty_tdc_test_snapshot(self._tdc_test_settings)
        self._last_tdc_test_snapshot = snapshot
        self._fpga_hist_assemblies.clear()
        self._fpga_hist_latest_snapshot_id.clear()
        self._tdc_test_clear_requested.emit()
        return snapshot

    def _handle_raw_bytes(self, data: bytes) -> None:
        if self.recording:
            self.raw_writer.write(data)

    def _set_raw_recording_enabled(self, enabled: bool) -> None:
        if enabled and not self._raw_bytes_connected:
            self.device_service.raw_bytes_received.connect(self._handle_raw_bytes)
            self._raw_bytes_connected = True
        elif not enabled and self._raw_bytes_connected:
            try:
                self.device_service.raw_bytes_received.disconnect(self._handle_raw_bytes)
            except (TypeError, RuntimeError):
                pass
            self._raw_bytes_connected = False
        self.device_service.set_raw_bytes_signal_enabled(enabled)

    @staticmethod
    def _empty_tdc_test_snapshot(settings: TdcTestSettings) -> HistogramSnapshot:
        return TdcTestHistogramBuilder(settings).snapshot()

    @QtCore.pyqtSlot(object)
    def _handle_tdc_test_worker_snapshot(self, snapshot: HistogramSnapshot) -> None:
        self._last_tdc_test_snapshot = snapshot
        self.tdc_test_histogram_updated.emit(snapshot)

    @QtCore.pyqtSlot(object)
    def _handle_tdc_histogram_chunk(self, chunk: TdcHistogramChunk) -> None:
        if self._tdc_test_settings.source != "gpx2_fpga":
            return
        expected_chunks = 0
        received_chunks = 0
        missing_chunks = 0
        partial_flag = 0
        last_chunk_index = int(chunk.bin_start) // 256 if int(chunk.bin_start) >= 0 else 0
        total_bins = max(0, int(chunk.total_bins))
        if total_bins <= 0:
            hist_id = int(chunk.hist_id)
            name = "tdc_test" if hist_id == 0 else "tdc_test_diagnostic"
            current = dict(self._last_tdc_test_snapshot.histograms)
            hist = np.zeros(0, dtype=np.uint32)
            current[name] = hist
            complete = True
        else:
            hist_id = int(chunk.hist_id)
            snapshot_id = int(chunk.snapshot_id)
            latest_snapshot_id = self._fpga_hist_latest_snapshot_id.get(hist_id, -1)
            if snapshot_id < latest_snapshot_id:
                return
            if snapshot_id > latest_snapshot_id:
                self._fpga_hist_latest_snapshot_id[hist_id] = snapshot_id
                for old_key in list(self._fpga_hist_assemblies.keys()):
                    if int(old_key[1]) == hist_id and int(old_key[0]) < snapshot_id:
                        self._fpga_hist_assemblies.pop(old_key, None)
            key = (snapshot_id, hist_id)
            assembly = self._fpga_hist_assemblies.get(key)
            if assembly is None or int(assembly["total_bins"]) != total_bins:
                assembly = {
                    "total_bins": total_bins,
                    "hist": np.zeros(total_bins, dtype=np.uint32),
                    "received": np.zeros(total_bins, dtype=np.bool_),
                    "last_chunk": chunk,
                }
                self._fpga_hist_assemblies[key] = assembly
            start = max(0, int(chunk.bin_start))
            stop = min(total_bins, start + int(chunk.bin_count))
            if stop > start:
                assembly["hist"][start:stop] = np.asarray(chunk.bins[: stop - start], dtype=np.uint32)
                assembly["received"][start:stop] = True
            assembly["last_chunk"] = chunk
            assembly["last_chunk_time_s"] = time.monotonic()
            name = "tdc_test" if int(chunk.hist_id) == 0 else "tdc_test_diagnostic"
            current = dict(self._last_tdc_test_snapshot.histograms)
            existing = current.get(name)
            if isinstance(existing, np.ndarray) and existing.shape == (total_bins,):
                live_hist = existing.astype(np.uint32, copy=True)
                if stop > start:
                    live_hist[start:stop] = np.asarray(chunk.bins[: stop - start], dtype=np.uint32)
            else:
                live_hist = np.asarray(assembly["hist"], dtype=np.uint32).copy()
            current[name] = live_hist
            complete = bool(np.all(assembly["received"]))
            expected_chunks = max(1, (total_bins + 255) // 256)
            received_mask = np.asarray(assembly["received"], dtype=np.bool_)
            received_chunks = sum(
                1
                for base in range(0, total_bins, 256)
                if bool(np.any(received_mask[base : min(total_bins, base + 256)]))
            )
            missing_chunks = max(0, expected_chunks - received_chunks)
            partial_flag = 0 if complete else 1
            hist = np.asarray(assembly["hist"], dtype=np.uint32).copy() if complete else live_hist

        if complete:
            current[name] = hist
        if int(chunk.hist_id) == 0:
            image = [[
                int(chunk.accepted_count),
                int(chunk.start_count),
                int(chunk.stop_count),
                int(chunk.last_dt_raw),
                0,
                0,
                0,
                0,
                int(chunk.out_of_window_count),
                0,
                0,
                0,
                0,
                int(chunk.no_start_count),
                int(chunk.accepted_count),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                int(chunk.tdc_deadtime_filtered_count),
                0,
                int(chunk.snapshot_id),
                int(chunk.reference_cleanup_count),
                int(chunk.status_flags),
                int(expected_chunks),
                int(received_chunks),
                int(missing_chunks),
                int(partial_flag),
                int(last_chunk_index),
            ]]
        else:
            image = self._last_tdc_test_snapshot.image_projection
        snapshot = HistogramSnapshot(
            histograms=current,
            image_projection=image,
            current_row=0,
            current_col=0,
        )
        self._last_tdc_test_snapshot = snapshot
        self.tdc_test_histogram_updated.emit(snapshot)

    def _handle_tdc_events(self, events: list[TdcEvent]) -> None:
        if self._tdc_test_settings.enabled:
            return
        histogram_changed = False
        for event in events:
            is_measurement, row, col = self.scan_state.process_event(event)
            if is_measurement:
                self.hist_builder.process_measurement(row, col, event)
                histogram_changed = True
        if histogram_changed:
            now = time.monotonic()
            if (now - self._last_histogram_snapshot_s) >= self._histogram_snapshot_interval_s:
                self._last_histogram_snapshot_s = now
                self.histogram_updated.emit(self.current_snapshot())

    def _handle_photon_events(self, events: list[PhotonEvent]) -> None:
        for event in events:
            self.scan_state.state.current_row = event.line_id
            self.scan_state.state.current_col = event.pixel_id
            self.hist_builder.process_photon(event)
        now = time.monotonic()
        if (now - self._last_histogram_snapshot_s) >= self._histogram_snapshot_interval_s:
            self._last_histogram_snapshot_s = now
            self.histogram_updated.emit(self.current_snapshot())

    def _handle_status(self, status: StatusPacket) -> None:
        self.status_packet_updated.emit(status)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._raw_bytes_connected:
            self._set_raw_recording_enabled(False)
        try:
            self.device_service.tdc_raw_batch_received.disconnect(self._tdc_test_worker.process_batch)
        except (TypeError, RuntimeError):
            pass
        if self._tdc_test_thread.isRunning():
            self._tdc_test_thread.quit()
            self._tdc_test_thread.wait(1000)
        try:
            _ACQUISITION_SERVICES.discard(self)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


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
