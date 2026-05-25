"""Probe GPX2 raw uplink state through FT601.

Run with the GUI closed:
    python -m app.tests.ft601_gpx2_raw_probe --duration 10 --start 2 --stop 3

This sends GPX2 default configuration, enables GPX2 raw upload, then reports
STATUS flags and any TDC_RAW events received. It is meant to split failures into:
command/config, GPX2/LCLK, raw-event reception, and FT601 uplink.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PyQt5 import QtCore

from app.data_processing import (
    TcspcCommercialAnalyzer,
    TcspcRawReplayAnalyzer,
    TdcTestHistogramBuilder,
    TcspcTimeReconstructor,
)
from app.models import TdcTestSettings
from app import protocol
from app.protocol import CommandEncoder, PacketParser
from app.usb_link import D3XXError, DeviceService


def main() -> int:
    parser = argparse.ArgumentParser(description="FT601 GPX2 raw uplink probe")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-size", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--start", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--stop", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument(
        "--live-parse",
        action="store_true",
        help="Parse while reading. Default captures USB blocks first to avoid creating FPGA backpressure.",
    )
    parser.add_argument(
        "--legacy-diagnostics",
        action="store_true",
        help="Also compute old same-ref modulo diagnostics. Slower; disabled by default for high-rate captures.",
    )
    parser.add_argument(
        "--shape-diag",
        action="store_true",
        help="Compare TCSPC time formulas and pairing modes on the same EXT raw capture.",
    )
    parser.add_argument(
        "--satellite-diag",
        action="store_true",
        help="Diagnose post-peak satellites: peak positions, pair coarse/fine deltas, and same-channel repeat intervals.",
    )
    parser.add_argument(
        "--ft-close",
        action="store_true",
        help="Call FT_Close before exit. By default this probe lets process exit release the handle.",
    )
    parser.add_argument(
        "--save-capture",
        action="store_true",
        help="Save the measured raw USB blocks to a .tdcpack file for GUI/probe replay comparison.",
    )
    parser.add_argument(
        "--capture-path",
        default="",
        help="Optional output path used with --save-capture.",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="Directory for the full diagnostic artifact package. Defaults to app/sessions/gpx2_diag_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--export-events",
        action="store_true",
        help="Export decoded EXT raw events to NPZ and selected CH2/CH3 CSV.GZ.",
    )
    parser.add_argument(
        "--export-histograms",
        action="store_true",
        help="Export stream/sorted histogram CSV files.",
    )
    parser.add_argument(
        "--sorted-replay",
        action="store_true",
        help="Rebuild histograms from globally timestamp-sorted EXT raw events.",
    )
    parser.add_argument(
        "--tdc-deadtime-ps",
        type=float,
        default=0.0,
        help="Optional same-channel TDC deadtime used only for the sorted deadtime replay histogram.",
    )
    parser.add_argument(
        "--bin-width-raw",
        type=int,
        default=5,
        help="Histogram bin width in GPX2 raw 8 ps ticks.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=50000,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--reference-deadtime-ns",
        type=float,
        default=800.0,
        help="Reference/start channel cleanup holdoff for commercial nearest-start histogram. Raw repeat diagnostics are still reported.",
    )
    args = parser.parse_args()
    artifact_requested = bool(
        args.export_dir
        or args.export_events
        or args.export_histograms
        or args.sorted_replay
        or args.save_capture
    )
    need_event_table = bool(artifact_requested or args.export_events or args.export_histograms or args.sorted_replay)
    write_raw_capture = bool(args.save_capture or artifact_requested)
    export_dir = (
        Path(args.export_dir)
        if args.export_dir
        else Path("app") / "sessions" / f"gpx2_diag_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if artifact_requested:
        export_dir.mkdir(parents=True, exist_ok=True)

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    encoder = CommandEncoder()

    legacy_tdc_events = 0
    raw_batches = 0
    raw_batch_events = 0
    raw_channel_counts = [0, 0, 0, 0]
    raw_blocks = 0
    status_packets = 0
    raw_payload_word_hist = Counter()
    raw_item_count_hist = Counter()
    raw_packet_byte_hist = Counter()
    raw_packet_kind_counts = Counter()
    packet_seq_gap_count = 0
    last_packet_seq: int | None = None
    last_packet_desc: str | None = None
    packet_seq_gap_samples: list[str] = []
    parser_reject_counts = Counter()
    parser_reject_samples: list[str] = []
    measurement_rx_bytes = 0
    status_at_start = None
    last_status = None
    sample_events: list[str] = []
    unselected_samples: list[str] = []
    regression_samples: list[str] = []
    ext_timestamp_regressions = 0
    ext_raw_events = 0
    last_ext_timestamp: int | None = None
    last_ext_event_text: str | None = None
    global_ext_event_index = 0
    sync_tstop_by_refid: dict[int, int] = {}
    exact_ref_dt_samples: list[int] = []
    legacy_modulo_hist = np.zeros(300000, dtype=np.uint32)
    legacy_modulo_pairs = 0
    shape_diag_channels: list[np.ndarray] = []
    shape_diag_ext_refs: list[np.ndarray] = []
    shape_diag_tstops: list[np.ndarray] = []
    event_raw_index_chunks: list[np.ndarray] = []
    event_packet_seq_chunks: list[np.ndarray] = []
    event_item_index_chunks: list[np.ndarray] = []
    event_channel_chunks: list[np.ndarray] = []
    event_timestamp_chunks: list[np.ndarray] = []
    event_refid_chunks: list[np.ndarray] = []
    event_tstop_chunks: list[np.ndarray] = []
    event_delta_prev_chunks: list[np.ndarray] = []
    event_regression_chunks: list[np.ndarray] = []
    measurement_active = False
    raw_capture_handle = None
    hist_builder = TdcTestHistogramBuilder(
        TdcTestSettings(
            enabled=True,
            start_channel=args.start - 1,
            stop_channel=args.stop - 1,
            source="gpx2",
            refclk_divisions=12500,
            bin_width_raw=max(1, int(args.bin_width_raw)),
            bin_offset=0,
            bin_count=max(1, int(args.bin_count)),
            acquisition_time_s=args.duration,
            tdc_deadtime_ps=float(args.tdc_deadtime_ps),
            reference_deadtime_ns=args.reference_deadtime_ns,
        )
    )

    def fmt_rate(bytes_per_second: float) -> str:
        if bytes_per_second >= 1_000_000.0:
            return f"{bytes_per_second / 1_000_000.0:.2f} MB/s"
        return f"{bytes_per_second / 1_000.0:.1f} kB/s"

    def u32_delta(current: int, baseline: int | None) -> str:
        if baseline is None:
            return "-"
        delta = int(current) - int(baseline)
        if delta < 0:
            delta += 1 << 32
        return str(delta)

    def fmt_ch_mask(mask: int) -> str:
        channels = [f"CH{idx + 1}" for idx in range(4) if mask & (1 << idx)]
        return ",".join(channels) if channels else "-"

    def decode_overflow_diag(value: int) -> dict[str, int]:
        return {
            "raw_full_mask": value & 0xF,
            "ext_full_mask": (value >> 4) & 0xF,
            "photon_full": (value >> 8) & 0x1,
            "raw_test_enable": (value >> 9) & 0x1,
            "raw_test_synthetic": (value >> 10) & 0x1,
            "raw_test_spi": (value >> 11) & 0x1,
            "ext_full_count_sat": (value >> 12) & 0x3FF,
            "photon_full_count_sat": (value >> 22) & 0x3FF,
        }

    def log(message: str) -> None:
        print(message, flush=True)

    def stop_transport() -> None:
        service.set_read_enabled(False)
        if service.tx_worker is not None:
            service.tx_worker.stop()
            service.tx_worker.wait(1000)
            service.tx_worker = None
        if args.ft_close:
            service.close_device()
        else:
            service.device = None
            service._status_event.clear()
            service.device_state_changed.emit(False, "Disconnected")
            service.log_message.emit("FT601 device close skipped; OS will release handle on process exit.")

    def read_once() -> bool:
        if service.device is None:
            return False
        try:
            block = service.device.read_block(args.read_size)
        except Exception as exc:
            if hasattr(service.device, "is_read_idle_exception") and service.device.is_read_idle_exception(exc):
                return False
            raise
        if not block:
            return False
        service._handle_raw_bytes(block)
        return True

    def pump_until(deadline: float, idle_sleep_s: float = 0.001) -> None:
        while time.monotonic() < deadline:
            got_data = read_once()
            app.processEvents(QtCore.QEventLoop.AllEvents, 1)
            if not got_data:
                time.sleep(idle_sleep_s)

    def read_blocks_until(deadline: float, idle_sleep_s: float = 0.0) -> list[bytes]:
        blocks: list[bytes] = []
        while time.monotonic() < deadline:
            if service.device is None:
                break
            try:
                block = service.device.read_block(args.read_size)
            except Exception as exc:
                if hasattr(service.device, "is_read_idle_exception") and service.device.is_read_idle_exception(exc):
                    if idle_sleep_s > 0.0:
                        time.sleep(idle_sleep_s)
                    continue
                raise
            if block:
                blocks.append(block)
            elif idle_sleep_s > 0.0:
                time.sleep(idle_sleep_s)
        return blocks

    def on_raw(_data: bytes) -> None:
        nonlocal raw_blocks, measurement_rx_bytes
        if not measurement_active:
            return
        raw_blocks += 1
        measurement_rx_bytes += len(_data)
        if raw_capture_handle is not None:
            raw_capture_handle.write(_data)

    def on_events(events) -> None:
        nonlocal legacy_tdc_events
        if not measurement_active:
            return
        legacy_tdc_events += len(events)

    def on_batch(batch, *, process_hist: bool = True, record_events: bool = True) -> None:
        nonlocal raw_batches, raw_batch_events, legacy_modulo_pairs
        nonlocal ext_timestamp_regressions, ext_raw_events, last_ext_timestamp
        nonlocal last_ext_event_text, global_ext_event_index
        if not measurement_active:
            return
        if process_hist:
            raw_batches += 1
            raw_batch_events += len(batch)
            hist_builder.process_batch(batch)
        if not record_events:
            return
        channels = np.asarray(batch.channels, dtype=np.int64)
        refids = np.asarray(batch.refids, dtype=np.int64)
        tstops = np.asarray(batch.tstops, dtype=np.int64)
        timestamps = getattr(batch, "timestamps", None)
        timestamp_values = (
            np.asarray(timestamps, dtype=np.uint64)
            if timestamps is not None and np.asarray(timestamps).size == len(batch)
            else None
        )
        if timestamp_values is not None and timestamp_values.size:
            ext_raw_events += int(timestamp_values.size)
            base_ext_index = int(global_ext_event_index)
            if need_event_table:
                item_indices = np.arange(timestamp_values.size, dtype=np.uint32)
                ts_i64 = timestamp_values.astype(np.int64, copy=False)
                delta_prev = np.empty(timestamp_values.size, dtype=np.int64)
                if last_ext_timestamp is None:
                    delta_prev[0] = 0
                else:
                    delta_prev[0] = int(ts_i64[0]) - int(last_ext_timestamp)
                if timestamp_values.size > 1:
                    delta_prev[1:] = np.diff(ts_i64)
                regression_flags = (delta_prev < 0).astype(np.uint8, copy=False)
                event_raw_index_chunks.append(
                    (base_ext_index + item_indices.astype(np.uint64, copy=False)).astype(np.uint64, copy=False)
                )
                event_packet_seq_chunks.append(
                    np.full(timestamp_values.size, int(batch.header.seq), dtype=np.uint32)
                )
                event_item_index_chunks.append(item_indices.copy())
                event_channel_chunks.append(channels.astype(np.uint8, copy=True))
                event_timestamp_chunks.append(timestamp_values.astype(np.uint64, copy=True))
                event_refid_chunks.append(refids.astype(np.uint32, copy=True))
                event_tstop_chunks.append(tstops.astype(np.uint32, copy=True))
                event_delta_prev_chunks.append(delta_prev.copy())
                event_regression_chunks.append(regression_flags)
            if args.shape_diag or args.satellite_diag or need_event_table:
                selected = (channels == (args.start - 1)) | (channels == (args.stop - 1))
                if np.any(selected):
                    shape_diag_channels.append(channels[selected].astype(np.uint8, copy=True))
                    shape_diag_ext_refs.append(
                        (timestamp_values[selected] // np.uint64(hist_builder.settings.refclk_divisions)).astype(
                            np.int64,
                            copy=True,
                        )
                    )
                    shape_diag_tstops.append(tstops[selected].astype(np.int64, copy=True))
            if timestamp_values.size > 1:
                regressions = np.flatnonzero(timestamp_values[1:] < timestamp_values[:-1])
                ext_timestamp_regressions += int(regressions.size)
                if len(regression_samples) < 12:
                    remaining = 12 - len(regression_samples)
                    for rel_idx in regressions[:remaining]:
                        prev_idx = int(rel_idx)
                        curr_idx = prev_idx + 1
                        regression_samples.append(
                            f"idx={global_ext_event_index + curr_idx}:"
                            f"prev=CH{int(channels[prev_idx]) + 1}:ref={int(refids[prev_idx])}:"
                            f"tstop={int(tstops[prev_idx])}:ts={int(timestamp_values[prev_idx])}->"
                            f"curr=CH{int(channels[curr_idx]) + 1}:ref={int(refids[curr_idx])}:"
                            f"tstop={int(tstops[curr_idx])}:ts={int(timestamp_values[curr_idx])}"
                        )
            if last_ext_timestamp is not None:
                if timestamp_values[0] < np.uint64(last_ext_timestamp):
                    ext_timestamp_regressions += 1
                    if len(regression_samples) < 12:
                        regression_samples.append(
                            f"idx={global_ext_event_index}:prev={last_ext_event_text}->"
                            f"curr=CH{int(channels[0]) + 1}:ref={int(refids[0])}:"
                            f"tstop={int(tstops[0])}:ts={int(timestamp_values[0])}"
                        )
            if len(unselected_samples) < 12:
                selected = {int(args.start) - 1, int(args.stop) - 1}
                bad_indices = np.flatnonzero(~np.isin(channels, list(selected)))
                remaining = 12 - len(unselected_samples)
                for idx in bad_indices[:remaining]:
                    unselected_samples.append(
                        f"idx={global_ext_event_index + int(idx)}:"
                        f"CH{int(channels[idx]) + 1}:ref={int(refids[idx])}:"
                        f"tstop={int(tstops[idx])}:ts={int(timestamp_values[idx])}"
                    )
            last_ext_timestamp = int(timestamp_values[-1])
            last_ext_event_text = (
                f"CH{int(channels[-1]) + 1}:ref={int(refids[-1])}:"
                f"tstop={int(tstops[-1])}:ts={int(timestamp_values[-1])}"
            )
            global_ext_event_index += int(timestamp_values.size)
        if len(sample_events) < 16:
            remaining = 16 - len(sample_events)
            for idx in range(min(remaining, len(batch))):
                ts_text = (
                    f":ts={int(timestamp_values[idx])}"
                    if timestamp_values is not None and idx < timestamp_values.size
                    else ""
                )
                sample_events.append(
                    f"CH{int(channels[idx]) + 1}:ref={int(refids[idx])}:tstop={int(tstops[idx])}{ts_text}"
                )
        if args.legacy_diagnostics:
            for idx in range(len(batch)):
                channel = int(channels[idx])
                refid = int(refids[idx])
                tstop = int(tstops[idx])
                if channel == args.start - 1:
                    sync_tstop_by_refid[refid] = tstop
                elif channel == args.stop - 1 and refid in sync_tstop_by_refid:
                    raw_dt = tstop - int(sync_tstop_by_refid[refid])
                    exact_ref_dt_samples.append(raw_dt)
                    folded_dt = raw_dt % max(1, hist_builder.settings.refclk_divisions)
                    bin_idx = (folded_dt - hist_builder.settings.bin_offset) // max(
                        1, hist_builder.settings.bin_width_raw
                    )
                    if 0 <= bin_idx < legacy_modulo_hist.size:
                        legacy_modulo_hist[int(bin_idx)] += 1
                        legacy_modulo_pairs += 1
        counts = np.bincount(channels[(channels >= 0) & (channels < 4)], minlength=4)
        for idx in range(4):
            raw_channel_counts[idx] += int(counts[idx])

    def on_status(status) -> None:
        nonlocal status_packets, last_status
        status_packets += 1
        last_status = status

    def on_packet(packet) -> None:
        if not measurement_active:
            return
        if packet.header.pkt_type == protocol.PKT_TDC_RAW:
            raw_packet_kind_counts["old"] += 1
        elif packet.header.pkt_type == protocol.PKT_TDC_RAW_EXT:
            raw_packet_kind_counts["ext"] += 1
        else:
            return
        raw_payload_word_hist[int(packet.header.payload_words)] += 1
        raw_item_count_hist[int(packet.header.item_count)] += 1
        raw_packet_byte_hist[int(len(packet.raw_bytes))] += 1

    def fmt_hist(hist: Counter) -> str:
        if not hist:
            return "-"
        return ",".join(f"{key}:{count}" for key, count in hist.most_common(6))

    def parse_blocks(blocks: list[bytes], *, measurement: bool) -> None:
        nonlocal measurement_active, raw_blocks, packet_seq_gap_count
        nonlocal last_packet_seq, last_packet_desc
        parser_state = PacketParser(decode_tdc_events=False)
        old_active = measurement_active
        measurement_active = bool(measurement)
        if measurement:
            raw_blocks += len(blocks)
        pending_batches = []
        pending_events = 0

        def flush_batches() -> None:
            nonlocal pending_batches, pending_events
            if not pending_batches:
                return
            on_batch(
                DeviceService._merge_tdc_batches(pending_batches),
                process_hist=True,
                record_events=not need_event_table,
            )
            pending_batches = []
            pending_events = 0

        for block in blocks:
            for packet in parser_state.feed(block):
                if measurement:
                    if last_packet_seq is not None:
                        expected_seq = (last_packet_seq + 1) & 0xFFFF
                        if int(packet.header.seq) != expected_seq:
                            packet_seq_gap_count += 1
                            if len(packet_seq_gap_samples) < 12:
                                packet_seq_gap_samples.append(
                                    f"prev={last_packet_desc} expected={expected_seq} "
                                    f"got=seq{int(packet.header.seq)}:"
                                    f"type=0x{int(packet.header.pkt_type):02X}:"
                                    f"words={int(packet.header.payload_words)}:"
                                    f"items={int(packet.header.item_count)}"
                                )
                    last_packet_seq = int(packet.header.seq)
                    last_packet_desc = (
                        f"seq{int(packet.header.seq)}:"
                        f"type=0x{int(packet.header.pkt_type):02X}:"
                        f"words={int(packet.header.payload_words)}:"
                        f"items={int(packet.header.item_count)}"
                    )
                if packet.status is not None:
                    on_status(packet.status)
                if measurement and packet.header.pkt_type in (protocol.PKT_TDC_RAW, protocol.PKT_TDC_RAW_EXT):
                    on_packet(packet)
                    if packet.tdc_event_batch is not None:
                        if need_event_table:
                            on_batch(packet.tdc_event_batch, process_hist=False, record_events=True)
                        pending_batches.append(packet.tdc_event_batch)
                        pending_events += len(packet.tdc_event_batch)
                        if pending_events >= 250_000:
                            flush_batches()
        flush_batches()
        if parser_state.reject_counts:
            parser_reject_counts.update(parser_state.reject_counts)
            for sample in parser_state.reject_samples:
                if len(parser_reject_samples) >= 16:
                    break
                parser_reject_samples.append(sample)
        measurement_active = old_active

    service.log_message.connect(log)
    service.set_raw_bytes_signal_enabled(True)
    service.set_emit_packet_objects(True)
    service.raw_bytes_received.connect(on_raw)
    service.packet_received.connect(on_packet)
    service.tdc_raw_batch_received.connect(on_batch)
    service.tdc_events_received.connect(on_events)
    service.status_received.connect(on_status)
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    channel_mask = (1 << (args.start - 1)) | (1 << (args.stop - 1))
    enable_frame = encoder.encode_tdc_test_upload(
        True,
        channel_mask,
        synthetic=False,
        extended_timestamp_raw=True,
    )
    enable_payload = int.from_bytes(enable_frame[4:8], byteorder="little", signed=False)
    disable_frame = encoder.encode_tdc_test_upload(False, 0)

    try:
        service.open_device(
            device_index=args.index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=args.read_size,
            read_enabled=False,
        )
    except D3XXError as exc:
        print(f"open failed: {exc}. Close the GUI/app using FT601 first.", flush=True)
        return 1

    try:
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="probe pre-disable any previous TDC upload",
        )
        service.send_command_sync(
            protocol.CMD_GPX2_CFG_CUSTOM,
            encoder.encode_gpx2_custom_config(protocol.GPX2_DEFAULT_CONFIG_BYTES),
            timeout_ms=1000,
            debug_details=(
                "probe apply max high-resolution GPX2 configuration "
                f"profile={protocol.GPX2_DEFAULT_PROFILE} sequence={protocol.GPX2_DEFAULT_SEQUENCE}"
            ),
        )

        last_status = None
        measurement_rx_bytes = 0
        if args.live_parse:
            pump_until(time.monotonic() + 1.2, idle_sleep_s=0.005)
        else:
            pre_blocks = read_blocks_until(time.monotonic() + 1.2, idle_sleep_s=0.001)
            parse_blocks(pre_blocks, measurement=False)
        status_at_start = last_status
        legacy_tdc_events = 0
        raw_batches = 0
        raw_batch_events = 0
        raw_channel_counts = [0, 0, 0, 0]
        raw_blocks = 0
        sample_events.clear()
        shape_diag_channels.clear()
        shape_diag_ext_refs.clear()
        shape_diag_tstops.clear()
        event_raw_index_chunks.clear()
        event_packet_seq_chunks.clear()
        event_item_index_chunks.clear()
        event_channel_chunks.clear()
        event_timestamp_chunks.clear()
        event_refid_chunks.clear()
        event_tstop_chunks.clear()
        event_delta_prev_chunks.clear()
        event_regression_chunks.clear()
        sync_tstop_by_refid.clear()
        exact_ref_dt_samples.clear()
        legacy_modulo_hist.fill(0)
        legacy_modulo_pairs = 0
        ext_timestamp_regressions = 0
        ext_raw_events = 0
        last_ext_timestamp = None
        last_ext_event_text = None
        global_ext_event_index = 0
        hist_builder.clear()
        capture_path = None
        if write_raw_capture:
            capture_path = (
                Path(args.capture_path)
                if args.capture_path
                else (export_dir / "raw_usb.tdcpack")
                if artifact_requested
                else Path("app") / "sessions" / f"gpx2_raw_capture_{time.strftime('%Y%m%d_%H%M%S')}.tdcpack"
            )
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            if args.live_parse:
                raw_capture_handle = open(capture_path, "wb")

        print(
            "enable_upload: "
            f"start=CH{args.start} stop=CH{args.stop} channel_mask=0x{channel_mask:X} "
            f"payload_word=0x{enable_payload:08X}",
            flush=True,
        )
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            enable_frame,
            timeout_ms=1000,
            debug_details=f"GPX2 raw probe start=CH{args.start} stop=CH{args.stop}",
        )

        deadline = time.monotonic() + max(0.1, float(args.duration))
        measurement_blocks: list[bytes] = []
        if args.live_parse:
            measurement_active = True
            pump_until(deadline)
            measurement_active = False
            if raw_capture_handle is not None:
                raw_capture_handle.close()
                raw_capture_handle = None
                print(f"saved_capture: {capture_path} bytes={measurement_rx_bytes}", flush=True)
        else:
            measurement_blocks = read_blocks_until(deadline, idle_sleep_s=0.0)
            measurement_rx_bytes = sum(len(block) for block in measurement_blocks)
            if write_raw_capture:
                with open(capture_path, "wb") as handle:
                    for block in measurement_blocks:
                        handle.write(block)
                print(f"saved_capture: {capture_path} bytes={measurement_rx_bytes}", flush=True)

        measurement_active = False
        log("disable_upload: stopping GPX2 raw upload")
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="GPX2 raw probe stop",
        )
        if args.live_parse:
            pump_until(time.monotonic() + 0.1, idle_sleep_s=0.005)
        else:
            post_blocks = read_blocks_until(time.monotonic() + 0.1, idle_sleep_s=0.001)
            parse_blocks(measurement_blocks, measurement=True)
            parse_blocks(post_blocks, measurement=False)
    finally:
        if raw_capture_handle is not None:
            raw_capture_handle.close()
            raw_capture_handle = None
        stop_transport()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    if last_status is not None:
        decoded = DeviceService.decode_status_flags(last_status.flags)
        start_raw = (
            (
                status_at_start.gpx2_raw_count_ch1,
                status_at_start.gpx2_raw_count_ch2,
                status_at_start.gpx2_raw_count_ch3,
                status_at_start.gpx2_raw_count_ch4,
            )
            if status_at_start is not None
            else (None, None, None, None)
        )
        last_raw = (
            last_status.gpx2_raw_count_ch1,
            last_status.gpx2_raw_count_ch2,
            last_status.gpx2_raw_count_ch3,
            last_status.gpx2_raw_count_ch4,
        )
        overflow_diag = int(last_status.tdc_drop_count)
        overflow_diag_start = int(status_at_start.tdc_drop_count) if status_at_start else None
        diag = decode_overflow_diag(overflow_diag)
        print(
            "last_status: "
            f"flags=0x{last_status.flags:04X} cfg_done={int(decoded.gpx2_cfg_done)} "
            f"cfg_error={int(decoded.gpx2_cfg_error)} lclk_locked={int(decoded.gpx2_lclk_locked)} "
            f"overflow={int(decoded.gpx2_event_overflow)} uptime={last_status.uptime_seconds}s "
            f"raw_ch={last_status.gpx2_raw_count_ch1}/"
            f"{last_status.gpx2_raw_count_ch2}/"
            f"{last_status.gpx2_raw_count_ch3}/"
            f"{last_status.gpx2_raw_count_ch4} "
            f"raw_delta={u32_delta(last_raw[0], start_raw[0])}/"
            f"{u32_delta(last_raw[1], start_raw[1])}/"
            f"{u32_delta(last_raw[2], start_raw[2])}/"
            f"{u32_delta(last_raw[3], start_raw[3])} "
            f"overflow_diag=0x{overflow_diag:08X} "
            f"overflow_diag_delta={u32_delta(overflow_diag, overflow_diag_start)} "
            f"raw_full_ch={fmt_ch_mask(diag['raw_full_mask'])} "
            f"ext_full_ch={fmt_ch_mask(diag['ext_full_mask'])} "
            f"photon_full={diag['photon_full']} "
            f"raw_test_seen={diag['raw_test_enable']} "
            f"synthetic_seen={diag['raw_test_synthetic']} "
            f"spi_seen={diag['raw_test_spi']} "
            f"ext_full_count_sat={diag['ext_full_count_sat']} "
            f"photon_full_count_sat={diag['photon_full_count_sat']} "
            f"usb_drop_delta={u32_delta(last_status.usb_drop_count, status_at_start.usb_drop_count if status_at_start else None)} "
            f"cfg_diag=0x{last_status.gpx2_cfg_diag:08X}",
            flush=True,
        )
    else:
        print("last_status: none", flush=True)

    snapshot = hist_builder.snapshot()
    hist = np.asarray(snapshot.histograms["tdc_test"], dtype=np.uint32)
    counts = snapshot.image_projection[0]
    peak_bin = int(hist.argmax()) if hist.size else -1
    peak_count = int(hist[peak_bin]) if peak_bin >= 0 else 0
    peak_ns = (peak_bin * hist_builder.settings.bin_width_raw + hist_builder.settings.bin_offset) * 0.008
    last_dt = int(counts[3])
    min_dt = int(counts[4])
    max_dt = int(counts[5])

    def fmt_raw_ns(raw_value: int) -> str:
        return "-" if raw_value < 0 else f"{raw_value * 0.008:.3f} ns"

    def peak_quality(histogram: np.ndarray, peak: int) -> tuple[float, float, float]:
        if histogram.size == 0 or peak < 0:
            return 0.0, 0.0, 0.0
        values = histogram.astype(float, copy=False)
        mask = np.ones(values.size, dtype=bool)
        left = max(0, peak - 2)
        right = min(values.size, peak + 3)
        mask[left:right] = False
        background = values[mask] if np.any(mask) else values
        mean = float(background.mean()) if background.size else 0.0
        std = float(background.std()) if background.size else 0.0
        ratio = float(values[peak]) / mean if mean > 0.0 else float(values[peak])
        return mean, std, ratio

    bg_mean, bg_std, peak_to_bg = peak_quality(hist, peak_bin)

    print(
        "tcspc: "
        f"sync={counts[1]} photon={counts[2]} accepted_pairs={counts[0]} "
        f"paired_photons={counts[14] if len(counts) > 14 else 0} "
        f"peak_bin={peak_bin} peak_ns={peak_ns:.3f} peak_count={peak_count} "
        f"bg={bg_mean:.3f}+/-{bg_std:.3f} peak_to_bg={peak_to_bg:.3f} "
        f"dt_last/min/max={fmt_raw_ns(last_dt)}/{fmt_raw_ns(min_dt)}/{fmt_raw_ns(max_dt)} "
        f"zero/neg/out/no_pair={counts[6]}/{counts[7]}/{counts[8]}/{counts[13]} "
        f"no_pair_before/after={counts[18] if len(counts) > 18 else 0}/"
        f"{counts[19] if len(counts) > 19 else 0} "
        f"diag_reg/trim/fold={counts[15] if len(counts) > 15 else 0}/"
        f"{counts[16] if len(counts) > 16 else 0}/{counts[17] if len(counts) > 17 else 0} "
        f"sync_buffer={counts[20] if len(counts) > 20 else 0} "
        f"history_depth_ms={(counts[21] if len(counts) > 21 else 0) * 0.008e-6:.3f}",
        flush=True,
    )

    legacy_peak_bin = int(legacy_modulo_hist.argmax()) if legacy_modulo_hist.size else -1
    legacy_peak_count = int(legacy_modulo_hist[legacy_peak_bin]) if legacy_peak_bin >= 0 else 0
    legacy_peak_ns = (
        (legacy_peak_bin * hist_builder.settings.bin_width_raw + hist_builder.settings.bin_offset) * 0.008
        if legacy_peak_bin >= 0
        else 0.0
    )
    legacy_bg_mean, legacy_bg_std, legacy_peak_to_bg = peak_quality(legacy_modulo_hist, legacy_peak_bin)
    print(
        "legacy_modulo_diag: "
        f"pairs={legacy_modulo_pairs} peak_bin={legacy_peak_bin} peak_ns={legacy_peak_ns:.3f} "
        f"peak_count={legacy_peak_count} bg={legacy_bg_mean:.3f}+/-{legacy_bg_std:.3f} "
        f"peak_to_bg={legacy_peak_to_bg:.3f}",
        flush=True,
    )

    def fmt_shape_time(raw_value: int | float) -> str:
        return f"{float(raw_value) * 0.008:.3f} ns"

    def fmt_counter_map(values: dict[int, int]) -> str:
        return "-".join(f"{key}:{value}" for key, value in values.items()) if values else "-"

    def concat_chunks(chunks: list[np.ndarray], dtype) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=dtype)
        return np.concatenate(chunks).astype(dtype, copy=False)

    def write_histogram_csv(path: Path, histogram: np.ndarray, settings: TdcTestSettings) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(histogram, dtype=np.uint64)
        bin_width = max(1, int(settings.bin_width_raw))
        bin_offset = int(settings.bin_offset)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bin", "dt_raw", "dt_ns", "count"])
            for bin_idx, count in enumerate(values):
                dt_raw = int(bin_idx * bin_width + bin_offset)
                writer.writerow([bin_idx, dt_raw, f"{dt_raw * 0.008:.6f}", int(count)])

    def write_selected_events_csv(
        path: Path,
        raw_index: np.ndarray,
        packet_seq: np.ndarray,
        item_index: np.ndarray,
        channels: np.ndarray,
        timestamps: np.ndarray,
        refids: np.ndarray,
        tstops: np.ndarray,
        delta_prev: np.ndarray,
        regression_flag: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = (channels == (args.start - 1)) | (channels == (args.stop - 1))
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "raw_index",
                    "packet_seq",
                    "item_index",
                    "channel",
                    "timestamp",
                    "refid",
                    "tstop",
                    "timestamp_delta_prev",
                    "regression_flag",
                ]
            )
            for idx in np.flatnonzero(selected):
                writer.writerow(
                    [
                        int(raw_index[idx]),
                        int(packet_seq[idx]),
                        int(item_index[idx]),
                        int(channels[idx]) + 1,
                        int(timestamps[idx]),
                        int(refids[idx]),
                        int(tstops[idx]),
                        int(delta_prev[idx]),
                        int(regression_flag[idx]),
                    ]
                )

    def write_report(path: Path, summary: dict) -> None:
        replay = summary.get("replay", {})
        histograms = replay.get("histograms", {})
        sorted_prev = histograms.get("sorted_prev_start", {})
        stream_prev = histograms.get("stream_prev_start", {})
        commercial = sorted_prev.get("commercial", {})
        shape = commercial.get("shape", {})
        lines = [
            "# GPX2 TCSPC Raw/Histogram Diagnostic Report",
            "",
            f"- Decision: `{replay.get('decision', 'not_available')}`",
            f"- Flags: `{', '.join(replay.get('flags', [])) or '-'}`",
            f"- Timestamp regressions: `{replay.get('timestamp_regressions', 0)}`",
            f"- Sorted prev-start peak: `{shape.get('peak_ns', 0.0):.6f} ns`, "
            f"count `{shape.get('peak_count', 0)}`, FWHM `{shape.get('fwhm_ns', 0.0):.6f} ns`",
            f"- Sorted prev-start peak/background: `{commercial.get('peak_to_bg', 0.0):.3f}`",
            f"- Deadtime violation: `{commercial.get('deadtime_violation_count', 0)}` "
            f"ratio `{commercial.get('deadtime_violation_ratio', 0.0):.6f}`",
            "",
            "## Stream vs Sorted",
            "",
            f"- Stream pairs: `{stream_prev.get('pairs', '-')}`; "
            f"sorted pairs: `{sorted_prev.get('pairs', '-')}`",
            f"- Stream no-pair ratio: `{stream_prev.get('no_pair_ratio', 0.0):.6f}`; "
            f"sorted no-pair ratio: `{sorted_prev.get('no_pair_ratio', 0.0):.6f}`",
            f"- Sorted peak-to-bg gain: "
            f"`{replay.get('stream_vs_sorted', {}).get('sorted_peak_to_bg_gain', 0.0):.3f}`",
            "",
            "## Interpretation",
            "",
        ]
        decision = str(replay.get("decision", ""))
        if decision.startswith("stream_order_fault"):
            lines.append(
                "Timestamp sorting materially improves the histogram. Focus on FPGA/host "
                "event ordering or add a host reorder buffer before trusting streaming GUI results."
            )
        elif decision.startswith("time_formula_fault"):
            lines.append(
                "A reverse fine-time formula is much sharper than the current forward formula. "
                "Fix the timestamp formula in both FPGA/host paths."
            )
        elif decision.startswith("start_retrigger"):
            lines.append(
                "Post-peak satellites align better with CH2 repeat intervals. Check sync input "
                "threshold, edge direction, termination, and comparator ringing."
            )
        elif decision.startswith("stop_retrigger"):
            lines.append(
                "Post-peak satellites align better with CH3 repeat intervals. Check avalanche "
                "front-end multi-edge triggering, afterpulse, threshold, and GPX2 hit filtering."
            )
        else:
            lines.append(
                "The exported raw data did not isolate a single software ordering/formula fault. "
                "Use the NPZ and CSV histograms for external replay, then compare clean electrical "
                "pulse input against the SPAD/front-end input."
            )
        lines.extend(
            [
                "",
                "## Files",
                "",
            ]
        )
        for name, file_path in summary.get("files", {}).items():
            lines.append(f"- `{name}`: `{file_path}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.shape_diag:
        if shape_diag_channels:
            diag_channels = np.concatenate(shape_diag_channels)
            diag_ext_refs = np.concatenate(shape_diag_ext_refs)
            diag_tstops = np.concatenate(shape_diag_tstops)
            shape_settings = TdcTestSettings(
                enabled=True,
                start_channel=args.start - 1,
                stop_channel=args.stop - 1,
                source="gpx2",
                refclk_divisions=hist_builder.settings.refclk_divisions,
                bin_width_raw=hist_builder.settings.bin_width_raw,
                bin_offset=hist_builder.settings.bin_offset,
                bin_count=hist_builder.settings.bin_count,
                acquisition_time_s=args.duration,
            )
            diagnostics = TcspcTimeReconstructor.shape_diagnostics(
                diag_channels,
                diag_ext_refs,
                diag_tstops,
                shape_settings,
            )
            print(
                "shape_diag_input: "
                f"events={diag_channels.size} start={int((diag_channels == args.start - 1).sum())} "
                f"stop={int((diag_channels == args.stop - 1).sum())}",
                flush=True,
            )
            for diag in diagnostics:
                shape = diag.shape
                print(
                    "shape_diag: "
                    f"name={diag.name} formula={diag.time_formula} pairing={diag.pairing_mode} "
                    f"pairs={diag.accepted_pairs} paired_photons={diag.paired_photons} "
                    f"no_pair={diag.orphan_photons} pairs_per_photon={diag.pairs_per_photon:.3f} "
                    f"peak={shape.peak_ns:.3f}ns count={shape.peak_count} "
                    f"fwhm={shape.fwhm_ns:.3f}ns rms={shape.rms_ns:.3f}ns "
                    f"sep={shape.peak_separation_ns:.3f}ns pbg={shape.peak_to_bg:.3f} "
                    f"coarse={fmt_counter_map(diag.coarse_ref_delta)} "
                    f"fine={fmt_counter_map(diag.fine_tstop_delta)}",
                    flush=True,
                )
            by_name = {diag.name: diag for diag in diagnostics}
            baseline = by_name.get("all_start_forward")
            viable = [diag for diag in diagnostics if diag.accepted_pairs > 0 and diag.shape.peak_count > 0]
            best = min(viable, key=lambda item: (item.shape.fwhm_raw, -item.shape.peak_to_bg), default=None)
            decision = "insufficient"
            if baseline is not None and best is not None and baseline.shape.fwhm_raw > 0:
                reverse_better = (
                    best.time_formula != "forward"
                    and best.shape.fwhm_raw * 5 <= baseline.shape.fwhm_raw
                    and 1200.0 <= best.shape.peak_separation_ns <= 2000.0
                )
                prev_same_formula = by_name.get(f"prev_start_{baseline.time_formula}")
                prev_better = (
                    prev_same_formula is not None
                    and prev_same_formula.shape.fwhm_raw > 0
                    and prev_same_formula.shape.fwhm_raw * 5 <= baseline.shape.fwhm_raw
                )
                if reverse_better:
                    decision = f"fix_time_formula:{best.time_formula}"
                elif prev_better:
                    decision = "pairing_pollution:prev_start_is_sharper"
                elif (
                    baseline.shape.fwhm_ns <= 1.0
                    and 1200.0 <= baseline.shape.peak_separation_ns <= 2000.0
                    and baseline.shape.peak_to_bg >= 20.0
                ):
                    decision = (
                        "host_reconstruction_ok:sub_ns_peak;"
                        "next_check_gpx2_calibration_or_input_jitter"
                    )
                else:
                    decision = "all_modes_wide:check_gpx2_calibration_or_input_jitter"
            print(
                "shape_decision: "
                f"decision={decision} "
                f"baseline_fwhm={fmt_shape_time(baseline.shape.fwhm_raw) if baseline else '-'} "
                f"best={best.name if best else '-'} "
                f"best_fwhm={fmt_shape_time(best.shape.fwhm_raw) if best else '-'}",
                flush=True,
            )
        else:
            print("shape_diag_input: no EXT raw events captured", flush=True)

    if args.satellite_diag:
        if shape_diag_channels:
            diag_channels = np.concatenate(shape_diag_channels).astype(np.int64, copy=False)
            diag_ext_refs = np.concatenate(shape_diag_ext_refs).astype(np.int64, copy=False)
            diag_tstops = np.concatenate(shape_diag_tstops).astype(np.int64, copy=False)
            sat_settings = TdcTestSettings(
                enabled=True,
                start_channel=args.start - 1,
                stop_channel=args.stop - 1,
                source="gpx2",
                pairing_mode="prev_start",
                refclk_divisions=hist_builder.settings.refclk_divisions,
                bin_width_raw=hist_builder.settings.bin_width_raw,
                bin_offset=hist_builder.settings.bin_offset,
                bin_count=hist_builder.settings.bin_count,
                acquisition_time_s=args.duration,
                reference_deadtime_ns=args.reference_deadtime_ns,
            )
            refclk_div = max(1, int(sat_settings.refclk_divisions))
            timestamps = diag_ext_refs * refclk_div + diag_tstops
            start_mask = diag_channels == int(sat_settings.start_channel)
            stop_mask = diag_channels == int(sat_settings.stop_channel)
            start_order = np.argsort(timestamps[start_mask], kind="mergesort")
            stop_order = np.argsort(timestamps[stop_mask], kind="mergesort")
            raw_sync_ts = timestamps[start_mask][start_order]
            stop_ts = timestamps[stop_mask][stop_order]
            raw_sync_refs = diag_ext_refs[start_mask][start_order]
            stop_refs = diag_ext_refs[stop_mask][stop_order]
            raw_sync_fine = diag_tstops[start_mask][start_order]
            stop_fine = diag_tstops[stop_mask][stop_order]
            reference_deadtime_raw = int(
                round(max(0.0, float(args.reference_deadtime_ns)) / TcspcTimeReconstructor.TICK_NS)
            )
            cleaned_sync_indices, reference_filtered = TdcTestHistogramBuilder._filter_channel_deadtime(
                np.arange(raw_sync_ts.size, dtype=np.int64),
                raw_sync_ts,
                reference_deadtime_raw,
            )
            sync_ts = raw_sync_ts[cleaned_sync_indices]
            sync_refs = raw_sync_refs[cleaned_sync_indices]
            sync_fine = raw_sync_fine[cleaned_sync_indices]
            min_dt_raw, max_dt_raw = TdcTestHistogramBuilder.window_bounds_for_settings(sat_settings)
            pairs = TcspcTimeReconstructor.pair_indices(
                sync_ts,
                stop_ts,
                min_dt_raw,
                max_dt_raw,
                "prev_start",
            )
            sat_hist = TcspcTimeReconstructor.histogram_from_dt(pairs.dt_raw, sat_settings)
            sat_shape = TcspcTimeReconstructor.measure_histogram_shape(sat_hist, sat_settings)
            bin_width = max(1, int(sat_settings.bin_width_raw))
            bin_width_ns = bin_width * 0.008
            peak_bin_sat = int(sat_shape.peak_bin)
            peak_count_sat = int(sat_shape.peak_count)
            threshold = max(
                3,
                int(np.ceil(float(sat_shape.background_mean) + 6.0 * float(sat_shape.background_std))),
                int(np.ceil(float(peak_count_sat) * 0.01)),
            )
            exclude_bins = max(3, int(round(2.0 / max(bin_width_ns, 1e-12))))
            min_sep_bins = max(1, int(round(5.0 / max(bin_width_ns, 1e-12))))
            search_lo = min(sat_hist.size - 1, max(1, peak_bin_sat + exclude_bins))
            search_hi = min(
                sat_hist.size - 1,
                peak_bin_sat + max(exclude_bins + 1, int(round(700.0 / max(bin_width_ns, 1e-12)))),
            )
            selected_peaks: list[int] = []
            if 0 < search_lo < search_hi:
                values_i = sat_hist.astype(np.int64, copy=False)
                candidate_region = np.arange(search_lo, search_hi, dtype=np.int64)
                is_peak = (
                    (values_i[candidate_region] >= values_i[candidate_region - 1])
                    & (values_i[candidate_region] >= values_i[candidate_region + 1])
                    & (values_i[candidate_region] >= threshold)
                )
                candidates = candidate_region[is_peak]
                candidates = candidates[np.argsort(values_i[candidates])[::-1]]
                for idx in candidates:
                    if all(abs(int(idx) - existing) >= min_sep_bins for existing in selected_peaks):
                        selected_peaks.append(int(idx))
                    if len(selected_peaks) >= 12:
                        break
                selected_peaks.sort()

            pair_bins = (
                (pairs.dt_raw - int(sat_settings.bin_offset)) // bin_width
                if pairs.dt_raw.size
                else np.array([], dtype=np.int64)
            )

            def pair_window_counters(target_bin: int, half_width_bins: int = 2) -> tuple[str, str, int]:
                if pair_bins.size == 0:
                    return "-", "-", 0
                in_window = (pair_bins >= target_bin - half_width_bins) & (pair_bins <= target_bin + half_width_bins)
                pair_count = int(in_window.sum())
                if pair_count <= 0:
                    return "-", "-", 0
                coarse = stop_refs[pairs.photon_indices[in_window]] - sync_refs[pairs.sync_indices[in_window]]
                fine = stop_fine[pairs.photon_indices[in_window]] - sync_fine[pairs.sync_indices[in_window]]
                return (
                    fmt_counter_map(dict(Counter(int(value) for value in coarse).most_common(6))),
                    fmt_counter_map(dict(Counter(int(value) for value in fine).most_common(6))),
                    pair_count,
                )

            main_coarse, main_fine, main_pairs = pair_window_counters(peak_bin_sat)

            def repeat_histogram(event_ts: np.ndarray) -> np.ndarray:
                hist = np.zeros(int(sat_settings.bin_count), dtype=np.uint32)
                if event_ts.size < 2:
                    return hist
                diffs = np.diff(event_ts.astype(np.int64, copy=False))
                diffs = diffs[(diffs > 0) & (diffs <= int(round(700.0 / 0.008)))]
                if diffs.size == 0:
                    return hist
                interval_bins = diffs // bin_width
                valid = (interval_bins > 0) & (interval_bins < hist.size)
                if np.any(valid):
                    counts = np.bincount(interval_bins[valid].astype(np.int64, copy=False), minlength=hist.size)
                    nonzero = np.nonzero(counts)[0]
                    hist[nonzero] = counts[nonzero].astype(hist.dtype, copy=False)
                return hist

            def repeat_summary(event_ts: np.ndarray) -> tuple[str, dict[int, int]]:
                hist = repeat_histogram(event_ts)
                if hist.sum() == 0 or not selected_peaks:
                    return "-", {}
                interval_bins = np.flatnonzero(hist)
                counts = Counter({int(bin_idx): int(hist[bin_idx]) for bin_idx in interval_bins})
                top = counts.most_common(8)
                text = "|".join(
                    f"{(bin_idx * bin_width) * 0.008:.3f}ns:{count}" for bin_idx, count in top
                )
                near_counts: dict[int, int] = {}
                for sat_bin in selected_peaks:
                    offset_raw = abs(sat_bin - peak_bin_sat) * bin_width
                    target = int(round(offset_raw / bin_width))
                    near_counts[sat_bin] = sum(
                        count for bin_idx, count in counts.items() if abs(bin_idx - target) <= 2
                    )
                return text or "-", near_counts

            start_repeat_text, start_repeat_near = repeat_summary(raw_sync_ts)
            stop_repeat_text, stop_repeat_near = repeat_summary(stop_ts)
            start_repeat_hist = repeat_histogram(raw_sync_ts)
            stop_repeat_hist = repeat_histogram(stop_ts)
            commercial = TcspcCommercialAnalyzer.analyze(
                sat_hist,
                sat_settings,
                start_repeat_histogram=start_repeat_hist,
                stop_repeat_histogram=stop_repeat_hist,
                expected_peak_separation_raw=TcspcTimeReconstructor.estimate_sync_period_raw(sync_ts),
                detector_deadtime_ns=100.0,
            )
            peak_text = (
                f"main_dt={sat_shape.peak_ns:.3f}ns count={peak_count_sat} "
                f"fwhm={sat_shape.fwhm_ns:.3f}ns threshold={threshold} "
                f"main_pairs={main_pairs} main_coarse={main_coarse} main_fine={main_fine} "
                f"reference_cleanup={args.reference_deadtime_ns:.1f}ns filtered={reference_filtered}"
            )
            print(f"satellite_diag: {peak_text}", flush=True)
            print(
                "commercial_diag: "
                f"fwhm={commercial.shape.fwhm_ns:.3f}ns "
                f"bg={commercial.background_mean:.3f}+/-{commercial.background_std:.3f} "
                f"cv={commercial.background_cv:.3f} pbg={commercial.peak_to_bg:.3f} "
                f"deadtime_violation={commercial.deadtime_violation_count} "
                f"ratio={commercial.deadtime_violation_ratio:.6f} "
                f"deadtime_window={commercial.deadtime_start_ns:.3f}..{commercial.deadtime_end_ns:.3f}ns "
                f"decision={commercial.satellite_decision}",
                flush=True,
            )
            if selected_peaks:
                sat_parts: list[str] = []
                for sat_bin in selected_peaks:
                    coarse_text, fine_text, pair_count = pair_window_counters(sat_bin)
                    sat_ns = (sat_bin * bin_width + int(sat_settings.bin_offset)) * 0.008
                    offset_ns = (sat_bin - peak_bin_sat) * bin_width * 0.008
                    sat_parts.append(
                        f"dt={sat_ns:.3f}ns(+{offset_ns:.3f}) "
                        f"count={int(sat_hist[sat_bin])} ratio={int(sat_hist[sat_bin]) / max(1, peak_count_sat):.4f} "
                        f"pairs={pair_count} coarse={coarse_text} fine={fine_text} "
                        f"near_stop_repeat={stop_repeat_near.get(sat_bin, 0)} "
                        f"near_start_repeat={start_repeat_near.get(sat_bin, 0)}"
                    )
                print("satellite_peaks: " + " | ".join(sat_parts), flush=True)
            else:
                print("satellite_peaks: none_above_threshold", flush=True)
            print(
                "satellite_repeat_intervals: "
                f"start_top={start_repeat_text} stop_top={stop_repeat_text}",
                flush=True,
            )
            stop_hits = sum(stop_repeat_near.get(bin_idx, 0) for bin_idx in selected_peaks)
            start_hits = sum(start_repeat_near.get(bin_idx, 0) for bin_idx in selected_peaks)
            if selected_peaks and stop_hits > max(5, start_hits * 3):
                decision = "likely_stop_channel_retrigger_or_afterpulse"
            elif selected_peaks and start_hits > max(5, stop_hits * 3):
                decision = "likely_start_channel_multi_edge_affecting_nearest_start"
            elif selected_peaks:
                decision = "satellites_present_but_repeat_source_inconclusive"
            else:
                decision = "no_significant_satellites"
            print(
                "satellite_decision: "
                f"{decision} start_repeat_hits={start_hits} stop_repeat_hits={stop_hits}",
                flush=True,
            )
        else:
            print("satellite_diag: no EXT raw events captured", flush=True)

    stats = service.runtime_stats
    elapsed = max(0.001, float(args.duration))
    rx_bytes = stats.rx_bytes if args.live_parse else measurement_rx_bytes
    rx_rate = rx_bytes / elapsed
    print(
        "summary: "
        f"raw_blocks={raw_blocks} status_packets={status_packets} "
        f"tdc_raw_packets={stats.tdc_raw_packet_count} "
        f"tdc_raw_events={stats.tdc_raw_event_count} "
        f"legacy_events={legacy_tdc_events} "
        f"raw_batches={raw_batches} raw_batch_events={raw_batch_events} "
        f"raw_packet_kind={fmt_hist(raw_packet_kind_counts)} "
        f"ext_raw_events={ext_raw_events} ext_ts_regressions={ext_timestamp_regressions} "
        f"packet_seq_gaps={packet_seq_gap_count} parser_rejects={fmt_hist(parser_reject_counts)} "
        f"raw_batch_ch={raw_channel_counts[0]}/{raw_channel_counts[1]}/"
        f"{raw_channel_counts[2]}/{raw_channel_counts[3]} "
        f"last_raw_ch={stats.last_tdc_raw_channels[0]}/{stats.last_tdc_raw_channels[1]}/"
        f"{stats.last_tdc_raw_channels[2]}/{stats.last_tdc_raw_channels[3]} "
        f"rx_bytes={rx_bytes} tx_bytes={stats.tx_bytes} rx_rate={fmt_rate(rx_rate)} "
        f"raw_payload_words={fmt_hist(raw_payload_word_hist)} "
        f"raw_item_count={fmt_hist(raw_item_count_hist)} "
        f"raw_packet_bytes={fmt_hist(raw_packet_byte_hist)}",
        flush=True,
    )
    print(
        "debug_events: "
        + (" | ".join(sample_events) if sample_events else "-"),
        flush=True,
    )
    print(
        "debug_unselected_events: "
        + (" | ".join(unselected_samples) if unselected_samples else "-"),
        flush=True,
    )
    print(
        "debug_timestamp_regressions: "
        + (" | ".join(regression_samples) if regression_samples else "-"),
        flush=True,
    )
    print(
        "debug_packet_seq_gaps: "
        + (" | ".join(packet_seq_gap_samples) if packet_seq_gap_samples else "-"),
        flush=True,
    )
    print(
        "debug_parser_rejects: "
        + (" | ".join(parser_reject_samples) if parser_reject_samples else "-"),
        flush=True,
    )
    if artifact_requested:
        files: dict[str, str] = {}
        if "capture_path" in locals() and capture_path is not None:
            files["raw_usb.tdcpack"] = str(capture_path)
        raw_index = concat_chunks(event_raw_index_chunks, np.uint64)
        packet_seq = concat_chunks(event_packet_seq_chunks, np.uint32)
        item_index = concat_chunks(event_item_index_chunks, np.uint32)
        event_channels = concat_chunks(event_channel_chunks, np.uint8)
        event_timestamps = concat_chunks(event_timestamp_chunks, np.uint64)
        event_refids = concat_chunks(event_refid_chunks, np.uint32)
        event_tstops = concat_chunks(event_tstop_chunks, np.uint32)
        event_delta_prev = concat_chunks(event_delta_prev_chunks, np.int64)
        event_regression = concat_chunks(event_regression_chunks, np.uint8)
        if need_event_table or args.export_events:
            events_npz = export_dir / "events_ext_raw.npz"
            np.savez_compressed(
                events_npz,
                raw_index=raw_index,
                packet_seq=packet_seq,
                item_index=item_index,
                channel=event_channels,
                timestamp=event_timestamps,
                refid=event_refids,
                tstop=event_tstops,
                timestamp_delta_prev=event_delta_prev,
                regression_flag=event_regression,
            )
            files["events_ext_raw.npz"] = str(events_npz)
            events_csv = export_dir / f"events_selected_ch{args.start}_ch{args.stop}.csv.gz"
            write_selected_events_csv(
                events_csv,
                raw_index,
                packet_seq,
                item_index,
                event_channels,
                event_timestamps,
                event_refids,
                event_tstops,
                event_delta_prev,
                event_regression,
            )
            files["events_selected_ch2_ch3.csv.gz"] = str(events_csv)

        replay_settings = TdcTestSettings(
            enabled=True,
            start_channel=args.start - 1,
            stop_channel=args.stop - 1,
            source="gpx2",
            pairing_mode="prev_start",
            time_formula="forward",
            refclk_divisions=hist_builder.settings.refclk_divisions,
            bin_width_raw=hist_builder.settings.bin_width_raw,
            bin_offset=hist_builder.settings.bin_offset,
            bin_count=hist_builder.settings.bin_count,
            acquisition_time_s=args.duration,
            tdc_deadtime_ps=float(args.tdc_deadtime_ps),
            reference_deadtime_ns=float(args.reference_deadtime_ns),
            detector_deadtime_ns=float(getattr(hist_builder.settings, "detector_deadtime_ns", 100.0)),
        )
        replay_result = TcspcRawReplayAnalyzer.analyze_events(
            event_channels,
            event_timestamps,
            event_refids,
            event_tstops,
            replay_settings,
            stream_histogram=hist,
            stream_counts=counts,
            stream_start_repeat_histogram=np.asarray(
                snapshot.histograms.get("tdc_start_repeat", np.zeros(replay_settings.bin_count, dtype=np.uint32)),
                dtype=np.uint32,
            ),
            stream_stop_repeat_histogram=np.asarray(
                snapshot.histograms.get("tdc_stop_repeat", np.zeros(replay_settings.bin_count, dtype=np.uint32)),
                dtype=np.uint32,
            ),
            tdc_deadtime_ps=float(args.tdc_deadtime_ps),
            reference_deadtime_ns=float(args.reference_deadtime_ns),
        )
        replay_summary = TcspcRawReplayAnalyzer.result_to_summary(replay_result)
        if args.export_histograms or args.sorted_replay or artifact_requested:
            hist_files = {
                "hist_stream_prev_start.csv": (
                    replay_result.stream_prev_start.histogram
                    if replay_result.stream_prev_start is not None
                    else hist
                ),
                "hist_sorted_prev_start.csv": replay_result.sorted_prev_start.histogram,
                "hist_sorted_prev_start_deadtime.csv": replay_result.sorted_prev_start_deadtime.histogram,
                "hist_sorted_all_start.csv": replay_result.sorted_all_start.histogram,
                "hist_start_repeat.csv": replay_result.start_repeat_histogram,
                "hist_stop_repeat.csv": replay_result.stop_repeat_histogram,
            }
            for file_name, histogram_values in hist_files.items():
                path = export_dir / file_name
                write_histogram_csv(path, histogram_values, replay_settings)
                files[file_name] = str(path)

        status_summary = None
        if last_status is not None:
            status_summary = {
                "flags": int(last_status.flags),
                "uptime_seconds": int(last_status.uptime_seconds),
                "tdc_drop_count": int(last_status.tdc_drop_count),
                "usb_drop_count": int(last_status.usb_drop_count),
                "gpx2_raw_counts": [
                    int(last_status.gpx2_raw_count_ch1),
                    int(last_status.gpx2_raw_count_ch2),
                    int(last_status.gpx2_raw_count_ch3),
                    int(last_status.gpx2_raw_count_ch4),
                ],
                "gpx2_cfg_diag": int(last_status.gpx2_cfg_diag),
            }
        summary_doc = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
            "files": files,
            "packet_stats": {
                "raw_blocks": int(raw_blocks),
                "status_packets": int(status_packets),
                "tdc_raw_packets": int(stats.tdc_raw_packet_count),
                "tdc_raw_events": int(stats.tdc_raw_event_count),
                "raw_batches": int(raw_batches),
                "raw_batch_events": int(raw_batch_events),
                "raw_packet_kind": dict(raw_packet_kind_counts),
                "raw_payload_words": {str(k): int(v) for k, v in raw_payload_word_hist.items()},
                "raw_item_count": {str(k): int(v) for k, v in raw_item_count_hist.items()},
                "raw_packet_bytes": {str(k): int(v) for k, v in raw_packet_byte_hist.items()},
                "packet_seq_gaps": int(packet_seq_gap_count),
                "parser_rejects": {str(k): int(v) for k, v in parser_reject_counts.items()},
                "rx_bytes": int(rx_bytes),
                "rx_rate_Bps": float(rx_rate),
                "ext_raw_events": int(ext_raw_events),
                "ext_timestamp_regressions": int(ext_timestamp_regressions),
                "raw_batch_ch": [int(value) for value in raw_channel_counts],
            },
            "status": status_summary,
            "tcspc_stream": {
                "sync": int(counts[1]),
                "photon": int(counts[2]),
                "accepted_pairs": int(counts[0]),
                "paired_photons": int(counts[14] if len(counts) > 14 else 0),
                "orphan_photons": int(counts[13] if len(counts) > 13 else 0),
                "timestamp_regression": int(counts[15] if len(counts) > 15 else 0),
                "reference_filtered": int(counts[26] if len(counts) > 26 else 0),
                "peak_bin": int(peak_bin),
                "peak_ns": float(peak_ns),
                "peak_count": int(peak_count),
                "background_mean": float(bg_mean),
                "background_std": float(bg_std),
                "peak_to_bg": float(peak_to_bg),
            },
            "replay": replay_summary,
        }
        summary_path = export_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        files["summary.json"] = str(summary_path)
        summary_doc["files"] = files
        summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        report_path = export_dir / "report.md"
        write_report(report_path, summary_doc)
        files["report.md"] = str(report_path)
        summary_doc["files"] = files
        summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"artifact_dir: {export_dir}", flush=True)
        print(f"artifact_decision: {replay_result.decision} flags={','.join(replay_result.flags)}", flush=True)
    if exact_ref_dt_samples:
        dt = np.asarray(exact_ref_dt_samples, dtype=np.int64)
        p50 = int(np.percentile(dt, 50))
        p90 = int(np.percentile(dt, 90))
        print(
            "debug_exact_ref_dt_raw: "
            f"count={dt.size} min={int(dt.min())} p50={p50} p90={p90} max={int(dt.max())} "
            f"min_ns={int(dt.min()) * 0.008:.3f} p50_ns={p50 * 0.008:.3f} max_ns={int(dt.max()) * 0.008:.3f}",
            flush=True,
        )
    else:
        print("debug_exact_ref_dt_raw: count=0", flush=True)
    if (
        last_status is not None
        and status_at_start is not None
        and (last_status.gpx2_raw_count_ch2 != status_at_start.gpx2_raw_count_ch2
             or last_status.gpx2_raw_count_ch3 != status_at_start.gpx2_raw_count_ch3)
        and stats.tdc_raw_packet_count == 0
        and ext_raw_events == 0
    ):
        print(
            "diagnosis: FPGA raw counters changed but no TDC raw packet reached host. "
            "Check gpx2_stream -> async_fifo -> uplink_packet_builder -> tx_fifo/FT601.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
