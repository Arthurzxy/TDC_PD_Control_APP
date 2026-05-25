"""Probe the FT601 uplink with the FPGA synthetic TDC source.

Run with:
    python -m app.tests.ft601_tdc_synthetic_probe --duration 5

Expected when the programmed FPGA bitstream is alive:
    - CMD_TDC_TEST_UPLOAD payload word is 0x00000063 for CH2/CH3 synthetic.
    - RX USB BLOCK and RX PARSED PACKET type=TDC_RAW messages appear.
    - The summary reports nonzero packets/events.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PyQt5 import QtCore

from app.data_processing import TdcTestHistogramBuilder
from app.models import TdcTestSettings
from app import protocol
from app.protocol import CommandEncoder
from app.usb_link import D3XXError, DeviceService


def main() -> int:
    parser = argparse.ArgumentParser(description="FT601 synthetic TDC uplink probe")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-size", type=int, default=16 * 1024)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--start", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--stop", type=int, choices=[1, 2, 3, 4], default=3)
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    encoder = CommandEncoder()

    packet_count = 0
    event_count = 0
    raw_blocks = 0
    raw_batch_count = 0
    raw_channel_counts = [0, 0, 0, 0]
    sample_events: list[str] = []
    sync_tstop_by_refid: dict[int, int] = {}
    exact_ref_dt_samples: list[int] = []
    measurement_active = False
    hist_builder = TdcTestHistogramBuilder(
        TdcTestSettings(
            enabled=True,
            start_channel=args.start - 1,
            stop_channel=args.stop - 1,
            source="synthetic",
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=300000,
            acquisition_time_s=args.duration,
        )
    )

    def log(message: str) -> None:
        print(message, flush=True)

    def on_packet(packet) -> None:
        nonlocal packet_count
        if not measurement_active:
            return
        packet_count += 1

    def on_events(events) -> None:
        nonlocal event_count
        if not measurement_active:
            return
        event_count += len(events)

    def on_batch(batch) -> None:
        nonlocal raw_batch_count
        if not measurement_active:
            return
        raw_batch_count += 1
        hist_builder.process_batch(batch)
        channels = np.asarray(batch.channels, dtype=np.int64)
        refids = np.asarray(batch.refids, dtype=np.int64)
        tstops = np.asarray(batch.tstops, dtype=np.int64)
        if len(sample_events) < 16:
            remaining = 16 - len(sample_events)
            for idx in range(min(remaining, len(batch))):
                sample_events.append(
                    f"CH{int(channels[idx]) + 1}:ref={int(refids[idx])}:tstop={int(tstops[idx])}"
                )
        for idx in range(len(batch)):
            channel = int(channels[idx])
            refid = int(refids[idx])
            tstop = int(tstops[idx])
            if channel == args.start - 1:
                sync_tstop_by_refid[refid] = tstop
            elif channel == args.stop - 1 and refid in sync_tstop_by_refid:
                exact_ref_dt_samples.append(tstop - int(sync_tstop_by_refid[refid]))
        counts = np.bincount(channels[(channels >= 0) & (channels < 4)], minlength=4)
        for idx in range(4):
            raw_channel_counts[idx] += int(counts[idx])

    def on_raw(_data: bytes) -> None:
        nonlocal raw_blocks
        if not measurement_active:
            return
        raw_blocks += 1

    service.log_message.connect(log)
    service.set_raw_bytes_signal_enabled(True)
    service.set_emit_packet_objects(True)
    service.packet_received.connect(on_packet)
    service.tdc_raw_batch_received.connect(on_batch)
    service.tdc_events_received.connect(on_events)
    service.raw_bytes_received.connect(on_raw)
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    channel_mask = (1 << (args.start - 1)) | (1 << (args.stop - 1))
    enable_frame = encoder.encode_tdc_test_upload(True, channel_mask, synthetic=True)
    disable_frame = encoder.encode_tdc_test_upload(False, 0, synthetic=False)

    try:
        service.open_device(
            device_index=args.index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=args.read_size,
            read_enabled=True,
        )
    except D3XXError as exc:
        print(
            "open failed: "
            f"{exc}. Close the GUI/app using FT601 first, unplug/replug if needed, "
            "then rerun this probe.",
            flush=True,
        )
        return 1

    try:
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="synthetic probe pre-disable any previous TDC upload",
        )
        packet_count = 0
        event_count = 0
        raw_blocks = 0
        raw_batch_count = 0
        raw_channel_counts = [0, 0, 0, 0]
        sample_events.clear()
        sync_tstop_by_refid.clear()
        exact_ref_dt_samples.clear()
        hist_builder.clear()
        measurement_active = True
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            enable_frame,
            timeout_ms=1000,
            debug_details=f"synthetic probe start=CH{args.start} stop=CH{args.stop}",
        )

        deadline = time.monotonic() + max(0.1, float(args.duration))
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.02)

        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="synthetic probe stop",
        )
        for _ in range(10):
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.02)
        measurement_active = False
    finally:
        service.close_device()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    stats = service.runtime_stats
    snapshot = hist_builder.snapshot()
    hist = np.asarray(snapshot.histograms["tdc_test"], dtype=np.uint32)
    counts = snapshot.image_projection[0]
    peak_bin = int(hist.argmax()) if hist.size else -1
    peak_count = int(hist[peak_bin]) if peak_bin >= 0 else 0
    peak_ns = (peak_bin * hist_builder.settings.bin_width_raw + hist_builder.settings.bin_offset) * 0.008
    print(
        "summary: "
        f"raw_blocks={raw_blocks} packets={packet_count} tdc_events={event_count} raw_batches={raw_batch_count} "
        f"raw_batch_ch={raw_channel_counts[0]}/{raw_channel_counts[1]}/"
        f"{raw_channel_counts[2]}/{raw_channel_counts[3]} "
        f"rx_bytes={stats.rx_bytes} tx_bytes={stats.tx_bytes}",
        flush=True,
    )
    print(
        "tcspc: "
        f"sync={counts[1]} photon={counts[2]} accepted_pairs={counts[0]} "
        f"paired_photons={counts[14] if len(counts) > 14 else 0} "
        f"peak_bin={peak_bin} peak_ns={peak_ns:.3f} peak_count={peak_count} "
        f"zero/neg/out/no_pair={counts[6]}/{counts[7]}/{counts[8]}/{counts[13]}",
        flush=True,
    )
    print(
        "debug_events: "
        + (" | ".join(sample_events) if sample_events else "-"),
        flush=True,
    )
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
    return 0 if raw_blocks > 0 and event_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
