"""Probe the FPGA-side GPX2 TCSPC histogram path.

Run with the GUI closed:
    python -m app.tests.ft601_gpx2_fpga_hist_probe --duration 10 --start 2 --stop 3

The probe configures GPX2, enables the FPGA histogram core, periodically reads
both commercial/first-stop and diagnostic/all-stop histograms, and optionally
exports them as CSV files.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PyQt5 import QtCore

from app import protocol
from app.fpga_control import AnalogControlService, FpgaControlService
from app.models import Ddr3BistResult, TdcHistogramChunk
from app.protocol import (
    GPX2_DEFAULT_CONFIG_BYTES,
    GPX2_DEFAULT_PROFILE,
    GPX2_DEFAULT_SEQUENCE,
    LegacyAnalogCodec,
    decode_tdc_hist_status_flags,
    describe_tdc_hist_status_flags,
)
from app.usb_link import D3XXError, DeviceService


def _write_histogram_csv(path: Path, histogram: np.ndarray, bin_width_raw: int, offset_raw: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "dt_raw", "dt_ns", "count"])
        for bin_index, count in enumerate(np.asarray(histogram, dtype=np.uint64)):
            dt_raw = offset_raw + bin_index * int(bin_width_raw)
            writer.writerow([bin_index, dt_raw, f"{dt_raw * 0.008:.6f}", int(count)])


def _peak_summary(histogram: np.ndarray, bin_width_raw: int, offset_raw: int) -> str:
    if histogram.size == 0 or int(histogram.sum()) == 0:
        return "empty"
    peak_bin = int(np.argmax(histogram))
    peak_count = int(histogram[peak_bin])
    peak_raw = int(offset_raw) + peak_bin * int(bin_width_raw)
    return (
        f"sum={int(np.asarray(histogram, dtype=np.uint64).sum())} "
        f"peak_bin={peak_bin} peak_ns={peak_raw * 0.008:.3f} peak_count={peak_count}"
    )


def _bist_summary(result: Ddr3BistResult) -> str:
    return (
        f"mode={result.mode} calibrated={int(result.calibrated)} done={int(result.done)} "
        f"pass={int(result.passed)} fail={int(result.failed)} "
        f"requested_words={result.requested_words} tested_words={result.tested_words} "
        f"errors={result.error_count} first_error_addr=0x{result.first_error_addr:08X} "
        f"expected=0x{result.expected:032X} actual=0x{result.actual:032X} "
        f"write_cycles={result.write_cycles} read_cycles={result.read_cycles} "
        f"app={result.app_data_width}b/{result.app_addr_width}addr"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="FT601 GPX2 FPGA histogram probe")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-size", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--readout-interval", type=float, default=0.5)
    parser.add_argument("--chunk-retries", type=int, default=5)
    parser.add_argument("--fail-fast-missing", action="store_true")
    parser.add_argument("--chunked-readout", action="store_true", help="Request each 256-bin chunk separately instead of one full snapshot.")
    readout_mode = parser.add_mutually_exclusive_group()
    readout_mode.add_argument(
        "--disable-before-readout",
        dest="disable_before_readout",
        action="store_true",
        default=True,
        help="Disable FPGA histogram acquisition before readout, without clearing accumulated bins (default).",
    )
    readout_mode.add_argument(
        "--live-readout-without-disable",
        dest="disable_before_readout",
        action="store_false",
        help="Request readout while acquisition remains enabled. This is a negative test for firmware that ignores live snapshots.",
    )
    parser.add_argument("--hist-mask", type=lambda value: int(value, 0), default=0x1, help="Histogram readout mask: 0x1 commercial, 0x2 diagnostic, 0x3 both.")
    parser.add_argument("--start", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--stop", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--bin-width-raw", type=int, default=5)
    parser.add_argument("--bin-count", type=int, default=50000)
    parser.add_argument("--offset-raw", type=int, default=0)
    parser.add_argument("--reference-deadtime-ns", type=float, default=0.0)
    parser.add_argument("--tdc-deadtime-ps", type=float, default=0.0)
    parser.add_argument("--skip-ddr3-bist", action="store_true")
    parser.add_argument("--bist-words", type=int, default=4096)
    parser.add_argument("--bist-timeout", type=float, default=5.0)
    parser.add_argument("--bist-modes", default="0,1,2,3", help="Comma-separated DDR3 BIST modes to run before histogram.")
    parser.add_argument("--skip-threshold-setup", action="store_true")
    parser.add_argument("--laser-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--pixel-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--avalanche-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--bias-v", type=float, default=50.0)
    parser.add_argument("--export-dir", default="", help="Optional directory for histogram CSV exports.")
    parser.add_argument("--verbose-usb-log", action="store_true", help="Print every USB command and transfer log.")
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    fpga = FpgaControlService(service)
    analog = AnalogControlService(service, LegacyAnalogCodec())

    total_bins = max(1, int(args.bin_count))
    histograms = {
        0: np.zeros(total_bins, dtype=np.uint32),
        1: np.zeros(total_bins, dtype=np.uint32),
    }
    requested_hist_ids = tuple(hist_id for hist_id in (0, 1) if (int(args.hist_mask) & (1 << hist_id)))
    if not requested_hist_ids:
        requested_hist_ids = (0,)
    chunk_counts = {0: 0, 1: 0}
    seen_chunks = {0: set(), 1: set()}
    last_chunks: dict[int, TdcHistogramChunk] = {}
    bist_results: list[Ddr3BistResult] = []
    packet_types: Counter[int] = Counter()
    packet_items: Counter[tuple[int, int]] = Counter()

    def log(message: str) -> None:
        print(message, flush=True)

    def on_chunk(chunk: TdcHistogramChunk) -> None:
        hist_id = int(chunk.hist_id)
        if hist_id not in histograms:
            return
        if histograms[hist_id].size != int(chunk.total_bins):
            histograms[hist_id] = np.zeros(int(chunk.total_bins), dtype=np.uint32)
        start = int(chunk.bin_start)
        stop = min(histograms[hist_id].size, start + int(chunk.bin_count))
        if stop > start:
            histograms[hist_id][start:stop] = np.asarray(chunk.bins[: stop - start], dtype=np.uint32)
        chunk_counts[hist_id] += 1
        if int(chunk.bin_count) > 0:
            seen_chunks[hist_id].add(start // 256)
        last_chunks[hist_id] = chunk

    def on_bist(result: Ddr3BistResult) -> None:
        bist_results.append(result)

    def on_packet(packet) -> None:
        pkt_type = int(packet.header.pkt_type)
        packet_types[pkt_type] += 1
        packet_items[(pkt_type, int(packet.header.item_count))] += 1

    if args.verbose_usb_log:
        service.log_message.connect(log)
    service.set_emit_packet_objects(True)
    service.packet_received.connect(on_packet)
    service.tdc_histogram_chunk_received.connect(on_chunk)
    service.ddr3_bist_result_received.connect(on_bist)
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    start_channel = int(args.start) - 1
    stop_channel = int(args.stop) - 1
    reference_cleanup_raw = max(0, int(round(float(args.reference_deadtime_ns) * 125.0)))
    tdc_deadtime_raw = max(0, int(round(float(args.tdc_deadtime_ps) / 8.0)))

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
        if not args.skip_threshold_setup:
            analog_result = analog.set_outputs(
                args.laser_threshold_mv,
                args.pixel_threshold_mv,
                args.avalanche_threshold_mv,
                args.bias_v,
                timeout_ms=1000,
            )
            print(f"analog_thresholds: success={analog_result.success} message={analog_result.message}", flush=True)

        fpga.configure_tdc_test_upload(False, 0, timeout_ms=1000)
        fpga.configure_tdc_histogram(
            enable=False,
            clear=True,
            start_channel=start_channel,
            stop_channel=stop_channel,
            bin_width_raw=args.bin_width_raw,
            bin_count=total_bins,
            offset_raw=args.offset_raw,
            reference_cleanup_raw=reference_cleanup_raw,
            tdc_deadtime_raw=tdc_deadtime_raw,
            timeout_ms=1000,
        )
        if not args.skip_ddr3_bist:
            modes: list[int] = []
            for item in str(args.bist_modes).split(","):
                item = item.strip()
                if not item:
                    continue
                modes.append(int(item, 0) & 0xF)
            if not modes:
                modes = [0]
            for mode in modes:
                before = len(bist_results)
                cmd = fpga.run_ddr3_bist(mode=mode, words=args.bist_words, timeout_ms=1000)
                print(f"ddr3_bist_cmd: mode={mode} success={cmd.success} message={cmd.message}", flush=True)
                deadline_bist = time.monotonic() + max(0.2, float(args.bist_timeout))
                while time.monotonic() < deadline_bist and len(bist_results) <= before:
                    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
                    time.sleep(0.01)
                if len(bist_results) <= before:
                    print(f"ddr3_bist: mode={mode} timeout waiting for PKT_DDR3_BIST", flush=True)
                    return 3
                result = bist_results[-1]
                print(f"ddr3_bist: {_bist_summary(result)}", flush=True)
                if not result.passed or result.failed or result.error_count != 0:
                    print("ddr3_bist: FAIL, refusing to continue histogram test", flush=True)
                    return 3

        cfg = fpga.configure_gpx2_custom(
            GPX2_DEFAULT_CONFIG_BYTES,
            timeout_ms=1000,
            profile=GPX2_DEFAULT_PROFILE,
            sequence_mode=GPX2_DEFAULT_SEQUENCE,
        )
        print(f"gpx2_config: success={cfg.success} message={cfg.message}", flush=True)
        hist_cfg = fpga.configure_tdc_histogram(
            enable=True,
            clear=True,
            start_channel=start_channel,
            stop_channel=stop_channel,
            bin_width_raw=args.bin_width_raw,
            bin_count=total_bins,
            offset_raw=args.offset_raw,
            reference_cleanup_raw=reference_cleanup_raw,
            tdc_deadtime_raw=tdc_deadtime_raw,
            timeout_ms=1000,
        )
        print(f"hist_config: success={hist_cfg.success} message={hist_cfg.message}", flush=True)

        deadline = time.monotonic() + max(0.1, float(args.duration))
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.02)

        if args.disable_before_readout:
            hist_stop = fpga.configure_tdc_histogram(
                enable=False,
                clear=False,
                start_channel=start_channel,
                stop_channel=stop_channel,
                bin_width_raw=args.bin_width_raw,
                bin_count=total_bins,
                offset_raw=args.offset_raw,
                reference_cleanup_raw=reference_cleanup_raw,
                tdc_deadtime_raw=tdc_deadtime_raw,
                timeout_ms=1000,
            )
            print(
                f"hist_disable_before_readout: success={hist_stop.success} message={hist_stop.message}",
                flush=True,
            )
            settle_deadline = time.monotonic() + 0.1
            while time.monotonic() < settle_deadline:
                app.processEvents(QtCore.QEventLoop.AllEvents, 20)
                time.sleep(0.005)

        expected_chunks = max(1, (total_bins + 255) // 256)
        missing_chunks: list[tuple[int, int]] = []

        def wait_for_chunk(hist_id: int, chunk_index: int) -> bool:
            wait_deadline = time.monotonic() + max(0.05, float(args.readout_interval))
            while time.monotonic() < wait_deadline:
                app.processEvents(QtCore.QEventLoop.AllEvents, 20)
                if chunk_index in seen_chunks[hist_id]:
                    return True
                time.sleep(0.005)
            return chunk_index in seen_chunks[hist_id]

        if args.chunked_readout:
            for hist_id in requested_hist_ids:
                mask = 1 << hist_id
                for chunk_index in range(expected_chunks):
                    received = chunk_index in seen_chunks[hist_id]
                    for _attempt in range(max(1, int(args.chunk_retries))):
                        if received:
                            break
                        fpga.request_tdc_histogram_readout(mask, timeout_ms=1000, chunk_index=chunk_index)
                        received = wait_for_chunk(hist_id, chunk_index)
                    if not received:
                        missing_chunks.append((hist_id, chunk_index))
                        if args.fail_fast_missing:
                            break
                if args.fail_fast_missing and missing_chunks:
                    break
        else:
            fpga.request_tdc_histogram_readout(int(args.hist_mask) & 0x3, timeout_ms=1000, chunk_index=None)
            snapshot_deadline = time.monotonic() + max(5.0, expected_chunks * max(0.05, float(args.readout_interval)))
            while time.monotonic() < snapshot_deadline:
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)
                if all(len(seen_chunks[hist_id]) >= expected_chunks for hist_id in requested_hist_ids):
                    break
                time.sleep(0.01)
            for hist_id in requested_hist_ids:
                for chunk_index in range(expected_chunks):
                    if chunk_index not in seen_chunks[hist_id]:
                        missing_chunks.append((hist_id, chunk_index))
                        if args.fail_fast_missing:
                            break
                if args.fail_fast_missing and missing_chunks:
                    break
        drain_deadline = time.monotonic() + 1.0
        while time.monotonic() < drain_deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.02)
    finally:
        try:
            fpga.configure_tdc_histogram(
                enable=False,
                clear=False,
                start_channel=start_channel,
                stop_channel=stop_channel,
                bin_width_raw=args.bin_width_raw,
                bin_count=total_bins,
                offset_raw=args.offset_raw,
                reference_cleanup_raw=reference_cleanup_raw,
                tdc_deadtime_raw=tdc_deadtime_raw,
                timeout_ms=1000,
            )
        finally:
            service.close_device()
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    print(
        "summary: "
        f"chunks commercial/diagnostic={len(seen_chunks[0])}/{len(seen_chunks[1])} "
        f"packets commercial/diagnostic={chunk_counts[0]}/{chunk_counts[1]} "
        f"rx_bytes={service.runtime_stats.rx_bytes} tx_bytes={service.runtime_stats.tx_bytes} "
        f"packets={service.runtime_stats.packet_count}",
        flush=True,
    )
    if missing_chunks:
        preview = ", ".join(f"{hist_id}:{chunk}" for hist_id, chunk in missing_chunks[:16])
        if len(missing_chunks) > 16:
            preview += ", ..."
        print(f"missing_chunks={len(missing_chunks)} [{preview}]", flush=True)
    else:
        print("missing_chunks=0", flush=True)
    type_desc = " ".join(
        f"0x{pkt_type:02X}:{count}" for pkt_type, count in sorted(packet_types.items())
    )
    item_desc = " ".join(
        f"0x{pkt_type:02X}/items={items}:{count}"
        for (pkt_type, items), count in sorted(packet_items.items())
    )
    print(f"packet_types: {type_desc or '-'}", flush=True)
    print(f"packet_items: {item_desc or '-'}", flush=True)
    if bist_results:
        print(f"ddr3_bist_last: {_bist_summary(bist_results[-1])}", flush=True)
    for hist_id, name in ((0, "commercial"), (1, "diagnostic")):
        chunk = last_chunks.get(hist_id)
        counters = "-"
        if chunk is not None:
            ddr_flags = decode_tdc_hist_status_flags(chunk.status_flags)
            ddr_health = (
                f"ddr_backend={int(ddr_flags['ddr_backend'])} "
                f"mig_cal={int(ddr_flags['ddr_calibrated'])} "
                f"ring_ovf={int(ddr_flags['ddr_ring_overflow'])} "
                f"storage_stall={int(ddr_flags['ddr_storage_stall'])} "
                f"cdc_ovf={int(ddr_flags['ddr_cdc_req_overflow'] or ddr_flags['ddr_cdc_rsp_overflow'])}"
            )
            counters = (
                f"starts={chunk.start_count} stops={chunk.stop_count} accepted={chunk.accepted_count} "
                f"no_start={chunk.no_start_count} out={chunk.out_of_window_count} "
                f"suppressed={chunk.first_stop_suppressed_count} "
                f"ref_cleanup={chunk.reference_cleanup_count} tdc_deadtime={chunk.tdc_deadtime_filtered_count} "
                f"last_dt_ns={chunk.last_dt_raw * 0.008:.3f} "
                f"status=0x{chunk.status_flags:08X}({describe_tdc_hist_status_flags(chunk.status_flags)}) "
                f"{ddr_health}"
            )
        print(f"{name}: {_peak_summary(histograms[hist_id], args.bin_width_raw, args.offset_raw)} | {counters}", flush=True)

    if args.export_dir:
        export_dir = Path(args.export_dir)
        _write_histogram_csv(export_dir / "hist_fpga_commercial.csv", histograms[0], args.bin_width_raw, args.offset_raw)
        _write_histogram_csv(export_dir / "hist_fpga_diagnostic.csv", histograms[1], args.bin_width_raw, args.offset_raw)
        print(f"export_dir={export_dir}", flush=True)

    return 0 if not missing_chunks and all(len(seen_chunks[hist_id]) > 0 for hist_id in requested_hist_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
