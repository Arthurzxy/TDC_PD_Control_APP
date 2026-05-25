"""Direct FT601 read benchmark for TDC upload.

This bypasses the DeviceService RX worker, Qt packet signals, and packet parser.
It only sends the upload command, then loops on FT_ReadPipe and counts bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import threading
import time

import numpy as np
from PyQt5 import QtCore

from app.data_processing import TdcTestHistogramBuilder
from app.models import TdcTestSettings
from app import protocol
from app.protocol import CommandEncoder, PacketParser
from app.usb_link import D3XXError, DeviceService


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct FT601 read benchmark")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-size", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--source", choices=["synthetic", "gpx2"], default="synthetic")
    parser.add_argument("--start", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--stop", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--parse", action="store_true", help="Parse uplink packets and report TDC histogram stats.")
    parser.add_argument("--threaded", action="store_true", help="Run the read loop in a background Python thread.")
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    encoder = CommandEncoder()
    service.log_message.connect(lambda message: print(message, flush=True))
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    channel_mask = (1 << (args.start - 1)) | (1 << (args.stop - 1))
    enable_frame = encoder.encode_tdc_test_upload(
        True,
        channel_mask,
        synthetic=args.source == "synthetic",
        extended_timestamp_raw=args.source == "gpx2",
    )
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
        print(f"open failed: {exc}", flush=True)
        return 1

    total = 0
    reads = 0
    empty_reads = 0
    hist = Counter()
    errors = Counter()
    packet_count = 0
    tdc_raw_packets = 0
    tdc_raw_events = 0
    raw_channel_counts = [0, 0, 0, 0]
    parser_state = PacketParser(decode_tdc_events=False)
    hist_builder = TdcTestHistogramBuilder(
        TdcTestSettings(
            enabled=True,
            start_channel=args.start - 1,
            stop_channel=args.stop - 1,
            source=args.source,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=300000,
            acquisition_time_s=args.duration,
        )
    )

    def consume_block(block: bytes) -> None:
        nonlocal total, reads, empty_reads, packet_count, tdc_raw_packets, tdc_raw_events
        reads += 1
        size = len(block)
        if size == 0:
            empty_reads += 1
            return
        total += size
        hist[size] += 1
        if not args.parse:
            return
        for packet in parser_state.feed(block):
            packet_count += 1
            if packet.tdc_event_batch is None:
                continue
            tdc_raw_packets += 1
            batch = packet.tdc_event_batch
            batch_len = len(batch)
            tdc_raw_events += batch_len
            hist_builder.process_batch(batch)
            channels = np.asarray(batch.channels, dtype=np.int64)
            counts = np.bincount(channels[(channels >= 0) & (channels < 4)], minlength=4)
            for idx in range(4):
                raw_channel_counts[idx] += int(counts[idx])

    def read_until(deadline: float) -> None:
        while time.monotonic() < deadline:
            try:
                block = service.device.read_block(int(args.read_size), int(args.read_pipe))
            except D3XXError as exc:
                errors[str(exc)] += 1
                continue
            consume_block(block)

    try:
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="direct probe pre-disable",
        )
        if args.source == "gpx2":
            service.send_command_sync(
            protocol.CMD_GPX2_CFG_CUSTOM,
            encoder.encode_gpx2_custom_config(protocol.GPX2_DEFAULT_CONFIG_BYTES),
            timeout_ms=1000,
            debug_details="direct probe GPX2 max high-resolution config",
        )
            time.sleep(0.5)

        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            enable_frame,
            timeout_ms=1000,
            debug_details=f"direct probe start {args.source}",
        )

        deadline = time.monotonic() + max(0.1, float(args.duration))
        if args.threaded:
            reader = threading.Thread(target=read_until, args=(deadline,), name="ft601-direct-reader")
            reader.start()
            reader.join(max(0.1, float(args.duration)) + 1.0)
        else:
            read_until(deadline)
        service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            disable_frame,
            timeout_ms=1000,
            debug_details="direct probe stop",
        )
    finally:
        service.close_device()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    elapsed = max(0.001, float(args.duration))
    print(
        "direct_summary: "
        f"source={args.source} bytes={total} rate={total / elapsed / 1000.0:.1f} kB/s "
        f"reads={reads} empty_reads={empty_reads} "
        f"chunk_hist={','.join(f'{k}:{v}' for k, v in hist.most_common(8)) or '-'} "
        f"errors={sum(errors.values())}",
        flush=True,
    )
    for message, count in errors.most_common(3):
        print(f"direct_error: count={count} {message}", flush=True)
    if args.parse:
        snapshot = hist_builder.snapshot()
        tdc_hist = np.asarray(snapshot.histograms["tdc_test"], dtype=np.uint32)
        projection = snapshot.image_projection[0]
        peak_bin = int(tdc_hist.argmax()) if tdc_hist.size else -1
        peak_count = int(tdc_hist[peak_bin]) if peak_bin >= 0 else 0
        peak_ns = (peak_bin * hist_builder.settings.bin_width_raw + hist_builder.settings.bin_offset) * 0.008
        print(
            "direct_parse: "
            f"packets={packet_count} tdc_raw_packets={tdc_raw_packets} "
            f"tdc_raw_events={tdc_raw_events} "
            f"raw_ch={raw_channel_counts[0]}/{raw_channel_counts[1]}/"
            f"{raw_channel_counts[2]}/{raw_channel_counts[3]} "
            f"sync={projection[1]} photon={projection[2]} accepted_pairs={projection[0]} "
            f"paired_photons={projection[14] if len(projection) > 14 else 0} "
            f"peak_bin={peak_bin} peak_ns={peak_ns:.3f} peak_count={peak_count} "
            f"zero/neg/out/no_pair={projection[6]}/{projection[7]}/{projection[8]}/{projection[13]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
