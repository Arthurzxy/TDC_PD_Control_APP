"""Direct FT601 status packet probe.

Run with:
    python -m app.tests.ft601_status_probe
"""

from __future__ import annotations

import argparse
from collections import Counter
import time

from PyQt5 import QtCore

from app.protocol import CommandEncoder, PacketParser, PKT_STATUS
from app.usb_link import D3XXError, DeviceService


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and parse FT601 uplink packets, focusing on status.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-size", type=int, default=4096)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--enable-synthetic-upload", action="store_true")
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    del app

    service = DeviceService()
    service.log_message.connect(lambda message: print(message, flush=True))
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))
    pkt_parser = PacketParser(decode_tdc_events=False)
    encoder = CommandEncoder()

    try:
        service.open_device(
            device_index=args.device_index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=args.read_size,
            read_enabled=False,
        )
        if args.enable_synthetic_upload:
            service.send_command_sync(
                0x26,
                encoder.encode_tdc_test_upload(True, (1 << 1) | (1 << 2), synthetic=True),
                timeout_ms=1000,
                debug_details="status probe synthetic upload",
            )

        deadline = time.monotonic() + max(0.1, float(args.duration))
        bytes_total = 0
        reads = 0
        errors = Counter()
        pkt_types = Counter()
        status_count = 0
        last_status = None

        while time.monotonic() < deadline:
            try:
                block = service.device.read_block(int(args.read_size), int(args.read_pipe))
            except D3XXError as exc:
                errors[str(exc)] += 1
                continue
            reads += 1
            if not block:
                continue
            bytes_total += len(block)
            for packet in pkt_parser.feed(block):
                pkt_types[packet.header.pkt_type] += 1
                if packet.header.pkt_type == PKT_STATUS and packet.status is not None:
                    status_count += 1
                    last_status = packet.status

        if args.enable_synthetic_upload:
            service.send_command_sync(
                0x26,
                encoder.encode_tdc_test_upload(False, 0),
                timeout_ms=1000,
                debug_details="status probe stop synthetic upload",
            )

        print(
            "status_probe_summary: "
            f"bytes={bytes_total} reads={reads} status_count={status_count} "
            f"pkt_types={dict(pkt_types)} errors={sum(errors.values())}",
            flush=True,
        )
        for message, count in errors.most_common(3):
            print(f"status_probe_error: count={count} {message}", flush=True)
        if last_status is not None:
            decoded = DeviceService.decode_status_flags(last_status.flags)
            print(
                "last_status: "
                f"seq={last_status.header.seq} flags=0x{last_status.flags:04X} "
                f"flash_busy={int(decoded.flash_busy)} flash_error={int(decoded.flash_error)} "
                f"uptime_s={last_status.uptime_seconds} temp_raw={last_status.temp_avg_raw} "
                f"counter_1s={last_status.counter_1s}",
                flush=True,
            )
        return 0 if status_count else 2
    finally:
        service.close_device()


if __name__ == "__main__":
    raise SystemExit(main())
