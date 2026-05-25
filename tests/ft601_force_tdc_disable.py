"""Force-disable FPGA TDC test upload without draining the uplink.

This is useful when the FPGA is already streaming data and the host wants to
give the FT601 RX command path a quiet window to consume CMD_TDC_TEST_UPLOAD=0.
"""

from __future__ import annotations

import argparse
import time

from PyQt5 import QtCore

from app import protocol
from app.protocol import CommandEncoder
from app.usb_link import D3XXError, DeviceService


def main() -> int:
    parser = argparse.ArgumentParser(description="Force-disable TDC test upload")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    encoder = CommandEncoder()
    frame = encoder.encode_tdc_test_upload(False, 0)

    service.log_message.connect(lambda message: print(message, flush=True))
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    try:
        service.open_device(
            device_index=args.index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=4096,
            read_enabled=False,
        )
    except D3XXError as exc:
        print(f"open failed: {exc}", flush=True)
        return 1

    try:
        for idx in range(max(1, int(args.count))):
            service.send_command_sync(
                protocol.CMD_TDC_TEST_UPLOAD,
                frame,
                timeout_ms=1000,
                debug_details=f"force disable {idx}",
            )
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(max(0.0, float(args.delay)))
        time.sleep(0.5)
    finally:
        service.close_device()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
