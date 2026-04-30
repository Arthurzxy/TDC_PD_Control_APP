from __future__ import annotations

import argparse
import time

from PyQt5 import QtCore

from app.usb_link import DeviceService, PyD3XXDevice


def main() -> int:
    parser = argparse.ArgumentParser(description="PyD3XX FT601 smoke test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--timeout-ms", type=int, default=100)
    parser.add_argument("--read-size", type=int, default=16 * 1024)
    parser.add_argument("--device-service", action="store_true")
    args = parser.parse_args()

    if args.device_service:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        service = DeviceService()
        service.log_message.connect(lambda message: print("log:", message, flush=True))
        service.device_state_changed.connect(
            lambda connected, message: print("state:", connected, message, flush=True)
        )
        print("service-open-start", flush=True)
        service.open_device(
            device_index=args.index,
            read_pipe=args.read_pipe,
            write_pipe=0x02,
            read_block_size=args.read_size,
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.05)
        print("service-close-start", flush=True)
        service.close_device()
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        print("service-closed", flush=True)
        return 0

    device = PyD3XXDevice()
    devices = device.enumerate_devices()
    print("devices:", [(item.index, item.description, item.serial) for item in devices], flush=True)
    if not devices:
        return 2

    print("open-start", flush=True)
    device.open_device(args.index)
    try:
        try:
            interface_count = max(1, int(device.get_configuration_descriptor()["num_interfaces"]))
        except Exception:
            interface_count = 2
        for interface_index in range(interface_count):
            try:
                print(f"pipes-if{interface_index}:", device.list_pipe_information(interface_index), flush=True)
            except Exception as exc:
                print(f"pipes-if{interface_index}: {exc}", flush=True)
        device.set_pipe_timeout(args.read_pipe, args.timeout_ms)
        try:
            block = device.read_block(args.read_size, args.read_pipe)
            print("read-returned:", len(block), flush=True)
        except Exception as exc:
            print("read-exc:", exc, flush=True)
    finally:
        device.close_device()
        print("closed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
