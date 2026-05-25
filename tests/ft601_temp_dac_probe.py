from __future__ import annotations

import argparse
import sys
import time

from PyQt5 import QtCore

from app.fpga_control import AnalogControlService, TemperatureControlService
from app.protocol import LegacyAnalogCodec
from app.usb_link import DeviceService


def _parse_temps(text: str) -> list[float]:
    temps: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            temps.append(float(item))
    return temps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send TEC/DAC8881 temperature commands over FT601.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--read-enabled", action="store_true")
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument("--laser-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--pixel-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--avalanche-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--bias-v", type=float, default=0.0)
    parser.add_argument("--temp-c", type=float, default=25.0)
    parser.add_argument(
        "--pulse-temps",
        default="",
        help="Comma separated temperatures to send in sequence, e.g. 20,25. Overrides --temp-c.",
    )
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    del app

    codec = LegacyAnalogCodec()
    service = DeviceService()
    service.log_message.connect(lambda message: print(message, flush=True))

    temps = _parse_temps(args.pulse_temps) if args.pulse_temps else [float(args.temp_c)]
    if not temps:
        raise ValueError("No temperature target specified")

    try:
        service.open_device(
            args.device_index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_enabled=bool(args.read_enabled),
        )
        analog = AnalogControlService(service, codec)
        temp = TemperatureControlService(service, codec)

        if not args.skip_thresholds:
            result = analog.set_outputs(
                args.laser_threshold_mv,
                args.pixel_threshold_mv,
                args.avalanche_threshold_mv,
                args.bias_v,
                timeout_ms=1000,
            )
            print(f"thresholds: success={result.success} message={result.message}", flush=True)

        for index in range(max(1, int(args.repeat))):
            for temp_c in temps:
                code = codec.temperature_c_to_code(temp_c)
                result = temp.set_target_temperature_c(temp_c, timeout_ms=1000)
                print(
                    f"temperature[{index}]: temp_c={temp_c:.3f} code={code} "
                    f"0x{code:04X} success={result.success} message={result.message}",
                    flush=True,
                )
                if args.interval_ms > 0:
                    time.sleep(args.interval_ms / 1000.0)
        return 0
    finally:
        service.close_device()


if __name__ == "__main__":
    raise SystemExit(main())
