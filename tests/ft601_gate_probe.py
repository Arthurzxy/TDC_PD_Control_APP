"""Probe the external-trigger gate generator through FT601 status counters.

Example:
    python -m app.tests.ft601_gate_probe --duration 1.0 --divider 10
"""

from __future__ import annotations

import argparse
import sys
import time

from PyQt5 import QtCore

from app.fpga_control import AnalogControlService, FpgaControlService
from app.protocol import LegacyAnalogCodec
from app.usb_link import D3XXError, DeviceService


def _split_ns(value_ns: int) -> tuple[int, int]:
    value = max(0, int(value_ns))
    return min(15, value // 10), min(31, value % 10)


def _delta(after: int, before: int) -> int:
    return (int(after) - int(before)) & 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description="FT601 gate trigger/divider probe")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--read-pipe", type=int, default=0x82)
    parser.add_argument("--write-pipe", type=int, default=0x02)
    parser.add_argument("--read-size", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--divider", type=int, default=10)
    parser.add_argument("--direct-delay-ns", type=int, default=5)
    parser.add_argument("--direct-width-ns", type=int, default=20)
    parser.add_argument("--divided-delay-ns", type=int, default=15)
    parser.add_argument("--divided-width-ns", type=int, default=20)
    parser.add_argument("--skip-threshold-setup", action="store_true")
    parser.add_argument("--laser-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--pixel-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--avalanche-threshold-mv", type=float, default=1000.0)
    parser.add_argument("--bias-v", type=float, default=50.0)
    args = parser.parse_args()

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    service = DeviceService()
    fpga = FpgaControlService(service)
    analog = AnalogControlService(service, LegacyAnalogCodec())
    latest_status = {"value": None}

    def on_status(status) -> None:
        latest_status["value"] = status

    def wait_seconds(seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)

    def wait_status(timeout_s: float = 2.0):
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline and latest_status["value"] is None:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        return latest_status["value"]

    def wait_next_status(after_seq: int | None = None, timeout_s: float = 3.0):
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            status = latest_status["value"]
            if status is not None:
                seq = int(status.header.seq)
                if after_seq is None or seq != int(after_seq):
                    return status
            time.sleep(0.01)
        return latest_status["value"]

    def wait_status_after(target_time: float, after_seq: int, timeout_s: float = 4.0):
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            status = latest_status["value"]
            if status is not None and int(status.header.seq) != int(after_seq) and time.monotonic() >= target_time:
                return status
            time.sleep(0.01)
        return latest_status["value"]

    def counters(status) -> tuple[int, int, int, int]:
        return (
            int(status.gate_trigger_count),
            int(status.gate_direct_pulse_count),
            int(status.gate_divided_pulse_count),
            int(status.gate_output_pulse_count),
        )

    def print_case(name: str, before, after, divider: int) -> None:
        b = counters(before)
        a = counters(after)
        d = tuple(_delta(x, y) for x, y in zip(a, b))
        trig, direct, divided, output = d
        expected_div = (trig / max(1, int(divider))) if trig else 0.0
        print(
            f"{name}: trigger_delta={trig} direct_delta={direct} "
            f"divided_delta={divided} output_delta={output} "
            f"expected_divided~={expected_div:.1f}",
            flush=True,
        )

    service.status_received.connect(on_status)
    service.device_state_changed.connect(lambda ok, msg: print(f"state={ok} {msg}", flush=True))

    try:
        service.open_device(
            device_index=args.index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=args.read_size,
            read_enabled=True,
        )
    except D3XXError as exc:
        print(f"open failed: {exc}. Close the GUI/app using FT601 first.", flush=True)
        return 1

    try:
        if not args.skip_threshold_setup:
            result = analog.set_outputs(
                args.laser_threshold_mv,
                args.pixel_threshold_mv,
                args.avalanche_threshold_mv,
                args.bias_v,
                timeout_ms=1000,
            )
            print(f"analog_thresholds: success={result.success} message={result.message}", flush=True)

        direct_delay = _split_ns(args.direct_delay_ns)
        direct_width = _split_ns(args.direct_width_ns)
        divided_delay = _split_ns(args.divided_delay_ns)
        divided_width = _split_ns(args.divided_width_ns)

        fpga.set_gate_signal(2, direct_delay[0], direct_delay[1], direct_width[0], direct_width[1])
        fpga.set_gate_signal(3, divided_delay[0], divided_delay[1], divided_width[0], divided_width[1])
        fpga.set_gate_div(args.divider)
        fpga.set_gate_enable(False, False, False)
        wait_seconds(0.1)

        status0 = wait_status()
        if status0 is None:
            print("status timeout: no status packet received", flush=True)
            return 2
        decoded0 = DeviceService.decode_status_flags(int(status0.flags))
        print(
            "initial_status: "
            f"flags=0x{int(status0.flags):04X} gate_clk_locked={int(decoded0.gate_clk_locked)} "
            f"gpx2_lclk_locked={int(decoded0.gpx2_lclk_locked)} "
            f"gate_counts={counters(status0)}",
            flush=True,
        )

        cases = (
            ("direct_only", True, False, 1),
            ("divided_only", False, True, max(1, int(args.divider))),
            ("direct_plus_divided", True, True, max(1, int(args.divider))),
        )
        for name, direct_en, divided_en, divider in cases:
            last_seq = int(latest_status["value"].header.seq) if latest_status["value"] is not None else None
            fpga.set_gate_div(divider)
            fpga.set_gate_enable(direct_en, divided_en, False)
            before = wait_next_status(last_seq, timeout_s=3.0)
            target_time = time.monotonic() + max(0.1, float(args.duration))
            before_seq = int(before.header.seq) if before is not None else -1
            after = wait_status_after(target_time, before_seq, timeout_s=max(4.0, float(args.duration) + 3.0))
            fpga.set_gate_enable(False, False, False)
            wait_seconds(0.1)
            if before is None or after is None:
                print(f"{name}: status missing", flush=True)
                return 2
            print_case(name, before, after, divider)

        return 0
    finally:
        try:
            try:
                fpga.set_gate_enable(False, False, False)
            except D3XXError as exc:
                print(f"gate_disable_cleanup: {exc}", flush=True)
        finally:
            service.close_device()
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)


if __name__ == "__main__":
    sys.exit(main())
