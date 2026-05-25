"""Hardware smoke test for analog/TEC Flash persistence command.

Run with:
    python -m app.tests.ft601_flash_analog_save_probe
"""

from __future__ import annotations

import argparse
import sys
import time

from PyQt5 import QtCore

from app.fpga_control import AnalogControlService, FlashService, TemperatureControlService
from app.protocol import CMD_FLASH_SAVE, CMD_FLASH_SAVE_ANALOG, LegacyAnalogCodec, PacketParser, PKT_STATUS
from app.usb_link import D3XXError, DeviceService


def _status_summary(status) -> str:
    if status is None:
        return "status=None"
    decoded = DeviceService.decode_status_flags(status.flags)
    return (
        f"seq={status.header.seq} flags=0x{status.flags:04X} "
        f"flash_busy={int(decoded.flash_busy)} flash_error={int(decoded.flash_error)} "
        f"uptime_s={status.uptime_seconds} temp_raw={status.temp_avg_raw}"
    )


def _read_status_packets(service: DeviceService, pkt_parser: PacketParser, timeout_s: float, read_size: int, read_pipe: int):
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    statuses = []
    while time.monotonic() < deadline:
        try:
            block = service.device.read_block(int(read_size), int(read_pipe))
        except D3XXError:
            continue
        if not block:
            continue
        for packet in pkt_parser.feed(block):
            if packet.header.pkt_type == PKT_STATUS and packet.status is not None:
                statuses.append(packet.status)
    return statuses


def _wait_flash_idle(
    service: DeviceService,
    pkt_parser: PacketParser,
    start_seq: int | None,
    timeout_s: float,
    read_size: int,
    read_pipe: int,
) -> tuple[bool, str]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    seen_new_status = start_seq is None
    seen_busy = False
    last_status = None

    while time.monotonic() < deadline:
        statuses = _read_status_packets(service, pkt_parser, 0.2, read_size, read_pipe)
        for status in statuses:
            last_status = status
            seq = int(status.header.seq)
            if start_seq is None or seq != start_seq:
                seen_new_status = True
            decoded = DeviceService.decode_status_flags(status.flags)
            if decoded.flash_error:
                return False, f"flash_error asserted: {_status_summary(status)}"
            if decoded.flash_busy:
                seen_busy = True
            elif seen_busy or seen_new_status:
                return True, f"flash idle: {_status_summary(status)}"

    return False, f"timeout waiting flash idle: {_status_summary(last_status)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply analog/TEC values and save them to Flash over FT601.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--read-pipe", type=lambda value: int(value, 0), default=0x82)
    parser.add_argument("--write-pipe", type=lambda value: int(value, 0), default=0x02)
    parser.add_argument("--read-block-size", type=int, default=4096)
    parser.add_argument("--temp-c", type=float, default=24.5)
    parser.add_argument("--laser-threshold-mv", type=float, default=1100.0)
    parser.add_argument("--pixel-threshold-mv", type=float, default=1200.0)
    parser.add_argument("--avalanche-threshold-mv", type=float, default=130.0)
    parser.add_argument("--bias-v", type=float, default=45.5)
    parser.add_argument("--command-timeout-ms", type=int, default=1500)
    parser.add_argument("--flash-timeout-s", type=float, default=30.0)
    parser.add_argument("--save-mode", choices=("analog", "full"), default="analog")
    parser.add_argument("--skip-load-check", action="store_true")
    args = parser.parse_args(argv)

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    del app

    codec = LegacyAnalogCodec()
    pkt_parser = PacketParser(decode_tdc_events=False)
    service = DeviceService()
    service.log_message.connect(lambda message: print(f"[log] {message}", flush=True))
    service.device_state_changed.connect(
        lambda ok, message: print(f"[state] {'OK' if ok else 'FAIL'}: {message}", flush=True)
    )

    try:
        print("Opening FT601...", flush=True)
        service.open_device(
            args.device_index,
            read_pipe=args.read_pipe,
            write_pipe=args.write_pipe,
            read_block_size=args.read_block_size,
            read_enabled=False,
        )
        initial_statuses = _read_status_packets(service, pkt_parser, 3.0, args.read_block_size, args.read_pipe)
        initial_status = initial_statuses[-1] if initial_statuses else None
        if initial_status is not None:
            print(f"Initial {_status_summary(initial_status)}", flush=True)
        else:
            print("No status packet within 3s; continuing with command write smoke test.", flush=True)

        temp = TemperatureControlService(service, codec)
        analog = AnalogControlService(service, codec)
        flash = FlashService(service)

        result = temp.set_target_temperature_c(args.temp_c, args.command_timeout_ms)
        print(f"Temperature command: success={result.success} message={result.message}", flush=True)
        if not result.success:
            return 2

        result = analog.set_outputs(
            args.laser_threshold_mv,
            args.pixel_threshold_mv,
            args.avalanche_threshold_mv,
            args.bias_v,
            args.command_timeout_ms,
        )
        print(f"Analog command: success={result.success} message={result.message}", flush=True)
        if not result.success:
            return 3

        start_seq = None if initial_status is None else int(initial_status.header.seq)
        if args.save_mode == "full":
            save_cmd_id = CMD_FLASH_SAVE
            result = flash.flash_save(args.command_timeout_ms)
        else:
            save_cmd_id = CMD_FLASH_SAVE_ANALOG
            result = flash.flash_save_analog(args.command_timeout_ms)
        print(f"Flash {args.save_mode} save command: success={result.success} cmd=0x{save_cmd_id:02X} message={result.message}", flush=True)
        if not result.success:
            return 4

        ok, message = _wait_flash_idle(
            service,
            pkt_parser,
            start_seq,
            args.flash_timeout_s,
            args.read_block_size,
            args.read_pipe,
        )
        print(message, flush=True)
        if not ok:
            return 5

        if not args.skip_load_check:
            statuses = _read_status_packets(service, pkt_parser, 1.0, args.read_block_size, args.read_pipe)
            load_start_seq = None if not statuses else int(statuses[-1].header.seq)
            result = flash.flash_load(args.command_timeout_ms)
            print(f"Flash load check command: success={result.success} message={result.message}", flush=True)
            if not result.success:
                return 6
            ok, message = _wait_flash_idle(
                service,
                pkt_parser,
                load_start_seq,
                args.flash_timeout_s,
                args.read_block_size,
                args.read_pipe,
            )
            print(message, flush=True)
            if not ok:
                return 7

        print(f"TEST PASSED: {args.save_mode} Flash save and load check completed without flash error.", flush=True)
        return 0
    except D3XXError as exc:
        print(f"FT601 ERROR: {exc}", flush=True)
        return 10
    finally:
        service.close_device()


if __name__ == "__main__":
    raise SystemExit(main())
