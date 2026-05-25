"""FT601 command debug script.

Sends a known AD5686 command and verifies the FT601 read/write path.
Run with: python -m app.tests.ft601_cmd_debug
"""

from __future__ import annotations

import struct
import sys
import time

from app.protocol import CMD_AD5686, CommandEncoder, PacketParser
from app.usb_link import DeviceService, D3XXError


SYNC = 0xBB


def pack_u32_le(word: int) -> bytes:
    return struct.pack("<I", word & 0xFFFFFFFF)


def build_frame(cmd_id: int, payload_words: list[int] | None = None) -> bytes:
    payload_words = payload_words or []
    header = ((SYNC & 0xFF) << 24) | ((cmd_id & 0xFF) << 16) | (len(payload_words) & 0xF)
    frame = bytearray(pack_u32_le(header))
    for word in payload_words:
        frame.extend(pack_u32_le(word))
    return bytes(frame)


def main() -> int:
    print("=== FT601 Command Debug Tool ===\n", flush=True)

    service = DeviceService()
    service.log_message.connect(lambda msg: print(f"  [log] {msg}", flush=True))
    service.device_state_changed.connect(
        lambda ok, msg: print(f"  [state] {'OK' if ok else 'FAIL'}: {msg}", flush=True)
    )

    # --- Open device ---
    print("1. Opening FT601 device...", flush=True)
    try:
        service.open_device(
            device_index=0,
            read_pipe=0x82,
            write_pipe=0x02,
            read_block_size=4096,
            read_enabled=True,
        )
    except D3XXError as exc:
        print(f"  ERROR: {exc}", flush=True)
        return 1

    # Wait for device to stabilize and collect initial status
    print("  Waiting for status packet (2s)...", flush=True)
    got_status = service.wait_for_status_available(timeout_ms=2000)
    if got_status:
        print("  Status packet received - FT601 uplink is working.", flush=True)
    else:
        print("  No status packet - uplink may not be working.", flush=True)

    # --- Send AD5686 command with known data ---
    encoder = CommandEncoder()

    # Use distinctive, non-BB values so we can verify them on ILA
    ch1 = 0x1234
    ch2 = 0x5678
    ch3 = 0x9ABC
    ch4 = 0xDEF0

    frame = encoder.encode_set_ad5686(ch1, ch2, ch3, ch4)

    # Verify frame structure
    words = [struct.unpack_from("<I", frame, i * 4)[0] for i in range(len(frame) // 4)]
    print(f"\n2. Sending AD5686 command:", flush=True)
    print(f"   Frame bytes: {frame.hex()}", flush=True)
    print(f"   Frame words: {' '.join(f'0x{w:08X}' for w in words)}", flush=True)
    print(f"   Header: sync=0x{(words[0]>>24)&0xFF:02X} cmd=0x{(words[0]>>16)&0xFF:02X} len={words[0]&0xF}", flush=True)
    if len(words) > 1:
        print(f"   Payload word0: 0x{words[1]:08X} (ch1=0x{ch1:04X}, ch2=0x{ch2:04X})", flush=True)
    if len(words) > 2:
        print(f"   Payload word1: 0x{words[2]:08X} (ch3=0x{ch3:04X}, ch4=0x{ch4:04X})", flush=True)

    # Check if 0xBB01 would appear in AD5686 data if header leaked into payload
    print(f"\n   *** If header 0x{words[0]:08X} leaked into payload:", flush=True)
    print(f"       ad5686_data3 would be 0x{(words[0]>>16)&0xFFFF:04X} (header upper 16b)", flush=True)
    print(f"       ad5686_data4 would be 0x{words[0]&0xFFFF:04X} (header lower 16b)", flush=True)

    result = service.send_command_sync(
        CMD_AD5686,
        frame,
        timeout_ms=2000,
        debug_details=f"ch1=0x{ch1:04X} ch2=0x{ch2:04X} ch3=0x{ch3:04X} ch4=0x{ch4:04X}",
    )
    print(f"   Result: {result.message}", flush=True)

    # --- Wait for response ---
    print("\n3. Waiting for status after command (2s)...", flush=True)
    time.sleep(2)

    # Check latest status
    if service.latest_status is not None:
        s = service.latest_status
        print(f"   Status: flags=0x{s.flags:04X} uptime={s.uptime_seconds}s "
              f"temp_raw={s.temp_avg_raw} counter={s.counter_1s}", flush=True)
    else:
        print("   No status packet received.", flush=True)

    # --- Send a second command with different data ---
    ch1_b = 0xA5A5
    ch2_b = 0x5A5A
    ch3_b = 0xFF00
    ch4_b = 0x00FF

    frame2 = encoder.encode_set_ad5686(ch1_b, ch2_b, ch3_b, ch4_b)
    words2 = [struct.unpack_from("<I", frame2, i * 4)[0] for i in range(len(frame2) // 4)]

    print(f"\n4. Sending second AD5686 command with different data:", flush=True)
    print(f"   Frame words: {' '.join(f'0x{w:08X}' for w in words2)}", flush=True)

    result2 = service.send_command_sync(
        CMD_AD5686,
        frame2,
        timeout_ms=2000,
        debug_details=f"ch1=0x{ch1_b:04X} ch2=0x{ch2_b:04X} ch3=0x{ch3_b:04X} ch4=0x{ch4_b:04X}",
    )
    print(f"   Result: {result2.message}", flush=True)

    # --- Send GPX2 config using the stable manual sequence ---
    print(f"\n5. Sending GPX2 stable default config command:", flush=True)
    frame3 = encoder.encode_gpx2_config()
    words3 = [struct.unpack_from("<I", frame3, i * 4)[0] for i in range(len(frame3) // 4)]
    print(f"   Frame words: {' '.join(f'0x{w:08X}' for w in words3)}", flush=True)

    from app.protocol import CMD_GPX2_CFG
    result3 = service.send_command_sync(CMD_GPX2_CFG, frame3, timeout_ms=2000)
    print(f"   Result: {result3.message}", flush=True)

    # --- Summary ---
    print("\n=== Debug Summary ===", flush=True)
    print("If ILA shows rx_data = 0xBB010002 and never changes:", flush=True)
    print("  1. Check ft_rxf_n - if HIGH, FT601 has no data (PC not sending)", flush=True)
    print("  2. Check ft_oe_n - if HIGH, FPGA is not reading FT601", flush=True)
    print("  3. Check state machine - if stuck in state 2 (ST_EXECUTE), downstream ready is blocking", flush=True)
    print("  4. Check rx_valid - if LOW, FIFO is empty (data consumed but no new data)", flush=True)
    print("  5. Check ad5686_cmd_ready - if LOW, CDC bridge is busy", flush=True)
    print("", flush=True)
    print("If AD5686 data shows 0xBB01:", flush=True)
    print("  → Header word leaked into payload buffer", flush=True)
    print("  → Likely cause: cmd_dispatcher read the same word twice", flush=True)
    print("  → Check rx_pop timing vs state transitions", flush=True)

    # Cleanup
    print("\nClosing device...", flush=True)
    service.close_device()
    return 0


if __name__ == "__main__":
    sys.exit(main())
