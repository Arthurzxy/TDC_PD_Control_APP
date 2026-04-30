from __future__ import annotations

import unittest

from app.protocol import CommandEncoder, PacketParser


def pack_u32_le(word: int) -> bytes:
    return int(word & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False)


class ProtocolTests(unittest.TestCase):
    def test_encode_temperature_command(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_set_temperature_code(0x1234)
        self.assertEqual(frame, pack_u32_le(0xBB040001) + pack_u32_le(0x00001234))

    def test_encode_ad5686_command(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_set_ad5686(1, 2, 3, 4)
        expected = pack_u32_le(0xBB010002) + pack_u32_le(0x00010002) + pack_u32_le(0x00030004)
        self.assertEqual(frame, expected)

    def test_parse_status_packet(self) -> None:
        parser = PacketParser()
        words = [
            0xA5020104,
            0x00010006,
            0x000100A5,
            0x00001000,
            0x000000A5,
            0x0000000F,
            0x00000064,
            0x00000020,
            0x00000030,
            0x00000040,
        ]
        packet = b"".join(pack_u32_le(word) for word in words)
        parsed = parser.feed(packet)
        self.assertEqual(len(parsed), 1)
        status = parsed[0].status
        self.assertIsNotNone(status)
        self.assertEqual(status.flags, 0x00A5)
        self.assertEqual(status.uptime_seconds, 0x0F)
        self.assertEqual(status.temp_avg_raw, 0x64)
        self.assertEqual(status.counter_1s, 0x20)
        self.assertEqual(status.tdc_drop_count, 0x30)
        self.assertEqual(status.usb_drop_count, 0x40)

    def test_parse_tdc_raw_packet(self) -> None:
        parser = PacketParser()
        event_record = (0 << 60) | (1 << 58) | (0 << 56) | (0x123456 << 32) | (0x54321 << 12)
        low = event_record & 0xFFFFFFFF
        high = (event_record >> 32) & 0xFFFFFFFF
        words = [
            0xA5010104,
            0x00020002,
            0x00010000,
            0x000000AA,
            low,
            high,
        ]
        packet = b"".join(pack_u32_le(word) for word in words)
        parsed = parser.feed(packet)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(parsed[0].tdc_events), 1)
        event = parsed[0].tdc_events[0]
        self.assertEqual(event.channel, 1)
        self.assertEqual(event.refid, 0x123456)
        self.assertEqual(event.tstop, 0x54321)

    def test_parse_photon_event_packet(self) -> None:
        parser = PacketParser()
        words = [
            0xA5040104,
            0x00030005,
            0x00010000,
            0x000000BB,
            0xA55AF00D,
            0x01000002,
            0x0003004E,
            0x00000271,
            0x12345678,
        ]
        packet = b"".join(pack_u32_le(word) for word in words)
        parsed = parser.feed(packet)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(parsed[0].photon_events), 1)
        event = parsed[0].photon_events[0]
        self.assertEqual(event.packet_type, 0x01)
        self.assertEqual(event.line_id, 2)
        self.assertEqual(event.pixel_id, 3)
        self.assertEqual(event.bin_index, 78)
        self.assertEqual(event.dt_8ps, 625)
        self.assertEqual(event.detector_timestamp_low, 0x12345678)

    def test_parse_multi_photon_event_packet_from_latest_fpga(self) -> None:
        parser = PacketParser()
        words = [
            0xA5040104,
            0x0001000F,
            0x00030000,
            0x00000010,
            0xA55AF00D,
            0x01000001,
            0x0002004E,
            0x00000271,
            0x00000659,
            0xA55AF00D,
            0x01000001,
            0x000300BB,
            0x000005DC,
            0x00003A98,
            0xA55AF00D,
            0x01000002,
            0x000101D4,
            0x00000EA6,
            0x00007436,
        ]
        packet = b"".join(pack_u32_le(word) for word in words)
        parsed = parser.feed(packet)
        self.assertEqual(len(parsed), 1)
        events = parsed[0].photon_events
        self.assertIsNotNone(events)
        self.assertEqual(len(events), 3)
        self.assertEqual([(e.line_id, e.pixel_id, e.bin_index, e.dt_8ps) for e in events], [
            (1, 2, 78, 625),
            (1, 3, 187, 1500),
            (2, 1, 468, 3750),
        ])


if __name__ == "__main__":
    unittest.main()
