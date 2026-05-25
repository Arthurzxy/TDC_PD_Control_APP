from __future__ import annotations

import unittest

from app.protocol import (
    GPX2_REGISTER_SPECS,
    GPX2_DEFAULT_CONFIG_BYTES,
    CommandEncoder,
    PacketParser,
    decode_tdc_hist_status_flags,
    describe_tdc_hist_status_flags,
    describe_gpx2_config_bytes,
    gpx2_lvds_test_config_bytes,
    gpx2_refclk_divisions,
    pack_gpx2_config_readback_words,
    sanitize_gpx2_config_bytes,
)


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

    def test_encode_flash_save_analog_command(self) -> None:
        encoder = CommandEncoder()
        self.assertEqual(encoder.encode_flash_save_analog(), pack_u32_le(0xBB320000))

    def test_encode_tdc_test_upload_synthetic(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_tdc_test_upload(True, 0x6, synthetic=True)
        self.assertEqual(frame, pack_u32_le(0xBB260001) + pack_u32_le(0x00000063))

    def test_encode_gate_divider_4096_as_zero(self) -> None:
        encoder = CommandEncoder()
        self.assertEqual(encoder.encode_gate_div(1), pack_u32_le(0xBB200001) + pack_u32_le(0x00000001))
        self.assertEqual(encoder.encode_gate_div(4096), pack_u32_le(0xBB200001) + pack_u32_le(0x00000000))

    def test_encode_sig3_gate_signal_command(self) -> None:
        encoder = CommandEncoder()
        expected_word = 1 | (2 << 4) | (3 << 9) | (4 << 12)
        self.assertEqual(
            encoder.encode_gate_signal(3, 1, 2, 3, 4),
            pack_u32_le(0xBB220001) + pack_u32_le(expected_word),
        )

    def test_encode_sig3_long_delay_command(self) -> None:
        encoder = CommandEncoder()
        expected_legacy_word = 15 | (9 << 4) | (2 << 9) | (10 << 12)
        self.assertEqual(
            encoder.encode_gate_sig3_long_delay(50000, 2, 10),
            pack_u32_le(0xBB330002) + pack_u32_le(expected_legacy_word) + pack_u32_le(50000),
        )

    def test_encode_tdc_test_upload_spi(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_tdc_test_upload(True, 0x6, spi=True)
        self.assertEqual(frame, pack_u32_le(0xBB260001) + pack_u32_le(0x00000065))

    def test_encode_tdc_test_upload_gpx2_raw_ch2_ch3(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_tdc_test_upload(True, 0x6, synthetic=False, spi=False)
        self.assertEqual(frame, pack_u32_le(0xBB260001) + pack_u32_le(0x00000061))

    def test_encode_tdc_test_upload_gpx2_raw_ext_ch2_ch3(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_tdc_test_upload(
            True,
            0x6,
            synthetic=False,
            spi=False,
            extended_timestamp_raw=True,
        )
        self.assertEqual(frame, pack_u32_le(0xBB260001) + pack_u32_le(0x00000069))

    def test_encode_tdc_hist_config_and_readout(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_tdc_hist_config(
            enable=True,
            clear=True,
            start_channel=1,
            stop_channel=2,
            bin_width_raw=5,
            bin_count=50000,
            offset_raw=10,
            reference_cleanup_raw=100000,
            tdc_deadtime_raw=12500,
        )
        expected_words = [
            0xBB270007,
            0x00000213,
            0x000030D4,
            0x00000005,
            0x0000000A,
            0x0000C350,
            0x000186A0,
            0x000030D4,
        ]
        self.assertEqual(frame, b"".join(pack_u32_le(word) for word in expected_words))
        self.assertEqual(
            encoder.encode_tdc_hist_readout(0x1),
            pack_u32_le(0xBB280001) + pack_u32_le(0x00000001),
        )

    def test_encode_ddr3_bist_command(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_ddr3_bist(mode=3, words=4096)
        self.assertEqual(frame, pack_u32_le(0xBB290001) + pack_u32_le(0x00100031))

    def test_encode_gpx2_config_profile(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(4, sequence_mode=2)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x00000024))

    def test_encode_gpx2_default_uses_stable_manual_full_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config()
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000A0))

    def test_encode_gpx2_legacy_no_readback_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(7, sequence_mode=8)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x00000087))

    def test_encode_gpx2_manual_slow_power_read_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(0, sequence_mode=9)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x00000090))

    def test_encode_gpx2_manual_slow_full_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(0, sequence_mode=10)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000A0))

    def test_encode_gpx2_manual_slow_write_only_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(0, sequence_mode=11)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000B0))

    def test_encode_gpx2_manual_slow_init_only_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(0, sequence_mode=12)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000C0))

    def test_encode_gpx2_extended_profile_and_sequence(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(15, sequence_mode=15)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000FF))

    def test_encode_gpx2_minimal_legal_init_config(self) -> None:
        encoder = CommandEncoder()
        frame = encoder.encode_gpx2_config(13, sequence_mode=10)
        self.assertEqual(frame, pack_u32_le(0xBB100001) + pack_u32_le(0x000000AD))

    def test_sanitize_gpx2_custom_config_masks_fixed_fields(self) -> None:
        regs = sanitize_gpx2_config_bytes([0xFF] * 17)
        self.assertEqual(regs[5], 0x0F)
        self.assertEqual(regs[6], 0xD0)
        self.assertEqual(regs[7], 0xF3)
        self.assertEqual(regs[8:16], (0xA1, 0x13, 0x00, 0x0A, 0xCC, 0xCC, 0xF1, 0x7D))
        self.assertEqual(regs[16], 0x04)

    def test_encode_gpx2_custom_config_packs_all_17_registers(self) -> None:
        encoder = CommandEncoder()
        regs = list(GPX2_DEFAULT_CONFIG_BYTES)
        regs[6] = 0x10
        regs[16] = 0x04
        sanitized = sanitize_gpx2_config_bytes(regs)
        frame = encoder.encode_gpx2_custom_config(regs, profile=3, sequence_mode=7)
        expected = pack_u32_le(0xBB110006) + pack_u32_le(0x00000073)
        for word in pack_gpx2_config_readback_words(sanitized):
            expected += pack_u32_le(word)
        self.assertEqual(frame, expected)

    def test_gpx2_register_specs_cover_all_17_registers(self) -> None:
        self.assertEqual(len(GPX2_REGISTER_SPECS), 17)
        self.assertEqual([spec.address for spec in GPX2_REGISTER_SPECS], list(range(17)))

    def test_gpx2_refclk_divisions_decodes_default_profile(self) -> None:
        self.assertEqual(gpx2_refclk_divisions(GPX2_DEFAULT_CONFIG_BYTES), 12500)

    def test_gpx2_lvds_test_preset_only_flips_expected_bits(self) -> None:
        regs = gpx2_lvds_test_config_bytes()
        self.assertEqual(regs[0], 0x3F)
        self.assertEqual(regs[6], 0xD0)
        self.assertEqual(regs[7], 0x53)

    def test_describe_gpx2_config_bytes_includes_refclk_summary(self) -> None:
        decoded = describe_gpx2_config_bytes(GPX2_DEFAULT_CONFIG_BYTES)
        self.assertEqual(decoded["reg01_HIT_COMBINE_RES"], "0x8F")
        self.assertEqual(decoded["reg03_REFCLK_DIV_LO"], "0xD4")
        self.assertEqual(decoded["reg16_CMOS_INPUT"], "0x00")
        self.assertEqual(decoded["refclk_divisions"], "12500")

    def test_parse_status_packet(self) -> None:
        parser = PacketParser()
        readback_regs = sanitize_gpx2_config_bytes(GPX2_DEFAULT_CONFIG_BYTES)
        words = [
            0xA5020104,
            0x00010014,
            0x000100A5,
            0x00001000,
            0x000000A5,
            0x0000000F,
            0x00000064,
            0x00000020,
            0x00000030,
            0x00000040,
            0x00000101,
            0x00000202,
            0x00000303,
            0x00000404,
            0x00054321,
        ]
        words.extend(pack_gpx2_config_readback_words(readback_regs))
        words.extend([0x10, 0x11, 0x12, 0x13])
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
        self.assertEqual(status.gpx2_raw_count_ch1, 0x101)
        self.assertEqual(status.gpx2_raw_count_ch2, 0x202)
        self.assertEqual(status.gpx2_raw_count_ch3, 0x303)
        self.assertEqual(status.gpx2_raw_count_ch4, 0x404)
        self.assertEqual(status.gpx2_cfg_diag, 0x54321)
        self.assertEqual(status.gpx2_cfg_readback, readback_regs)
        self.assertEqual(status.gate_trigger_count, 0x10)
        self.assertEqual(status.gate_direct_pulse_count, 0x11)
        self.assertEqual(status.gate_divided_pulse_count, 0x12)
        self.assertEqual(status.gate_output_pulse_count, 0x13)

    def test_parse_tdc_raw_packet(self) -> None:
        parser = PacketParser()
        low = (0x0321 & 0x3FFF) << 12
        high = (0x1 << 28) | (1 << 26) | (0x3456 & 0xFFFF)
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
        self.assertEqual(event.rec_type, 1)
        self.assertEqual(event.channel, 1)
        self.assertEqual(event.refid, 0x3456)
        self.assertEqual(event.tstop, 0x0321)
        batch = parsed[0].tdc_event_batch
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 1)
        self.assertEqual(int(batch.channels[0]), event.channel)
        self.assertEqual(int(batch.refids[0]), event.refid)
        self.assertEqual(int(batch.tstops[0]), event.tstop)
        self.assertIsNone(batch.timestamps)

    def test_parse_tdc_raw_ext_packet(self) -> None:
        parser = PacketParser()
        timestamp = 0x0123456789ABCDEF
        meta = (2 << 30) | ((0x0321 & 0x3FFF) << 16) | (0x3456 & 0xFFFF)
        words = [
            0xA5050104,
            0x00030003,
            0x00010000,
            0x000000AA,
            timestamp & 0xFFFFFFFF,
            (timestamp >> 32) & 0xFFFFFFFF,
            meta,
        ]
        packet = b"".join(pack_u32_le(word) for word in words)
        parsed = parser.feed(packet)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(parsed[0].tdc_events), 1)
        event = parsed[0].tdc_events[0]
        self.assertEqual(event.rec_type, 1)
        self.assertEqual(event.channel, 2)
        self.assertEqual(event.refid, 0x3456)
        self.assertEqual(event.tstop, 0x0321)
        batch = parsed[0].tdc_event_batch
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 1)
        self.assertIsNotNone(batch.timestamps)
        self.assertEqual(int(batch.timestamps[0]), timestamp)

    def test_parse_tdc_hist_packet(self) -> None:
        parser = PacketParser()
        payload = [
            7,
            1,
            5,
            3,
            20,
            5,
            100,
            0x13,
            101,
            202,
            77,
            9,
            4,
            3,
            1234,
            (11 << 16) | 22,
            10,
            20,
            30,
        ]
        words = [
            0xA5060104,
            (19 << 16) | 19,
            0x00010000,
            0x000000AA,
            *payload,
        ]

        parsed = parser.feed(b"".join(pack_u32_le(word) for word in words))

        self.assertEqual(len(parsed), 1)
        chunk = parsed[0].histogram_chunk
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.snapshot_id, 7)
        self.assertEqual(chunk.hist_id, 1)
        self.assertEqual(chunk.bin_start, 5)
        self.assertEqual(chunk.bin_count, 3)
        self.assertEqual(chunk.total_bins, 20)
        self.assertEqual(chunk.bin_width_raw, 5)
        self.assertEqual(chunk.offset_raw, 100)
        self.assertEqual(chunk.status_flags, 0x13)
        self.assertEqual(chunk.start_count, 101)
        self.assertEqual(chunk.stop_count, 202)
        self.assertEqual(chunk.accepted_count, 77)
        self.assertEqual(chunk.out_of_window_count, 9)
        self.assertEqual(chunk.no_start_count, 4)
        self.assertEqual(chunk.first_stop_suppressed_count, 3)
        self.assertEqual(chunk.last_dt_raw, 1234)
        self.assertEqual(chunk.reference_cleanup_count, 22)
        self.assertEqual(chunk.tdc_deadtime_filtered_count, 11)
        self.assertEqual(chunk.bins.tolist(), [10, 20, 30])

    def test_parse_ddr3_bist_packet(self) -> None:
        parser = PacketParser()
        expected_low = 0x89ABCDEF
        expected_high = 0x01234567
        actual_low = 0x76543210
        actual_high = 0xFEDCBA98
        payload = [
            0xDDB1300D,
            4096,
            4096,
            0,
            0,
            expected_low,
            expected_high,
            0,
            0,
            actual_low,
            actual_high,
            0,
            0,
            123,
            456,
            (29 << 8) | 128,
        ]
        words = [
            0xA5070104,
            (16 << 16) | 16,
            (16 << 16),
            0x000000AA,
            *payload,
        ]

        parsed = parser.feed(b"".join(pack_u32_le(word) for word in words))

        self.assertEqual(len(parsed), 1)
        result = parsed[0].ddr3_bist_result
        self.assertIsNotNone(result)
        self.assertEqual(result.mode, 3)
        self.assertTrue(result.calibrated)
        self.assertFalse(result.running)
        self.assertTrue(result.done)
        self.assertTrue(result.passed)
        self.assertFalse(result.failed)
        self.assertEqual(result.requested_words, 4096)
        self.assertEqual(result.tested_words, 4096)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.expected, (expected_high << 32) | expected_low)
        self.assertEqual(result.actual, (actual_high << 32) | actual_low)
        self.assertEqual(result.write_cycles, 123)
        self.assertEqual(result.read_cycles, 456)
        self.assertEqual(result.app_addr_width, 29)
        self.assertEqual(result.app_data_width, 128)

    def test_decode_tdc_hist_status_flags(self) -> None:
        decoded = decode_tdc_hist_status_flags(0x000063B1)
        self.assertTrue(decoded["enable"])
        self.assertTrue(decoded["input_stall"])
        self.assertTrue(decoded["core_present"])
        self.assertTrue(decoded["ddr_backend"])
        self.assertTrue(decoded["ddr_calibrated"])
        self.assertTrue(decoded["ddr_ring_overflow"])
        self.assertTrue(decoded["ddr_cdc_req_overflow"])
        self.assertTrue(decoded["ddr_cdc_rsp_overflow"])
        self.assertFalse(decoded["readout_busy"])
        self.assertFalse(decoded["ddr_not_calibrated"])
        description = describe_tdc_hist_status_flags(0x000063B1)
        self.assertIn("ddr_backend", description)
        self.assertIn("ddr_calibrated", description)

    def test_parser_skips_malformed_ext_payload_as_one_packet(self) -> None:
        parser = PacketParser()
        bad_timestamp = PacketParser.MAX_EXT_TIMESTAMP + 1
        bad_words = [
            0xA5050104,
            0x00030003,
            0x00010000,
            0x000000AA,
            bad_timestamp & 0xFFFFFFFF,
            (bad_timestamp >> 32) & 0xFFFFFFFF,
            (2 << 30) | (0x0321 << 16) | 0x3456,
        ]
        good_words = [
            0xA5020104,
            0x00010010,
            0x00010000,
            0x000000BB,
        ] + [0] * 16

        packet = b"".join(pack_u32_le(word) for word in bad_words + good_words)
        parsed = parser.feed(packet)

        self.assertEqual(len(parsed), 1)
        self.assertIsNotNone(parsed[0].status)
        self.assertEqual(parser.reject_counts, {"bad_ext_timestamp": 1})

    def test_parse_tdc_raw_packet_can_skip_event_objects(self) -> None:
        parser = PacketParser(decode_tdc_events=False)
        low = (0x321 & 0x3FFF) << 12
        high = (0x1 << 28) | (2 << 26) | (0x1234 & 0xFFFF)
        words = [
            0xA5010104,
            0x00020002,
            0x00010000,
            0x000000AA,
            low,
            high,
        ]

        parsed = parser.feed(b"".join(pack_u32_le(word) for word in words))

        self.assertEqual(len(parsed), 1)
        self.assertIsNone(parsed[0].tdc_events)
        batch = parsed[0].tdc_event_batch
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 1)
        self.assertEqual(int(batch.channels[0]), 2)
        self.assertEqual(int(batch.refids[0]), 0x1234)
        self.assertEqual(int(batch.tstops[0]), 0x321)

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
