from __future__ import annotations

import unittest

from app.protocol import LegacyAnalogCodec


class LegacyAnalogCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = LegacyAnalogCodec()

    def test_threshold_mv_to_code(self) -> None:
        self.assertEqual(self.codec.threshold_mv_to_code(0.0), 0)
        self.assertEqual(self.codec.threshold_mv_to_code(2500.0), 65535)
        self.assertEqual(self.codec.threshold_mv_to_code(1250.0), 32768)

    def test_bias_v_to_code(self) -> None:
        self.assertEqual(self.codec.bias_v_to_code(0.0), 0)
        self.assertEqual(self.codec.bias_v_to_code(1.0), 433)
        self.assertEqual(self.codec.bias_v_to_code(45.0), 19485)

    def test_temperature_c_to_code_matches_legacy_formula(self) -> None:
        self.assertEqual(self.codec.temperature_c_to_code(25.0), 5992)
        self.assertEqual(self.codec.temperature_c_to_code(20.0), 7298)

    def test_encode_all_analog_targets(self) -> None:
        codes = self.codec.encode_analog_targets(100.0, 200.0, 300.0, 50.0)
        self.assertEqual(codes.laser_sync_code, 2621)
        self.assertEqual(codes.pixel_sync_code, 5243)
        self.assertEqual(codes.avalanche_code, 7864)
        self.assertEqual(codes.bias_code, 21650)


if __name__ == "__main__":
    unittest.main()
