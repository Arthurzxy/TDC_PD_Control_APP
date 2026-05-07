from __future__ import annotations

import unittest

from app.data_processing import HistogramBuilder, TdcTestHistogramBuilder
from app.models import HistogramSettings, PacketHeader, PhotonEvent, TdcEvent, TdcTestSettings


class DataProcessingTests(unittest.TestCase):
    def test_histogram_builder_uses_fpga_photon_bin_index(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x04,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=5,
            item_count=1,
            flags=0,
            timestamp_us=123,
        )
        event = PhotonEvent(
            header=header,
            packet_type=0x01,
            frame_id=0,
            line_id=2,
            pixel_id=3,
            bin_index=187,
            dt_8ps=1500,
            detector_timestamp_low=15000,
            packet_seq=1,
            packet_timestamp_us=123,
        )
        builder = HistogramBuilder(HistogramSettings(bin_width_raw=8, bin_offset=99, bin_count=512))

        builder.process_photon(event)
        snapshot = builder.snapshot(current_row=2, current_col=3)

        self.assertEqual(snapshot.current_row, 2)
        self.assertEqual(snapshot.current_col, 3)
        self.assertEqual(snapshot.image_projection[2][3], 1)
        self.assertEqual(snapshot.histograms["2,3"][187], 1)

    def test_tdc_test_histogram_pairs_start_stop_channels(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=4,
            item_count=2,
            flags=0,
            timestamp_us=123,
        )
        builder = TdcTestHistogramBuilder(
            TdcTestSettings(
                enabled=True,
                start_channel=0,
                stop_channel=1,
                refclk_divisions=12500,
                bin_width_raw=8,
                bin_offset=0,
                bin_count=64,
            )
        )
        start = TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123)
        stop = TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123)

        updated = builder.process_events([start, stop])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)


if __name__ == "__main__":
    unittest.main()
