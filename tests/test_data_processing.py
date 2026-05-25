from __future__ import annotations

import unittest

import numpy as np

from app.data_processing import (
    HistogramBuilder,
    TcspcCommercialAnalyzer,
    TcspcRawReplayAnalyzer,
    TdcTestHistogramBuilder,
    TcspcTimeReconstructor,
)
from app.models import HistogramSettings, PacketHeader, PhotonEvent, TdcEvent, TdcRawEventBatch, TdcTestSettings


class DataProcessingTests(unittest.TestCase):
    def test_tdc_test_snapshot_keeps_histogram_as_numpy_array(self) -> None:
        builder = TdcTestHistogramBuilder(TdcTestSettings(bin_count=16))

        snapshot = builder.snapshot()
        hist = snapshot.histograms["tdc_test"]

        self.assertIsInstance(hist, np.ndarray)
        self.assertEqual(hist.dtype, np.uint32)
        self.assertEqual(hist.shape, (16,))

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

    def test_tdc_test_histogram_accepts_photon_after_sync(self) -> None:
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
        self.assertEqual(snapshot.image_projection[0][1], 1)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][3], 80)
        self.assertEqual(snapshot.image_projection[0][4], 80)
        self.assertEqual(snapshot.image_projection[0][5], 80)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_reports_sync_without_photon(self) -> None:
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

        updated = builder.process_events([TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123)])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 0)
        self.assertEqual(snapshot.image_projection[0][1], 1)
        self.assertEqual(snapshot.image_projection[0][2], 0)

    def test_tdc_test_histogram_multi_hit_does_not_consume_sync(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=8,
            item_count=4,
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
        events = [
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 220, 0, 1, 123),
        ]

        updated = builder.process_events(events)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 2)
        self.assertEqual(snapshot.image_projection[0][1], 1)
        self.assertEqual(snapshot.image_projection[0][2], 2)
        self.assertEqual(snapshot.image_projection[0][7], 0)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][15], 1)

    def test_tdc_test_histogram_counts_photon_before_first_sync_as_orphan(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=2,
            item_count=1,
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

        updated = builder.process_events([TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123)])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 0)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][13], 1)

    def test_tdc_test_histogram_counts_photon_without_window_pair_as_orphan(self) -> None:
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

        updated = builder.process_events([
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 1000, 0, 1, 123),
        ])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 0)
        self.assertEqual(snapshot.image_projection[0][8], 0)
        self.assertEqual(snapshot.image_projection[0][13], 1)

    def test_tdc_test_histogram_applies_bin_offset(self) -> None:
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
                bin_offset=40,
                bin_count=64,
            )
        )

        updated = builder.process_events([
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123),
        ])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.histograms["tdc_test"][5], 1)

    def test_tdc_test_histogram_extends_16_bit_refid_wrap(self) -> None:
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
                bin_width_raw=5,
                bin_offset=0,
                bin_count=128,
            )
        )
        start = TdcEvent(header, 0, 0, 0, 0xFFFF, 12400, 0, 1, 123)
        stop = TdcEvent(header, 0, 1, 0, 0x0000, 100, 0, 1, 123)

        updated = builder.process_events([start, stop])
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][3], 200)
        self.assertEqual(snapshot.image_projection[0][7], 0)
        self.assertEqual(snapshot.histograms["tdc_test"][40], 1)

    def test_tdc_test_histogram_processes_raw_event_batch(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=6,
            item_count=3,
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
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([0, 1, 1], dtype=np.uint8),
            refids=np.array([0, 0, 0], dtype=np.uint32),
            tstops=np.array([100, 180, 220], dtype=np.uint32),
            rec_types=np.array([0, 0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0, 0], dtype=np.uint8),
            reserved=np.array([0, 0, 0], dtype=np.uint32),
        )

        updated = builder.process_batch(batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 2)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][15], 1)

    def test_tdc_test_histogram_processes_ch2_ch3_raw_event_batch(self) -> None:
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
                start_channel=1,
                stop_channel=2,
                refclk_divisions=12500,
                bin_width_raw=8,
                bin_offset=0,
                bin_count=64,
            )
        )
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([1, 2], dtype=np.uint8),
            refids=np.array([0, 0], dtype=np.uint32),
            tstops=np.array([100, 180], dtype=np.uint32),
            rec_types=np.array([0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0], dtype=np.uint8),
            reserved=np.array([0, 0], dtype=np.uint32),
        )

        updated = builder.process_batch(batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][1], 1)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][10], 1)
        self.assertEqual(snapshot.image_projection[0][11], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_uses_extended_timestamps_directly(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x05,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=9,
            item_count=3,
            flags=0,
            timestamp_us=123,
        )
        refclk_divisions = 12500
        builder = TdcTestHistogramBuilder(
            TdcTestSettings(
                enabled=True,
                start_channel=1,
                stop_channel=2,
                refclk_divisions=refclk_divisions,
                bin_width_raw=8,
                bin_offset=0,
                bin_count=64,
                reference_deadtime_ns=0.0,
            )
        )
        sync_old = 65535 * refclk_divisions + 100
        sync_new = 65536 * refclk_divisions + 100
        photon = sync_new + 80
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([1, 1, 2], dtype=np.uint8),
            refids=np.array([0xFFFF, 0x0000, 0x0000], dtype=np.uint32),
            tstops=np.array([100, 100, 180], dtype=np.uint32),
            rec_types=np.array([1, 1, 1], dtype=np.uint8),
            event_classes=np.array([0, 0, 0], dtype=np.uint8),
            reserved=np.array([0, 0, 0], dtype=np.uint32),
            timestamps=np.array([sync_old, sync_new, photon], dtype=np.uint64),
        )

        updated = builder.process_batch(batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][15], 0)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_pairs_by_timestamp_when_packet_order_is_late(self) -> None:
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
                start_channel=1,
                stop_channel=2,
                refclk_divisions=12500,
                bin_width_raw=8,
                bin_offset=0,
                bin_count=64,
            )
        )
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([2, 1], dtype=np.uint8),
            refids=np.array([0, 0], dtype=np.uint32),
            tstops=np.array([180, 100], dtype=np.uint32),
            rec_types=np.array([0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0], dtype=np.uint8),
            reserved=np.array([0, 0], dtype=np.uint32),
        )

        updated = builder.process_batch(batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][7], 0)
        self.assertEqual(snapshot.image_projection[0][8], 0)
        self.assertEqual(snapshot.image_projection[0][13], 0)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_keeps_syncs_for_delayed_channel_batches(self) -> None:
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
                start_channel=1,
                stop_channel=2,
                refclk_divisions=12500,
                bin_width_raw=8,
                bin_offset=0,
                bin_count=64,
            )
        )
        sync_batch = TdcRawEventBatch(
            header=header,
            channels=np.array([1, 1], dtype=np.uint8),
            refids=np.array([0, 160], dtype=np.uint32),
            tstops=np.array([100, 0], dtype=np.uint32),
            rec_types=np.array([0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0], dtype=np.uint8),
            reserved=np.array([0, 0], dtype=np.uint32),
        )
        photon_batch = TdcRawEventBatch(
            header=header,
            channels=np.array([2], dtype=np.uint8),
            refids=np.array([0], dtype=np.uint32),
            tstops=np.array([180], dtype=np.uint32),
            rec_types=np.array([0], dtype=np.uint8),
            event_classes=np.array([0], dtype=np.uint8),
            reserved=np.array([0], dtype=np.uint32),
        )

        builder.process_batch(sync_batch)
        updated = builder.process_batch(photon_batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][13], 0)
        self.assertEqual(snapshot.image_projection[0][15], 1)
        self.assertEqual(snapshot.image_projection[0][16], 0)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_pairs_one_photon_with_all_references_in_window(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=6,
            item_count=3,
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
                pairing_mode="all_start",
                reference_deadtime_ns=0.0,
            )
        )
        events = [
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 0, 0, 0, 140, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123),
        ]

        updated = builder.process_events(events)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 2)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][14], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][5], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_defaults_to_nearest_reference(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=6,
            item_count=3,
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
                reference_deadtime_ns=0.0,
            )
        )
        events = [
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 0, 0, 0, 140, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123),
        ]

        updated = builder.process_events(events)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][14], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][5], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 0)

    def test_tdc_test_histogram_optional_deadtime_filters_same_channel_repeats(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=8,
            item_count=4,
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
                tdc_deadtime_ps=100.0,
            )
        )
        events = [
            TdcEvent(header, 0, 0, 0, 0, 100, 0, 1, 123),
            TdcEvent(header, 0, 0, 0, 0, 105, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 180, 0, 1, 123),
            TdcEvent(header, 0, 1, 0, 0, 185, 0, 1, 123),
        ]

        updated = builder.process_events(events)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 1)
        self.assertEqual(snapshot.image_projection[0][1], 1)
        self.assertEqual(snapshot.image_projection[0][2], 1)
        self.assertEqual(snapshot.image_projection[0][23], 2)
        self.assertEqual(snapshot.histograms["tdc_test"][10], 1)

    def test_tcspc_shape_uses_linear_background_corrected_interpolated_fwhm(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=50,
        )
        hist = np.zeros(50, dtype=np.uint32)
        hist[30] = 100
        hist[31] = 40

        shape = TcspcTimeReconstructor.measure_histogram_shape(hist, settings)

        self.assertAlmostEqual(shape.half_count, 50.0)
        self.assertAlmostEqual(shape.fwhm_left_raw, 147.5)
        self.assertAlmostEqual(shape.fwhm_right_raw, 154.1666666667)
        self.assertAlmostEqual(shape.fwhm_ns, 0.0533333333)

    def test_tdc_test_histogram_does_not_fold_photon_before_reference_with_modulo(self) -> None:
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
                bin_count=2048,
            )
        )
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([1, 0], dtype=np.uint8),
            refids=np.array([10, 10], dtype=np.uint32),
            tstops=np.array([100, 200], dtype=np.uint32),
            rec_types=np.array([0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0], dtype=np.uint8),
            reserved=np.array([0, 0], dtype=np.uint32),
        )

        updated = builder.process_batch(batch)
        snapshot = builder.snapshot()

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 0)
        self.assertEqual(snapshot.image_projection[0][13], 1)
        self.assertEqual(snapshot.image_projection[0][17], 1)
        self.assertEqual(int(snapshot.histograms["tdc_test"].sum()), 0)

    def test_tdc_test_histogram_dark_control_can_fill_window_without_single_fake_peak(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x01,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=40,
            item_count=20,
            flags=0,
            timestamp_us=123,
        )
        builder = TdcTestHistogramBuilder(
            TdcTestSettings(
                enabled=True,
                start_channel=0,
                stop_channel=1,
                refclk_divisions=10_000,
                bin_width_raw=10,
                bin_offset=0,
                bin_count=10,
                reference_deadtime_ns=0.0,
            )
        )
        events: list[TdcEvent] = []
        for idx in range(10):
            base_ref = idx * 2
            events.append(TdcEvent(header, 0, 0, 0, base_ref, 0, 0, 1, 123))
            events.append(TdcEvent(header, 0, 1, 0, base_ref, idx * 10 + 1, 0, 1, 123))

        updated = builder.process_events(events)
        snapshot = builder.snapshot()
        hist = np.asarray(snapshot.histograms["tdc_test"], dtype=np.uint32)

        self.assertTrue(updated)
        self.assertEqual(snapshot.image_projection[0][0], 10)
        self.assertTrue(np.all(hist == 1))

    def test_tcspc_shape_diag_all_start_has_two_period_peaks(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=10,
            bin_offset=0,
            bin_count=25000,
        )
        period_raw = 200_000
        sync_ts = np.arange(8, dtype=np.int64) * period_raw
        photon_ts = sync_ts + 1_000

        pairs = TcspcTimeReconstructor.pair_indices(
            sync_ts,
            photon_ts,
            0,
            250_000,
            "all_start",
        )
        hist = TcspcTimeReconstructor.histogram_from_dt(pairs.dt_raw, settings)
        shape = TcspcTimeReconstructor.measure_histogram_shape(
            hist,
            settings,
            expected_peak_separation_raw=period_raw,
        )

        self.assertEqual(hist[100], 8)
        self.assertEqual(hist[20100], 7)
        self.assertEqual(shape.peak_separation_raw, period_raw)

    def test_tcspc_shape_diag_can_identify_reverse_fine_time_formula(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=128,
        )
        refs = np.arange(12, dtype=np.int64) * 16
        start_tstop = 500 + np.arange(12, dtype=np.int64) * 7
        stop_tstop = start_tstop - 80
        channels = np.ravel(np.column_stack((
            np.zeros(refs.size, dtype=np.int64),
            np.ones(refs.size, dtype=np.int64),
        )))
        ext_refs = np.ravel(np.column_stack((refs, refs)))
        tstops = np.ravel(np.column_stack((start_tstop, stop_tstop)))

        diagnostics = TcspcTimeReconstructor.shape_diagnostics(channels, ext_refs, tstops, settings)
        by_name = {diag.name: diag for diag in diagnostics}

        self.assertEqual(by_name["all_start_forward"].accepted_pairs, 0)
        self.assertEqual(by_name["all_start_reverse_same"].accepted_pairs, refs.size)
        self.assertEqual(by_name["all_start_reverse_same"].shape.fwhm_raw, settings.bin_width_raw)

    def test_tcspc_prev_start_pairs_only_nearest_reference(self) -> None:
        sync_ts = np.array([0, 40], dtype=np.int64)
        photon_ts = np.array([80], dtype=np.int64)

        all_pairs = TcspcTimeReconstructor.pair_indices(sync_ts, photon_ts, 0, 100, "all_start")
        prev_pairs = TcspcTimeReconstructor.pair_indices(sync_ts, photon_ts, 0, 100, "prev_start")

        self.assertEqual(sorted(all_pairs.dt_raw.tolist()), [40, 80])
        self.assertEqual(prev_pairs.dt_raw.tolist(), [40])
        self.assertEqual(all_pairs.paired_photons, 1)
        self.assertEqual(prev_pairs.paired_photons, 1)

    def test_tdc_test_histogram_discards_stale_raw_epoch_batches(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x05,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=6,
            item_count=2,
            flags=0,
            timestamp_us=123,
        )
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=8,
            bin_offset=0,
            bin_count=64,
            raw_epoch=2,
        )
        stale = TdcRawEventBatch(
            header=header,
            channels=np.array([0, 1], dtype=np.uint8),
            refids=np.array([0, 0], dtype=np.uint32),
            tstops=np.array([100, 180], dtype=np.uint32),
            rec_types=np.array([0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0], dtype=np.uint8),
            reserved=np.array([0, 0], dtype=np.uint32),
            timestamps=np.array([100, 180], dtype=np.uint64),
            raw_epoch=1,
            raw_sequence=1,
        )
        fresh = TdcRawEventBatch(
            header=header,
            channels=stale.channels,
            refids=stale.refids,
            tstops=stale.tstops,
            rec_types=stale.rec_types,
            event_classes=stale.event_classes,
            reserved=stale.reserved,
            timestamps=stale.timestamps,
            raw_epoch=2,
            raw_sequence=2,
        )
        builder = TdcTestHistogramBuilder(settings)

        self.assertTrue(builder.process_batch(stale))
        stale_snapshot = builder.snapshot()
        self.assertEqual(int(stale_snapshot.histograms["tdc_test"].sum()), 0)
        self.assertEqual(stale_snapshot.image_projection[0][24], 1)

        self.assertTrue(builder.process_batch(fresh))
        fresh_snapshot = builder.snapshot()
        self.assertEqual(fresh_snapshot.image_projection[0][0], 1)
        self.assertEqual(fresh_snapshot.histograms["tdc_test"][10], 1)

    def test_tdc_test_histogram_reference_cleanup_removes_start_multi_edges(self) -> None:
        header = PacketHeader(
            sync=0xA5,
            pkt_type=0x05,
            version=0x01,
            hdr_words=4,
            seq=1,
            payload_words=9,
            item_count=3,
            flags=0,
            timestamp_us=123,
        )
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=1,
            bin_offset=0,
            bin_count=2000,
            pairing_mode="prev_start",
            reference_deadtime_ns=10.0,
        )
        batch = TdcRawEventBatch(
            header=header,
            channels=np.array([0, 0, 1], dtype=np.uint8),
            refids=np.array([0, 0, 0], dtype=np.uint32),
            tstops=np.array([0, 500, 1500], dtype=np.uint32),
            rec_types=np.array([0, 0, 0], dtype=np.uint8),
            event_classes=np.array([0, 0, 0], dtype=np.uint8),
            reserved=np.array([0, 0, 0], dtype=np.uint32),
            timestamps=np.array([0, 500, 1500], dtype=np.uint64),
        )
        builder = TdcTestHistogramBuilder(settings)

        self.assertTrue(builder.process_batch(batch))
        snapshot = builder.snapshot()

        self.assertEqual(snapshot.histograms["tdc_test"][1500], 1)
        self.assertEqual(snapshot.histograms["tdc_test"][1000], 0)
        self.assertEqual(snapshot.image_projection[0][26], 1)

    def test_commercial_analyzer_reports_deadtime_violation_and_stop_repeat_source(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=4000,
            detector_deadtime_ns=100.0,
        )
        hist = np.zeros(settings.bin_count, dtype=np.uint32)
        hist[100] = 1000
        hist[200] = 30
        stop_repeat = np.zeros(settings.bin_count, dtype=np.uint32)
        stop_repeat[100] = 50

        metrics = TcspcCommercialAnalyzer.analyze(
            hist,
            settings,
            stop_repeat_histogram=stop_repeat,
            detector_deadtime_ns=100.0,
        )

        self.assertGreater(metrics.deadtime_violation_count, 0)
        self.assertGreater(metrics.deadtime_violation_ratio, 0.0)
        self.assertEqual(metrics.satellite_decision, "likely_stop_channel_retrigger_or_afterpulse")
        self.assertEqual(metrics.satellite_peaks[0].bin_index, 200)

    def test_raw_replay_sorted_histogram_recovers_stream_order_regression(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=1,
            bin_offset=0,
            bin_count=2000,
            reference_deadtime_ns=0.0,
        )
        channels = np.array([1, 0, 1, 0], dtype=np.uint8)
        timestamps = np.array([1150, 1000, 2150, 2000], dtype=np.uint64)
        refids = (timestamps // settings.refclk_divisions).astype(np.uint32)
        tstops = (timestamps % settings.refclk_divisions).astype(np.uint32)
        stream_hist = np.zeros(settings.bin_count, dtype=np.uint32)
        stream_counts = [0, 2, 2, -1, -1, -1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 1]

        result = TcspcRawReplayAnalyzer.analyze_events(
            channels,
            timestamps,
            refids,
            tstops,
            settings,
            stream_histogram=stream_hist,
            stream_counts=stream_counts,
            reference_deadtime_ns=0.0,
        )

        self.assertEqual(result.timestamp_regressions, 2)
        self.assertEqual(result.sorted_prev_start.histogram[150], 2)
        self.assertIn("stream_order_fault", result.flags)

    def test_raw_replay_satellite_flags_stop_repeat(self) -> None:
        settings = TdcTestSettings(
            enabled=True,
            start_channel=0,
            stop_channel=1,
            refclk_divisions=12500,
            bin_width_raw=1,
            bin_offset=0,
            bin_count=3000,
            reference_deadtime_ns=0.0,
        )
        channels = []
        timestamps = []
        for cycle in range(20):
            base = cycle * 10000
            channels.extend([0, 1, 1])
            timestamps.extend([base, base + 1000, base + 1500])
        channel_arr = np.asarray(channels, dtype=np.uint8)
        timestamp_arr = np.asarray(timestamps, dtype=np.uint64)
        refids = (timestamp_arr // settings.refclk_divisions).astype(np.uint32)
        tstops = (timestamp_arr % settings.refclk_divisions).astype(np.uint32)

        result = TcspcRawReplayAnalyzer.analyze_events(
            channel_arr,
            timestamp_arr,
            refids,
            tstops,
            settings,
            reference_deadtime_ns=0.0,
        )

        self.assertEqual(result.sorted_prev_start.histogram[1000], 20)
        self.assertEqual(result.sorted_prev_start.histogram[1500], 20)
        self.assertIn("stop_retrigger_or_afterpulse", result.flags)
        self.assertGreater(result.stop_repeat_histogram[500], 0)


if __name__ == "__main__":
    unittest.main()
