from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

from app.models import CommandResult, HistogramSnapshot, PacketHeader, StatusPacket
from app.protocol import CMD_FLASH_SAVE_ANALOG, CMD_GATE_ENABLE, CMD_GATE_SIG3, CMD_GATE_SIG3_LONG, CMD_TDC_TEST_UPLOAD
from app.ui.main_window import MainWindow


class FakeController:
    def __init__(self) -> None:
        self.configure_calls: list[dict] = []
        self.save_analog_flash_calls: list[tuple] = []
        self.gate_signal_calls: list[tuple] = []
        self.gate_sig3_long_calls: list[tuple] = []
        self.gate_enable_calls: list[tuple] = []
        self._last_upload_result = CommandResult(True, CMD_TDC_TEST_UPLOAD, "SENT")

    def configure_tdc_test(self, **kwargs) -> HistogramSnapshot:
        self.configure_calls.append(kwargs)
        bin_count = max(1, int(kwargs.get("bin_count", 16)))
        return HistogramSnapshot(
            histograms={"tdc_test": [0] * bin_count},
            image_projection=[[0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]],
        )

    def last_tdc_test_upload_result(self) -> CommandResult:
        return self._last_upload_result

    def save_analog_defaults_to_flash(self, *args) -> CommandResult:
        self.save_analog_flash_calls.append(args)
        return CommandResult(True, CMD_FLASH_SAVE_ANALOG, "SENT")

    def set_gate_signal(
        self,
        signal_id: int,
        delay_coarse: int,
        delay_fine: int,
        width_coarse: int,
        width_fine: int,
    ) -> CommandResult:
        self.gate_signal_calls.append((signal_id, delay_coarse, delay_fine, width_coarse, width_fine))
        return CommandResult(True, CMD_GATE_SIG3, "SENT")

    def set_gate_sig3_long_delay(self, delay_ns: int, width_coarse: int, width_fine: int) -> CommandResult:
        self.gate_sig3_long_calls.append((delay_ns, width_coarse, width_fine))
        return CommandResult(True, CMD_GATE_SIG3_LONG, "SENT")

    def set_gate_enable(self, sig2_enable: bool, sig3_enable: bool, pixel_mode: bool) -> CommandResult:
        self.gate_enable_calls.append((sig2_enable, sig3_enable, pixel_mode))
        return CommandResult(True, CMD_GATE_ENABLE, "SENT")


def make_status(
    seq: int,
    *,
    counter_1s: int,
    trigger_count: int,
    sig3_count: int,
    output_count: int,
) -> StatusPacket:
    return StatusPacket(
        header=PacketHeader(
            sync=0xA5,
            pkt_type=0x02,
            version=1,
            hdr_words=4,
            seq=seq,
            payload_words=20,
            item_count=0,
            flags=0,
            timestamp_us=seq * 1000000,
        ),
        flags=0,
        uptime_seconds=seq,
        temp_avg_raw=0,
        counter_1s=counter_1s,
        tdc_drop_count=0,
        usb_drop_count=0,
        gate_trigger_count=trigger_count,
        gate_divided_pulse_count=sig3_count,
        gate_output_pulse_count=output_count,
    )


class MainWindowTdcStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.controller = FakeController()
        self.window.controller = self.controller
        self.window.tdc_source_combo.setCurrentIndex(self.window.tdc_source_combo.findData("gpx2"))
        self.window.tdc_start_combo.setCurrentIndex(self.window.tdc_start_combo.findData(2))
        self.window.tdc_stop_combo.setCurrentIndex(self.window.tdc_stop_combo.findData(3))
        self.window.tdc_bin_count_spin.setValue(32)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self._qt_app.processEvents()

    def test_start_forces_upload_enabled_even_when_running_box_is_clear(self) -> None:
        self.window.tdc_test_enable_check.setChecked(False)
        self.window.tdc_acq_time_spin.setValue(0.0)

        self.window.on_apply_tdc_test()

        self.assertEqual(len(self.controller.configure_calls), 1)
        self.assertTrue(self.controller.configure_calls[0]["enabled"])
        self.assertEqual(self.controller.configure_calls[0]["pairing_mode"], "prev_start")
        self.assertEqual(self.controller.configure_calls[0]["tdc_deadtime_ps"], 0.0)
        self.assertTrue(self.window.tdc_test_enable_check.isChecked())
        self.assertEqual(self.window._tdc_test_upload_payload_word, 0x69)
        self.assertIn("upload: 0x00000069", self.window.tdc_test_status_label.text())
        self.assertFalse(self.window._tdc_test_timer.isActive())

    def test_positive_acq_time_starts_auto_stop_timer(self) -> None:
        self.window.tdc_test_enable_check.setChecked(False)
        self.window.tdc_acq_time_spin.setValue(0.25)

        self.window.on_apply_tdc_test()

        self.assertTrue(self.controller.configure_calls[0]["enabled"])
        self.assertEqual(self.window._tdc_test_upload_payload_word, 0x69)
        self.assertTrue(self.window._tdc_test_timer.isActive())
        self.window._tdc_test_timer.stop()

    def test_tdc_snapshot_accepts_numpy_histogram(self) -> None:
        snapshot = HistogramSnapshot(
            histograms={"tdc_test": np.asarray([0, 3, 1, 0], dtype=np.uint32)},
            image_projection=[[4, 2, 2, 10, 10, 20, 0, 0, 0, 0, 2, 2, 0, 0]],
        )

        self.window.on_tdc_test_snapshot(snapshot)

        self.assertIn("pairs: 4", self.window.tdc_test_status_label.text())
        self.assertIn("pairs/photon: 2.00", self.window.tdc_test_status_label.text())
        self.assertIn("commercial/nearest-start", self.window.tdc_test_status_label.text())
        self.assertIn("FWHM(linear full-res)", self.window.tdc_test_status_label.text())
        self.assertIn("(3)", self.window.tdc_test_status_label.text())
        _x, y = self.window.tdc_test_curve.getData()
        np.testing.assert_allclose(y, np.repeat(np.log10(np.asarray([1, 3, 1, 1], dtype=float)), 2))

    def test_save_analog_defaults_button_sends_all_current_values(self) -> None:
        self.window.temp_target_spin.setValue(23.5)
        self.window.laser_thr_spin.setValue(111.0)
        self.window.pixel_thr_spin.setValue(222.0)
        self.window.avalanche_thr_spin.setValue(33.0)
        self.window.bias_spin.setValue(44.5)

        self.window.on_save_analog_defaults_to_flash()

        self.assertEqual(
            self.controller.save_analog_flash_calls[-1],
            (23.5, 111.0, 222.0, 33.0, 44.5, 1500, 30000),
        )

    def test_sig3_sweep_sends_delay_points_and_records_counts(self) -> None:
        self.window.sig3_sweep_start_ns_spin.setValue(0)
        self.window.sig3_sweep_stop_ns_spin.setValue(12)
        self.window.sig3_sweep_step_ns_spin.setValue(12)
        self.window.sig3_sweep_restore_check.setChecked(False)

        self.window.on_start_sig3_sweep()

        self.assertEqual(self.controller.gate_enable_calls, [(False, True, False)])
        self.assertEqual(self.controller.gate_sig3_long_calls[0], (0, 0, 10))

        self.window._last_status_packet = make_status(
            1,
            counter_1s=10,
            trigger_count=100,
            sig3_count=200,
            output_count=300,
        )
        self.window._on_sig3_sweep_timer()
        self.window._sig3_sweep_deadline = 0.0
        self.window._last_status_packet = make_status(
            2,
            counter_1s=123,
            trigger_count=105,
            sig3_count=207,
            output_count=312,
        )
        self.window._on_sig3_sweep_timer()

        self.assertEqual(self.controller.gate_sig3_long_calls[1], (12, 0, 10))

        self.window._last_status_packet = make_status(
            3,
            counter_1s=20,
            trigger_count=200,
            sig3_count=400,
            output_count=600,
        )
        self.window._on_sig3_sweep_timer()
        self.window._sig3_sweep_deadline = 0.0
        self.window._last_status_packet = make_status(
            4,
            counter_1s=456,
            trigger_count=211,
            sig3_count=419,
            output_count=623,
        )
        self.window._on_sig3_sweep_timer()

        self.assertFalse(self.window._sig3_sweep_active)
        self.assertEqual(self.window.sig3_sweep_table.rowCount(), 2)
        self.assertEqual(self.window.sig3_sweep_table.item(0, 0).text(), "0")
        self.assertEqual(self.window.sig3_sweep_table.item(0, 1).text(), "123")
        self.assertEqual(self.window.sig3_sweep_table.item(0, 2).text(), "7")
        self.assertEqual(self.window.sig3_sweep_table.item(1, 0).text(), "12")
        self.assertEqual(self.window.sig3_sweep_table.item(1, 1).text(), "456")
        self.assertEqual(self.window.sig3_sweep_table.item(1, 2).text(), "19")
        self.assertTrue(self.window.sig3_sweep_apply_best_btn.isEnabled())

        self.window.on_apply_best_sig3_sweep_delay()

        self.assertEqual(self.controller.gate_sig3_long_calls[-1], (12, 0, 10))
        self.assertEqual(self.window.sig3_sweep_table.currentRow(), 1)
        self.assertIn("12 ns", self.window.sig3_sweep_status_label.text())

    def test_sig3_sweep_window_is_separate_and_supports_50us_stop(self) -> None:
        self.assertEqual(self.window.sig3_sweep_stop_ns_spin.maximum(), 50000)
        self.assertFalse(self.window.sig3_sweep_apply_best_btn.isEnabled())

        self.window.on_open_sig3_sweep_window()

        self.assertIsNotNone(self.window.sig3_sweep_window)
        self.assertTrue(self.window.sig3_sweep_window.isVisible())
        self.assertIs(self.window.sig3_sweep_plot.window(), self.window.sig3_sweep_window)

    def test_tdc_snapshot_respects_min_bin_for_peak_and_axis(self) -> None:
        self.window.tdc_display_min_bin_spin.setValue(2)
        self.window.tdc_bin_width_spin.setValue(5)
        snapshot = HistogramSnapshot(
            histograms={"tdc_test": np.asarray([99, 5, 8, 2], dtype=np.uint32)},
            image_projection=[[114, 2, 2, 10, 10, 20, 0, 0, 0, 0, 2, 2, 0, 0]],
        )

        self.window.on_tdc_test_snapshot(snapshot)

        x, y = self.window.tdc_test_curve.getData()
        self.assertIn("peak: bin 2", self.window.tdc_test_status_label.text())
        np.testing.assert_allclose(x, np.asarray([80.0, 120.0, 120.0, 160.0]))
        np.testing.assert_allclose(y, np.repeat(np.log10(np.asarray([8, 2], dtype=float)), 2))

    def test_tdc_snapshot_reports_full_resolution_fwhm_when_plot_downsamples(self) -> None:
        self.window._tdc_test_plot_max_points = 10
        self.window.tdc_bin_width_spin.setValue(5)
        hist = np.zeros(50, dtype=np.uint32)
        hist[30] = 100
        hist[31] = 40
        snapshot = HistogramSnapshot(
            histograms={"tdc_test": hist},
            image_projection=[[140, 2, 2, 150, 150, 155, 0, 0, 0, 0, 2, 2, 0, 0]],
        )

        self.window.on_tdc_test_snapshot(snapshot)

        self.assertIn("FWHM(linear full-res) 53.3 ps", self.window.tdc_test_status_label.text())
        x, y = self.window.tdc_test_curve.getData()
        self.assertLessEqual(len(x), self.window._tdc_test_plot_max_points * 2)
        self.assertLessEqual(len(y), self.window._tdc_test_plot_max_points * 2)
        zx, zy = self.window.tdc_peak_curve.getData()
        self.assertGreater(len(zx), 0)
        self.assertGreater(float(np.max(zy)), 0.0)


if __name__ == "__main__":
    unittest.main()
