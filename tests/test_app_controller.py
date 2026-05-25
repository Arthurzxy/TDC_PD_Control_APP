from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from app.app_controller import AppController
from app.models import CommandResult, PixelParamRecord
from app.protocol import CMD_FLASH_SAVE_ANALOG, GPX2_DEFAULT_CONFIG_BYTES
from app.usb_link import DeviceInfo


class DummyWindow:
    def __init__(self) -> None:
        self.bound_controller = None

    def bind_controller(self, controller) -> None:
        self.bound_controller = controller


class AppControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.window = DummyWindow()
        self.controller = AppController(self.window)
        self.tempdir = tempfile.TemporaryDirectory()
        self.controller.default_config_path = Path(self.tempdir.name) / "test_config.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_binds_controller_to_window(self) -> None:
        self.assertIs(self.window.bound_controller, self.controller)

    def test_refresh_devices_delegates_to_usb_service(self) -> None:
        expected = [DeviceInfo(index=1, description="FT60x Device 1")]
        self.controller._device_service.enumerate_devices = Mock(return_value=expected)

        self.assertEqual(self.controller.refresh_devices(), expected)

    def test_apply_temperature_target_updates_config_and_saves(self) -> None:
        self.controller._temperature_service.set_target_temperature_c = Mock(
            return_value=CommandResult(True, 0x04, "SENT")
        )
        self.controller.save_current_config = Mock()

        result = self.controller.apply_temperature_target(27.5, 1500)

        self.assertTrue(result.success)
        self.assertEqual(self.controller.config.target_temperature_c, 27.5)
        self.controller.save_current_config.assert_called_once()

    def test_save_analog_defaults_to_flash_applies_values_then_saves_and_waits(self) -> None:
        self.controller.apply_temperature_target = Mock(return_value=CommandResult(True, 0x04, "TEMP"))
        self.controller.apply_analog_outputs = Mock(return_value=CommandResult(True, 0x01, "DAC"))
        self.controller.flash_save_analog = Mock(return_value=CommandResult(True, CMD_FLASH_SAVE_ANALOG, "SENT"))
        self.controller._latest_status_seq = Mock(return_value=123)
        self.controller._wait_for_flash_idle = Mock(
            return_value=CommandResult(True, CMD_FLASH_SAVE_ANALOG, "Flash is idle")
        )

        result = self.controller.save_analog_defaults_to_flash(24.5, 110.0, 120.0, 130.0, 45.5, 1500, 30000)

        self.assertTrue(result.success)
        self.assertEqual(result.cmd_id, CMD_FLASH_SAVE_ANALOG)
        self.controller.apply_temperature_target.assert_called_once_with(24.5, 1500)
        self.controller.apply_analog_outputs.assert_called_once_with(110.0, 120.0, 130.0, 45.5, 1500)
        self.controller.flash_save_analog.assert_called_once_with(1500)
        self.controller._wait_for_flash_idle.assert_called_once_with(CMD_FLASH_SAVE_ANALOG, 123, 30000)

    def test_write_all_pixel_records_updates_service_records(self) -> None:
        records = [PixelParamRecord(addr=1, value36=2)]
        self.controller._pixel_array_service.set_records = Mock()
        self.controller._pixel_array_service.write_all_records = Mock(
            return_value=[CommandResult(True, 0x25, "SENT")]
        )

        results = self.controller.write_all_pixel_records(records, 2000)

        self.assertEqual(len(results), 1)
        self.controller._pixel_array_service.set_records.assert_called_once_with(records)
        self.controller._pixel_array_service.write_all_records.assert_called_once_with(2000)

    def test_configure_tdc_test_sends_upload_command_and_stores_result(self) -> None:
        result = CommandResult(True, 0x26, "SENT")
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=result)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller.save_current_config = Mock()
        self.controller.config.read_enabled = False

        snapshot = self.controller.configure_tdc_test(
            enabled=True,
            start_channel_ui=2,
            stop_channel_ui=3,
            refclk_divisions=12500,
            bin_width_raw=1,
            bin_offset=0,
            bin_count=1024,
            acquisition_time_s=2.5,
            source="synthetic",
            pairing_mode="all_start",
            tdc_deadtime_ps=2000.0,
        )

        self.assertIsNotNone(snapshot)
        self.controller._fpga_control_service.configure_tdc_test_upload.assert_has_calls(
            [
                call(False, 0, timeout_ms=1000),
                call(
                    True,
                    0x6,
                    timeout_ms=1000,
                    synthetic=True,
                    spi=False,
                    extended_timestamp_raw=False,
                ),
            ]
        )
        self.assertEqual(self.controller._fpga_control_service.configure_tdc_test_upload.call_count, 2)
        self.assertIs(self.controller.last_tdc_test_upload_result(), result)
        self.assertEqual(self.controller.config.tdc_test_settings.acquisition_time_s, 2.5)
        self.assertEqual(self.controller.config.tdc_test_settings.source, "synthetic")
        self.assertEqual(self.controller.config.tdc_test_settings.pairing_mode, "all_start")
        self.assertEqual(self.controller.config.tdc_test_settings.tdc_deadtime_ps, 2000.0)
        self.controller._device_service.set_read_enabled.assert_called_once_with(True)
        self.assertTrue(self.controller.config.read_enabled)

    def test_configure_tdc_test_gpx2_preflights_stable_default_config(self) -> None:
        gpx2_result = CommandResult(True, 0x11, "SENT")
        upload_result = CommandResult(True, 0x26, "SENT")
        self.controller._fpga_control_service.configure_gpx2_custom = Mock(return_value=gpx2_result)
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=upload_result)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller.save_current_config = Mock()
        self.controller.config.read_enabled = False

        snapshot = self.controller.configure_tdc_test(
            enabled=True,
            start_channel_ui=2,
            stop_channel_ui=3,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=300000,
            acquisition_time_s=30.0,
            source="gpx2",
        )

        self.assertIsNotNone(snapshot)
        self.controller._fpga_control_service.configure_gpx2_custom.assert_called_once_with(
            GPX2_DEFAULT_CONFIG_BYTES,
            timeout_ms=1000,
            profile=0,
            sequence_mode=10,
        )
        self.controller._fpga_control_service.configure_tdc_test_upload.assert_has_calls(
            [
                call(False, 0, timeout_ms=1000),
                call(
                    True,
                    0x6,
                    timeout_ms=1000,
                    synthetic=False,
                    spi=False,
                    extended_timestamp_raw=True,
                ),
            ]
        )
        self.assertEqual(self.controller._fpga_control_service.configure_tdc_test_upload.call_count, 2)
        self.assertIs(self.controller.last_tdc_test_upload_result(), upload_result)
        self.assertEqual(self.controller.config.tdc_test_settings.source, "gpx2")

    def test_configure_tdc_test_gpx2_fpga_uses_histogram_path(self) -> None:
        gpx2_result = CommandResult(True, 0x11, "SENT")
        pre_stop_result = CommandResult(True, 0x26, "SENT")
        hist_result = CommandResult(True, 0x27, "SENT")
        readout_result = CommandResult(True, 0x28, "SENT")
        self.controller._fpga_control_service.configure_gpx2_custom = Mock(return_value=gpx2_result)
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=pre_stop_result)
        self.controller._fpga_control_service.configure_tdc_histogram = Mock(return_value=hist_result)
        self.controller._fpga_control_service.request_tdc_histogram_readout = Mock(return_value=readout_result)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller.save_current_config = Mock()
        self.controller.config.read_enabled = False

        snapshot = self.controller.configure_tdc_test(
            enabled=True,
            start_channel_ui=2,
            stop_channel_ui=3,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=10,
            bin_count=50000,
            acquisition_time_s=30.0,
            source="gpx2_fpga",
            reference_deadtime_ns=800.0,
            tdc_deadtime_ps=2000.0,
        )

        self.assertIsNotNone(snapshot)
        self.controller._fpga_control_service.configure_tdc_test_upload.assert_called_once_with(
            False,
            0,
            timeout_ms=1000,
        )
        self.controller._fpga_control_service.configure_gpx2_custom.assert_called_once_with(
            GPX2_DEFAULT_CONFIG_BYTES,
            timeout_ms=1000,
            profile=0,
            sequence_mode=10,
        )
        self.controller._fpga_control_service.configure_tdc_histogram.assert_called_once_with(
            enable=True,
            clear=True,
            start_channel=1,
            stop_channel=2,
            bin_width_raw=5,
            bin_count=50000,
            offset_raw=10,
            reference_cleanup_raw=100000,
            tdc_deadtime_raw=250,
            refclk_divisions=12500,
            timeout_ms=1000,
        )
        self.controller._fpga_control_service.request_tdc_histogram_readout.assert_not_called()
        self.assertIs(self.controller.last_tdc_test_upload_result(), hist_result)
        self.assertEqual(self.controller.config.tdc_test_settings.source, "gpx2_fpga")

    def test_fpga_hist_live_readout_pauses_before_requesting_chunks(self) -> None:
        hist_result = CommandResult(True, 0x27, "SENT")
        readout_result = CommandResult(True, 0x28, "SENT")
        self.controller._fpga_control_service.configure_tdc_histogram = Mock(return_value=hist_result)
        self.controller._fpga_control_service.request_tdc_histogram_readout = Mock(return_value=readout_result)
        settings = self.controller.config.tdc_test_settings
        settings.enabled = True
        settings.source = "gpx2_fpga"
        settings.start_channel = 1
        settings.stop_channel = 2
        settings.bin_width_raw = 5
        settings.bin_count = 1024
        settings.bin_offset = 10
        settings.reference_deadtime_ns = 0.0
        settings.tdc_deadtime_ps = 0.0
        settings.refclk_divisions = 12500
        self.controller.config.tdc_test_settings = settings
        self.controller._tdc_hist_readout_chunks_per_tick = 2

        self.controller._request_fpga_hist_readout(timeout_ms=1000, full=False)

        self.assertEqual(self.controller._fpga_control_service.configure_tdc_histogram.call_count, 2)
        self.controller._fpga_control_service.configure_tdc_histogram.assert_has_calls(
            [
                call(
                    enable=False,
                    clear=False,
                    start_channel=1,
                    stop_channel=2,
                    bin_width_raw=5,
                    bin_count=1024,
                    offset_raw=10,
                    reference_cleanup_raw=0,
                    tdc_deadtime_raw=0,
                    refclk_divisions=12500,
                    timeout_ms=1000,
                ),
                call(
                    enable=True,
                    clear=False,
                    start_channel=1,
                    stop_channel=2,
                    bin_width_raw=5,
                    bin_count=1024,
                    offset_raw=10,
                    reference_cleanup_raw=0,
                    tdc_deadtime_raw=0,
                    refclk_divisions=12500,
                    timeout_ms=1000,
                ),
            ]
        )
        self.controller._fpga_control_service.request_tdc_histogram_readout.assert_has_calls(
            [
                call(0x1, timeout_ms=1000, chunk_index=0),
                call(0x1, timeout_ms=1000, chunk_index=1),
            ]
        )

    def test_configure_tdc_test_restarts_stale_rx_reader_when_config_enabled(self) -> None:
        result = CommandResult(True, 0x26, "SENT")
        self.controller.config.read_enabled = True
        self.controller._device_service.is_rx_reader_running = Mock(return_value=False)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=result)
        self.controller.save_current_config = Mock()

        self.controller.configure_tdc_test(
            enabled=True,
            start_channel_ui=2,
            stop_channel_ui=3,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=1024,
            acquisition_time_s=1.0,
            source="synthetic",
        )

        self.controller._device_service.set_read_enabled.assert_called_once_with(True)
        self.assertTrue(self.controller.config.read_enabled)
        self.assertFalse(self.controller._tdc_test_started_rx)

    def test_stop_tdc_test_upload_disables_without_clearing_builder(self) -> None:
        result = CommandResult(True, 0x26, "SENT")
        self.controller.config.tdc_test_settings.enabled = True
        self.controller.config.tdc_test_settings.source = "synthetic"
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=result)
        self.controller.save_current_config = Mock()

        stop_result = self.controller.stop_tdc_test_upload()

        self.assertIs(stop_result, result)
        self.assertFalse(self.controller.config.tdc_test_settings.enabled)
        self.controller._fpga_control_service.configure_tdc_test_upload.assert_called_once_with(
            False,
            0,
            timeout_ms=1000,
        )

    def test_stop_tdc_test_upload_restores_temp_rx_reader(self) -> None:
        result = CommandResult(True, 0x26, "SENT")
        self.controller.config.tdc_test_settings.enabled = True
        self.controller.config.tdc_test_settings.source = "synthetic"
        self.controller.config.read_enabled = True
        self.controller._tdc_test_started_rx = True
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=result)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller.save_current_config = Mock()

        self.controller.stop_tdc_test_upload()

        self.controller._device_service.set_read_enabled.assert_called_once_with(False)
        self.assertFalse(self.controller.config.read_enabled)
        self.assertFalse(self.controller._tdc_test_started_rx)

    def test_configure_tdc_test_gpx2_spi_uses_example_fpga_path(self) -> None:
        upload_result = CommandResult(True, 0x26, "SENT")
        self.controller._fpga_control_service.configure_gpx2 = Mock()
        self.controller._fpga_control_service.configure_tdc_test_upload = Mock(return_value=upload_result)
        self.controller._device_service.set_read_enabled = Mock()
        self.controller.save_current_config = Mock()
        self.controller.config.read_enabled = False

        snapshot = self.controller.configure_tdc_test(
            enabled=True,
            start_channel_ui=2,
            stop_channel_ui=3,
            refclk_divisions=12500,
            bin_width_raw=5,
            bin_offset=0,
            bin_count=300000,
            acquisition_time_s=30.0,
            source="gpx2_spi",
        )

        self.assertIsNotNone(snapshot)
        self.controller._fpga_control_service.configure_gpx2.assert_not_called()
        self.controller._fpga_control_service.configure_tdc_test_upload.assert_has_calls(
            [
                call(False, 0, timeout_ms=1000),
                call(
                    True,
                    0x6,
                    timeout_ms=1000,
                    synthetic=False,
                    spi=True,
                    extended_timestamp_raw=False,
                ),
            ]
        )
        self.assertEqual(self.controller._fpga_control_service.configure_tdc_test_upload.call_count, 2)
        self.assertEqual(self.controller.config.tdc_test_settings.source, "gpx2_spi")
