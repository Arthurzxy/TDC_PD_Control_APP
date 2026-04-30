from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PyQt5 import QtCore

from app.app_controller import AppController
from app.models import CommandResult, PixelParamRecord
from app.usb_link import DeviceInfo


class DummyWindow:
    def __init__(self) -> None:
        self.bound_controller = None

    def bind_controller(self, controller) -> None:
        self.bound_controller = controller


class AppControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

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
