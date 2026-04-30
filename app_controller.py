from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5 import QtCore

from app.data_processing import AcquisitionService, AnalysisService
from app.fpga_control import AnalogControlService, FlashService, FpgaControlService, PixelArrayService, TemperatureControlService
from app.models import AppConfig, CommandResult, HistogramSnapshot, PixelParamRecord, StatusFlagsDecoded
from app.protocol import LegacyAnalogCodec
from app.storage import AnalysisExportService, ConfigRepository, PixelArrayRepository
from app.usb_link import DeviceInfo, DeviceService


class AppController(QtCore.QObject):
    device_state_changed = QtCore.pyqtSignal(bool, str)
    log_message = QtCore.pyqtSignal(str)
    stats_updated = QtCore.pyqtSignal(object, object)
    status_received = QtCore.pyqtSignal(object)
    histogram_updated = QtCore.pyqtSignal(object)
    recording_state_changed = QtCore.pyqtSignal(bool, str)
    replay_completed = QtCore.pyqtSignal(object)
    temperature_target_changed = QtCore.pyqtSignal(float, int)
    analog_targets_changed = QtCore.pyqtSignal(object, object)
    read_state_changed = QtCore.pyqtSignal(bool)

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent)
        self.window = window
        self.default_config_path = Path(__file__).resolve().parent / "default_config.json"
        self.config = self._load_config()

        self._device_service = DeviceService(self)
        self._analog_codec = LegacyAnalogCodec(
            threshold_range_mv=self.config.threshold_range_mv,
            bias_scale=self.config.bias_scale,
        )
        self._temperature_service = TemperatureControlService(self._device_service, self._analog_codec, self)
        self._analog_service = AnalogControlService(self._device_service, self._analog_codec, self)
        self._fpga_control_service = FpgaControlService(self._device_service, self)
        self._pixel_array_service = PixelArrayService(self._device_service, self)
        self._flash_service = FlashService(self._device_service, self)
        self._acquisition_service = AcquisitionService(self._device_service, self.config, self)
        self._analysis_service = AnalysisService(self.config, self)

        self._apply_config_to_services()
        self._bind_internal_signals()
        self.window.bind_controller(self)

    def _bind_internal_signals(self) -> None:
        self._device_service.device_state_changed.connect(self.device_state_changed.emit)
        self._device_service.log_message.connect(self.log_message.emit)
        self._device_service.stats_updated.connect(self.stats_updated.emit)
        self._device_service.read_state_changed.connect(self.read_state_changed.emit)
        self._device_service.status_received.connect(self.status_received.emit)
        self._temperature_service.target_changed.connect(self.temperature_target_changed.emit)
        self._analog_service.analog_targets_changed.connect(self.analog_targets_changed.emit)
        self._acquisition_service.histogram_updated.connect(self.histogram_updated.emit)
        self._acquisition_service.recording_state_changed.connect(self.recording_state_changed.emit)
        self._analysis_service.replay_completed.connect(self.replay_completed.emit)
        self.log_message.emit(
            f"Host app build active: {Path(__file__).resolve().parent} | usb_debug_logging=enabled"
        )

    def _load_config(self) -> AppConfig:
        if self.default_config_path.exists():
            return ConfigRepository.load(self.default_config_path)
        return AppConfig()

    def save_current_config(self) -> None:
        ConfigRepository.save(self.default_config_path, self.config)

    def _apply_config_to_services(self) -> None:
        self._temperature_service.target_temperature_c = self.config.target_temperature_c
        self._temperature_service.target_temperature_code = self._analog_codec.temperature_c_to_code(
            self.config.target_temperature_c
        )

        analog = self.config.analog_targets
        self._analog_service.targets.laser_sync_threshold_mv = analog.laser_sync_threshold_mv
        self._analog_service.targets.pixel_sync_threshold_mv = analog.pixel_sync_threshold_mv
        self._analog_service.targets.avalanche_threshold_mv = analog.avalanche_threshold_mv
        self._analog_service.targets.bias_voltage_v = analog.bias_voltage_v
        self._analog_service.codes = self._analog_codec.encode_analog_targets(
            analog.laser_sync_threshold_mv,
            analog.pixel_sync_threshold_mv,
            analog.avalanche_threshold_mv,
            analog.bias_voltage_v,
        )
        self._fpga_control_service.gate_settings = self.config.gate_settings

    def refresh_devices(self) -> list[DeviceInfo]:
        return self._device_service.enumerate_devices()

    def connect_device(
        self,
        device_index: int,
        read_pipe: int,
        write_pipe: int,
        read_block_size: int,
        read_enabled: bool | None = None,
    ) -> None:
        if read_enabled is None:
            read_enabled = self.config.read_enabled
        self._device_service.open_device(
            device_index=device_index,
            read_pipe=read_pipe,
            write_pipe=write_pipe,
            read_block_size=read_block_size,
            read_enabled=read_enabled,
        )
        self.config.read_enabled = bool(read_enabled)

    def disconnect_device(self) -> None:
        self._device_service.close_device()

    def set_read_enabled(self, enabled: bool) -> None:
        self.config.read_enabled = bool(enabled)
        self._device_service.set_read_enabled(enabled)
        self.save_current_config()

    def run_basic_self_test(self) -> list[str]:
        results: list[str] = []

        if self._device_service.device is None:
            return ["Self-test skipped: device not connected."]

        if self._device_service.wait_for_status_available(1500):
            results.append("STATUS receive: PASS")
        else:
            results.append("STATUS receive: TIMEOUT")

        gpx2_result = self.configure_gpx2(1500)
        results.append(
            "GPX2 config send: " + ("PASS" if gpx2_result.success else f"FAIL ({gpx2_result.message})")
        )

        temp_result = self.apply_temperature_target(self.config.target_temperature_c, 1500)
        results.append(
            "TEC setpoint send: "
            + ("PASS" if temp_result.success else f"FAIL ({temp_result.message})")
        )

        analog = self.config.analog_targets
        analog_result = self.apply_analog_outputs(
            analog.laser_sync_threshold_mv,
            analog.pixel_sync_threshold_mv,
            analog.avalanche_threshold_mv,
            analog.bias_voltage_v,
            1500,
        )
        results.append(
            "AD5686 analog send: "
            + ("PASS" if analog_result.success else f"FAIL ({analog_result.message})")
        )
        return results

    def apply_temperature_target(self, temp_c: float, timeout_ms: int = 1000) -> CommandResult:
        result = self._temperature_service.set_target_temperature_c(temp_c, timeout_ms)
        if result.success:
            self.config.target_temperature_c = float(temp_c)
            self.save_current_config()
        return result

    def apply_analog_outputs(
        self,
        laser_mv: float,
        pixel_mv: float,
        avalanche_mv: float,
        bias_v: float,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        result = self._analog_service.set_outputs(laser_mv, pixel_mv, avalanche_mv, bias_v, timeout_ms)
        if result.success:
            self.config.analog_targets.laser_sync_threshold_mv = float(laser_mv)
            self.config.analog_targets.pixel_sync_threshold_mv = float(pixel_mv)
            self.config.analog_targets.avalanche_threshold_mv = float(avalanche_mv)
            self.config.analog_targets.bias_voltage_v = float(bias_v)
            self.save_current_config()
        return result

    def configure_gpx2(self, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.configure_gpx2(timeout_ms)

    def flash_save(self, timeout_ms: int = 1000) -> CommandResult:
        return self._flash_service.flash_save(timeout_ms)

    def flash_load(self, timeout_ms: int = 1000) -> CommandResult:
        return self._flash_service.flash_load(timeout_ms)

    def set_gate_holdoff(self, holdoff: int, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.set_gate_holdoff(holdoff, timeout_ms)

    def set_gate_div(self, divider: int, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.set_gate_div(divider, timeout_ms)

    def set_nb6(self, delay_a: int, delay_b: int, enable: bool, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.set_nb6(delay_a, delay_b, enable, timeout_ms)

    def set_gate_enable(
        self,
        sig2_enable: bool,
        sig3_enable: bool,
        pixel_mode: bool,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        return self._fpga_control_service.set_gate_enable(sig2_enable, sig3_enable, pixel_mode, timeout_ms)

    def set_gate_signal(
        self,
        signal_id: int,
        delay_coarse: int,
        delay_fine: int,
        width_coarse: int,
        width_fine: int,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        return self._fpga_control_service.set_gate_signal(
            signal_id,
            delay_coarse,
            delay_fine,
            width_coarse,
            width_fine,
            timeout_ms,
        )

    def set_pixel_records(self, records: list[PixelParamRecord]) -> None:
        self._pixel_array_service.set_records(records)

    def load_pixel_records(self, path: str | Path, fmt: str) -> list[PixelParamRecord]:
        if fmt == "json":
            return PixelArrayRepository.load_json(path)
        return PixelArrayRepository.load_csv(path)

    def save_pixel_records(self, path: str | Path, records: list[PixelParamRecord], fmt: str) -> None:
        if fmt == "json":
            PixelArrayRepository.save_json(path, records)
        else:
            PixelArrayRepository.save_csv(path, records)

    def write_pixel_param(self, addr: int, value36: int, timeout_ms: int = 1000) -> CommandResult:
        return self._pixel_array_service.write_pixel_param(addr, value36, timeout_ms)

    def write_all_pixel_records(self, records: list[PixelParamRecord], timeout_ms: int = 1000) -> list[CommandResult]:
        self._pixel_array_service.set_records(records)
        return self._pixel_array_service.write_all_records(timeout_ms)

    def reload_pixel_params(self, timeout_ms: int = 1000) -> CommandResult:
        return self._pixel_array_service.reload_pixel_params(timeout_ms)

    def start_recording(self, session_name: str = "capture", notes: str = "") -> Path:
        return self._acquisition_service.start_recording(session_name, notes)

    def stop_recording(self) -> None:
        self._acquisition_service.stop_recording()

    def replay_session(self, session_json_path: str | Path) -> HistogramSnapshot:
        return self._analysis_service.replay_session(session_json_path)

    def current_snapshot(self) -> HistogramSnapshot:
        return self._acquisition_service.current_snapshot()

    def export_histogram_csv(self, path: str | Path) -> bool:
        snapshot = self.current_snapshot()
        if not snapshot.histograms:
            return False
        key = f"{snapshot.current_row},{snapshot.current_col}"
        hist = snapshot.histograms.get(key)
        if hist is None:
            hist = next(iter(snapshot.histograms.values()))
        AnalysisExportService.export_histogram_csv(path, np.array(hist, dtype=np.uint32))
        return True

    @staticmethod
    def decode_status_flags(flags: int) -> StatusFlagsDecoded:
        return DeviceService.decode_status_flags(flags)
