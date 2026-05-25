from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt5 import QtCore

from app.data_processing import AcquisitionService, AnalysisService
from app.fpga_control import AnalogControlService, FlashService, FpgaControlService, PixelArrayService, TemperatureControlService
from app.models import AppConfig, CommandResult, HistogramSnapshot, PixelParamRecord, StatusFlagsDecoded, TdcTestSettings
from app.protocol import (
    CMD_FLASH_SAVE_ANALOG,
    GPX2_DEFAULT_CONFIG_BYTES,
    GPX2_DEFAULT_PROFILE,
    GPX2_DEFAULT_SEQUENCE,
    LegacyAnalogCodec,
)
from app.storage import AnalysisExportService, ConfigRepository, PixelArrayRepository
from app.usb_link import DeviceInfo, DeviceService


class AppController(QtCore.QObject):
    device_state_changed = QtCore.pyqtSignal(bool, str)
    log_message = QtCore.pyqtSignal(str)
    stats_updated = QtCore.pyqtSignal(object, object)
    status_received = QtCore.pyqtSignal(object)
    histogram_updated = QtCore.pyqtSignal(object)
    tdc_test_histogram_updated = QtCore.pyqtSignal(object)
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
        self._last_tdc_test_upload_result: CommandResult | None = None
        self._tdc_test_started_rx = False
        self._tdc_hist_readout_timer = QtCore.QTimer(self)
        self._tdc_hist_readout_timer.setInterval(250)
        self._tdc_hist_readout_timer.timeout.connect(self._request_fpga_hist_readout)
        self._tdc_hist_readout_chunk_index = 0
        self._tdc_hist_readout_chunks_per_tick = 2
        self._tdc_hist_readout_priority_chunks: list[int] = []

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
        self._acquisition_service.tdc_test_histogram_updated.connect(self.tdc_test_histogram_updated.emit)
        self._acquisition_service.recording_state_changed.connect(self.recording_state_changed.emit)
        self._analysis_service.replay_completed.connect(self.replay_completed.emit)
        self.log_message.emit(
            f"Host app build active: {Path(__file__).resolve().parent} | usb_debug_logging=summary_only"
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

    def _rx_reader_running(self) -> bool:
        checker = getattr(self._device_service, "is_rx_reader_running", None)
        if callable(checker):
            return bool(checker())
        return bool(self.config.read_enabled)

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

    def configure_gpx2(
        self,
        timeout_ms: int = 1000,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> CommandResult:
        return self._fpga_control_service.configure_gpx2(timeout_ms, profile, sequence_mode)

    def configure_gpx2_custom(
        self,
        config_bytes,
        timeout_ms: int = 1000,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> CommandResult:
        return self._fpga_control_service.configure_gpx2_custom(
            config_bytes,
            timeout_ms=timeout_ms,
            profile=profile,
            sequence_mode=sequence_mode,
        )

    def configure_gpx2_lvds_test(self, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.configure_gpx2_lvds_test(timeout_ms)

    def reset_gpx2_power_opcode(self, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.reset_gpx2_power_opcode(timeout_ms)

    def read_gpx2_config(self, timeout_ms: int = 1000) -> CommandResult:
        return self._fpga_control_service.read_gpx2_config(timeout_ms)

    def flash_save(self, timeout_ms: int = 1000) -> CommandResult:
        return self._flash_service.flash_save(timeout_ms)

    def flash_load(self, timeout_ms: int = 1000) -> CommandResult:
        return self._flash_service.flash_load(timeout_ms)

    def flash_save_analog(self, timeout_ms: int = 1000) -> CommandResult:
        return self._flash_service.flash_save_analog(timeout_ms)

    def save_analog_defaults_to_flash(
        self,
        temp_c: float,
        laser_mv: float,
        pixel_mv: float,
        avalanche_mv: float,
        bias_v: float,
        command_timeout_ms: int = 1500,
        flash_timeout_ms: int = 30000,
    ) -> CommandResult:
        temp_result = self.apply_temperature_target(temp_c, command_timeout_ms)
        if not temp_result.success:
            return temp_result

        analog_result = self.apply_analog_outputs(
            laser_mv,
            pixel_mv,
            avalanche_mv,
            bias_v,
            command_timeout_ms,
        )
        if not analog_result.success:
            return analog_result

        start_seq = self._latest_status_seq()
        save_result = self.flash_save_analog(command_timeout_ms)
        if not save_result.success:
            return save_result

        idle_result = self._wait_for_flash_idle(CMD_FLASH_SAVE_ANALOG, start_seq, flash_timeout_ms)
        if idle_result.success:
            return CommandResult(True, CMD_FLASH_SAVE_ANALOG, "Analog/TEC defaults saved to Flash")
        return idle_result

    def _latest_status_seq(self) -> int | None:
        status = getattr(self._device_service, "latest_status", None)
        if status is None:
            return None
        return int(status.header.seq)

    def _wait_for_flash_idle(self, cmd_id: int, start_seq: int | None, timeout_ms: int) -> CommandResult:
        deadline = time.monotonic() + (max(1, int(timeout_ms)) / 1000.0)
        seen_new_status = start_seq is None
        seen_busy = False

        while True:
            status = getattr(self._device_service, "latest_status", None)
            if status is not None:
                status_seq = int(status.header.seq)
                if start_seq is None or status_seq != start_seq:
                    seen_new_status = True
                decoded = DeviceService.decode_status_flags(status.flags)
                if decoded.flash_error:
                    return CommandResult(False, cmd_id, "Flash error asserted while saving analog defaults")
                if decoded.flash_busy:
                    seen_busy = True
                elif seen_busy or seen_new_status:
                    return CommandResult(True, cmd_id, "Flash is idle")

            now = time.monotonic()
            if now >= deadline:
                return CommandResult(False, cmd_id, "Timed out waiting for flash_busy to clear")
            time.sleep(min(0.05, deadline - now))

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

    def set_gate_sig3_long_delay(
        self,
        delay_ns: int,
        width_coarse: int,
        width_fine: int,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        return self._fpga_control_service.set_gate_sig3_long_delay(
            delay_ns,
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

    def current_tdc_test_snapshot(self) -> HistogramSnapshot:
        return self._acquisition_service.current_tdc_test_snapshot()

    def last_tdc_test_upload_result(self) -> CommandResult | None:
        return self._last_tdc_test_upload_result

    def configure_tdc_test(
        self,
        enabled: bool,
        start_channel_ui: int,
        stop_channel_ui: int,
        refclk_divisions: int,
        bin_width_raw: int,
        bin_offset: int,
        bin_count: int,
        acquisition_time_s: float | None = None,
        source: str | None = None,
        pairing_mode: str | None = None,
        tdc_deadtime_ps: float | None = None,
        reference_deadtime_ns: float | None = None,
    ) -> HistogramSnapshot:
        if acquisition_time_s is None:
            acquisition_time_s = self.config.tdc_test_settings.acquisition_time_s
        if source is None:
            source = self.config.tdc_test_settings.source
        if pairing_mode is None:
            pairing_mode = getattr(self.config.tdc_test_settings, "pairing_mode", "prev_start")
        if tdc_deadtime_ps is None:
            tdc_deadtime_ps = getattr(self.config.tdc_test_settings, "tdc_deadtime_ps", 0.0)
        if reference_deadtime_ns is None:
            reference_deadtime_ns = getattr(
                self.config.tdc_test_settings,
                "reference_deadtime_ns",
                800.0,
            )
        source = str(source).lower()
        if source not in {"gpx2_fpga", "gpx2", "gpx2_spi", "synthetic"}:
            source = "gpx2_fpga"
        pairing_mode = str(pairing_mode).lower()
        if pairing_mode not in {"prev_start", "all_start"}:
            pairing_mode = "prev_start"
        raw_epoch = 0
        if bool(enabled):
            pre_stop = self._fpga_control_service.configure_tdc_test_upload(False, 0, timeout_ms=1000)
            if not pre_stop.success:
                self.log_message.emit(f"TCSPC pre-stop previous upload failed: {pre_stop.message}")
            if source == "gpx2_fpga":
                raw_epoch = 0
            else:
                raw_epoch = self._device_service.begin_tdc_raw_epoch(
                    restart_reader=True,
                    flush_pipe=(source == "gpx2"),
                )
        settings = TdcTestSettings(
            enabled=bool(enabled),
            start_channel=max(0, min(3, int(start_channel_ui) - 1)),
            stop_channel=max(0, min(3, int(stop_channel_ui) - 1)),
            source=source,
            pairing_mode=pairing_mode,
            refclk_divisions=max(1, int(refclk_divisions)),
            bin_width_raw=max(1, int(bin_width_raw)),
            bin_offset=int(bin_offset),
            bin_count=max(1, int(bin_count)),
            acquisition_time_s=max(0.0, float(acquisition_time_s)),
            tdc_deadtime_ps=max(0.0, float(tdc_deadtime_ps)),
            reference_deadtime_ns=max(0.0, float(reference_deadtime_ns)),
            detector_deadtime_ns=getattr(self.config.tdc_test_settings, "detector_deadtime_ns", 100.0),
            raw_epoch=raw_epoch,
        )
        self.config.tdc_test_settings = settings
        self.save_current_config()
        snapshot = self._acquisition_service.configure_tdc_test(settings)
        channel_mask = 0
        if settings.enabled:
            desired_rx_before = bool(self.config.read_enabled)
            if not self._rx_reader_running():
                if desired_rx_before:
                    self.log_message.emit("TDC test found FT601 RX reader stale; restarting data read.")
                else:
                    self.log_message.emit("TDC test requires FT601 RX reader; enabling data read.")
                    self._tdc_test_started_rx = True
                self.set_read_enabled(True)
            channel_mask = (1 << settings.start_channel) | (1 << settings.stop_channel)
            if settings.source == "gpx2":
                profile = GPX2_DEFAULT_PROFILE
                sequence = GPX2_DEFAULT_SEQUENCE
                self.log_message.emit(
                    "TDC GPX2 raw source selected: applying max high-resolution "
                    f"custom config first (profile {profile}, sequence {sequence})."
                )
                gpx2_result = self._fpga_control_service.configure_gpx2_custom(
                    GPX2_DEFAULT_CONFIG_BYTES,
                    timeout_ms=1000,
                    profile=profile,
                    sequence_mode=sequence,
                )
                self.log_message.emit(
                    "TDC GPX2 preflight config: "
                    f"{'SENT' if gpx2_result.success else 'FAILED'} ({gpx2_result.message})"
                )
            elif settings.source == "gpx2_spi":
                self.log_message.emit(
                    "TDC GPX2 SPI source selected: using FPGA example-style "
                    "0x30/write/read/0x18 sequence with LVDS STOP input."
                )
            elif settings.source == "gpx2_fpga":
                profile = GPX2_DEFAULT_PROFILE
                sequence = GPX2_DEFAULT_SEQUENCE
                self.log_message.emit(
                    "TDC GPX2 FPGA histogram source selected: applying high-resolution "
                    f"GPX2 config first (profile {profile}, sequence {sequence})."
                )
                gpx2_result = self._fpga_control_service.configure_gpx2_custom(
                    GPX2_DEFAULT_CONFIG_BYTES,
                    timeout_ms=1000,
                    profile=profile,
                    sequence_mode=sequence,
                )
                self.log_message.emit(
                    "TDC GPX2 FPGA histogram preflight config: "
                    f"{'SENT' if gpx2_result.success else 'FAILED'} ({gpx2_result.message})"
                )
                hist_result = self._fpga_control_service.configure_tdc_histogram(
                    enable=True,
                    clear=True,
                    start_channel=settings.start_channel,
                    stop_channel=settings.stop_channel,
                    bin_width_raw=settings.bin_width_raw,
                    bin_count=settings.bin_count,
                    offset_raw=settings.bin_offset,
                    reference_cleanup_raw=int(round(settings.reference_deadtime_ns * 125.0)),
                    tdc_deadtime_raw=int(round(settings.tdc_deadtime_ps / 8.0)),
                    refclk_divisions=settings.refclk_divisions,
                    timeout_ms=1000,
                )
                self._last_tdc_test_upload_result = hist_result
                self._tdc_hist_readout_chunk_index = 0
                self._tdc_hist_readout_priority_chunks = self._initial_tdc_hist_priority_chunks(settings.bin_count)
                if settings.enabled:
                    self._tdc_hist_readout_timer.start()
                else:
                    self._tdc_hist_readout_timer.stop()
        if settings.source == "gpx2_fpga":
            if not settings.enabled:
                self._tdc_hist_readout_timer.stop()
            result = self._last_tdc_test_upload_result or CommandResult(False, 0, "not started")
        else:
            self._tdc_hist_readout_timer.stop()
            result = self._fpga_control_service.configure_tdc_test_upload(
                settings.enabled,
                channel_mask,
                timeout_ms=1000,
                synthetic=(settings.source == "synthetic"),
                spi=(settings.source == "gpx2_spi"),
                extended_timestamp_raw=(settings.source == "gpx2"),
            )
            self._last_tdc_test_upload_result = result
        self.log_message.emit(
            "TCSPC configured: "
            f"enabled={settings.enabled} reference=CH{settings.start_channel + 1} "
            f"photon=CH{settings.stop_channel + 1} bins={settings.bin_count} "
            f"source={settings.source} pairing={settings.pairing_mode} "
            f"deadtime={settings.tdc_deadtime_ps:g} ps "
            f"ref_cleanup={settings.reference_deadtime_ns:g} ns "
            f"fpga_upload={'on' if result.success and settings.enabled else 'off'}"
        )
        return snapshot

    def set_tdc_test_acquisition_time(self, seconds: float) -> None:
        self.config.tdc_test_settings.acquisition_time_s = max(0.0, float(seconds))
        self.save_current_config()

    def stop_tdc_test_upload(self, timeout_ms: int = 1000) -> CommandResult:
        settings = self.config.tdc_test_settings
        settings.enabled = False
        self.config.tdc_test_settings = settings
        self.save_current_config()
        self._tdc_hist_readout_timer.stop()
        if settings.source == "gpx2_fpga":
            result = self._fpga_control_service.configure_tdc_histogram(
                enable=False,
                clear=False,
                start_channel=settings.start_channel,
                stop_channel=settings.stop_channel,
                bin_width_raw=settings.bin_width_raw,
                bin_count=settings.bin_count,
                offset_raw=settings.bin_offset,
                reference_cleanup_raw=int(round(settings.reference_deadtime_ns * 125.0)),
                tdc_deadtime_raw=int(round(settings.tdc_deadtime_ps / 8.0)),
                refclk_divisions=settings.refclk_divisions,
                timeout_ms=timeout_ms,
            )
            self._request_fpga_hist_readout(timeout_ms=timeout_ms, full=True)
        else:
            result = self._fpga_control_service.configure_tdc_test_upload(False, 0, timeout_ms=timeout_ms)
        self._last_tdc_test_upload_result = result
        if self._tdc_test_started_rx:
            self.set_read_enabled(False)
            self._tdc_test_started_rx = False
        self.log_message.emit(
            "TCSPC stopped: "
            f"fpga_upload={'off' if result.success else 'stop_failed'}"
        )
        return result

    def clear_tdc_test_histogram(self) -> HistogramSnapshot:
        snapshot = self._acquisition_service.clear_tdc_test()
        settings = self.config.tdc_test_settings
        if settings.source == "gpx2_fpga":
            self._fpga_control_service.configure_tdc_histogram(
                enable=settings.enabled,
                clear=True,
                start_channel=settings.start_channel,
                stop_channel=settings.stop_channel,
                bin_width_raw=settings.bin_width_raw,
                bin_count=settings.bin_count,
                offset_raw=settings.bin_offset,
                reference_cleanup_raw=int(round(settings.reference_deadtime_ns * 125.0)),
                tdc_deadtime_raw=int(round(settings.tdc_deadtime_ps / 8.0)),
                refclk_divisions=settings.refclk_divisions,
                timeout_ms=1000,
            )
            self._tdc_hist_readout_chunk_index = 0
            self._tdc_hist_readout_priority_chunks = self._initial_tdc_hist_priority_chunks(settings.bin_count)
            self._request_fpga_hist_readout(full=not settings.enabled)
        self.log_message.emit("TCSPC histogram cleared.")
        return snapshot

    @staticmethod
    def _initial_tdc_hist_priority_chunks(bin_count: int) -> list[int]:
        chunks = max(1, (int(bin_count) + 255) // 256)
        # Paint the low-delay area immediately after Start/Clear. The normal
        # round-robin scan below still visits the whole histogram.
        return [idx for idx in range(min(8, chunks))]

    def _request_fpga_hist_readout(self, timeout_ms: int = 1000, full: bool = False) -> None:
        settings = self.config.tdc_test_settings
        if settings.source != "gpx2_fpga":
            return
        if not settings.enabled and not full:
            return
        if full:
            result = self._fpga_control_service.request_tdc_histogram_readout(
                0x1,
                timeout_ms=timeout_ms,
                chunk_index=None,
            )
            if not result.success:
                self.log_message.emit(f"FPGA histogram readout request failed: {result.message}")
            return

        pause_result = self._fpga_control_service.configure_tdc_histogram(
            enable=False,
            clear=False,
            start_channel=settings.start_channel,
            stop_channel=settings.stop_channel,
            bin_width_raw=settings.bin_width_raw,
            bin_count=settings.bin_count,
            offset_raw=settings.bin_offset,
            reference_cleanup_raw=int(round(settings.reference_deadtime_ns * 125.0)),
            tdc_deadtime_raw=int(round(settings.tdc_deadtime_ps / 8.0)),
            refclk_divisions=settings.refclk_divisions,
            timeout_ms=timeout_ms,
        )
        if not pause_result.success:
            self.log_message.emit(f"FPGA histogram live readout pause failed: {pause_result.message}")
            return

        chunks = max(1, (int(settings.bin_count) + 255) // 256)
        request_budget = min(max(1, int(self._tdc_hist_readout_chunks_per_tick)), chunks)
        requested: set[int] = set()
        request_list: list[int] = []

        while self._tdc_hist_readout_priority_chunks and len(request_list) < request_budget:
            chunk_index = int(self._tdc_hist_readout_priority_chunks.pop(0)) % chunks
            if chunk_index not in requested:
                requested.add(chunk_index)
                request_list.append(chunk_index)

        while len(request_list) < request_budget:
            chunk_index = int(self._tdc_hist_readout_chunk_index) % chunks
            self._tdc_hist_readout_chunk_index = (chunk_index + 1) % chunks
            if chunk_index not in requested:
                requested.add(chunk_index)
                request_list.append(chunk_index)

        for chunk_index in request_list:
            result = self._fpga_control_service.request_tdc_histogram_readout(
                0x1,
                timeout_ms=timeout_ms,
                chunk_index=chunk_index,
            )
            if not result.success:
                self.log_message.emit(
                    f"FPGA histogram chunk {chunk_index} readout request failed: {result.message}"
                )

        resume_result = self._fpga_control_service.configure_tdc_histogram(
            enable=True,
            clear=False,
            start_channel=settings.start_channel,
            stop_channel=settings.stop_channel,
            bin_width_raw=settings.bin_width_raw,
            bin_count=settings.bin_count,
            offset_raw=settings.bin_offset,
            reference_cleanup_raw=int(round(settings.reference_deadtime_ns * 125.0)),
            tdc_deadtime_raw=int(round(settings.tdc_deadtime_ps / 8.0)),
            refclk_divisions=settings.refclk_divisions,
            timeout_ms=timeout_ms,
        )
        if not resume_result.success:
            self.log_message.emit(f"FPGA histogram live readout resume failed: {resume_result.message}")

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
