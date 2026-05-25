from __future__ import annotations

from typing import List

from PyQt5 import QtCore

import app.protocol as protocol
from app.models import AnalogTargets, CommandResult, GateSettings, PixelParamRecord
from app.protocol import CommandEncoder, GPX2_DEFAULT_PROFILE, GPX2_DEFAULT_SEQUENCE, LegacyAnalogCodec
from app.usb_link import DeviceService


class TemperatureControlService(QtCore.QObject):
    target_changed = QtCore.pyqtSignal(float, int)

    def __init__(self, device_service: DeviceService, analog_codec: LegacyAnalogCodec, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.encoder = CommandEncoder()
        self.analog_codec = analog_codec
        self.target_temperature_c = 25.0
        self.target_temperature_code = self.analog_codec.temperature_c_to_code(self.target_temperature_c)

    def set_target_temperature_c(self, temp_c: float, timeout_ms: int = 1000) -> CommandResult:
        self.target_temperature_c = float(temp_c)
        self.target_temperature_code = self.analog_codec.temperature_c_to_code(self.target_temperature_c)
        frame = self.encoder.encode_set_temperature_code(self.target_temperature_code)
        result = self.device_service.send_command_sync(
            protocol.CMD_TEC_PID,
            frame,
            timeout_ms,
            debug_details=(
                f"target_temp_c={self.target_temperature_c:.2f}, "
                f"temp_code={self.target_temperature_code} (0x{self.target_temperature_code:04X})"
            ),
        )
        if result.success:
            self.target_changed.emit(self.target_temperature_c, self.target_temperature_code)
        return result


class AnalogControlService(QtCore.QObject):
    analog_targets_changed = QtCore.pyqtSignal(object, object)

    def __init__(self, device_service: DeviceService, analog_codec: LegacyAnalogCodec, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.analog_codec = analog_codec
        self.encoder = CommandEncoder()
        self.targets = AnalogTargets()
        self.codes = self.analog_codec.encode_analog_targets(
            self.targets.laser_sync_threshold_mv,
            self.targets.pixel_sync_threshold_mv,
            self.targets.avalanche_threshold_mv,
            self.targets.bias_voltage_v,
        )

    def set_outputs(
        self,
        laser_mv: float,
        pixel_mv: float,
        avalanche_mv: float,
        bias_v: float,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        self.targets.laser_sync_threshold_mv = float(laser_mv)
        self.targets.pixel_sync_threshold_mv = float(pixel_mv)
        self.targets.avalanche_threshold_mv = float(avalanche_mv)
        self.targets.bias_voltage_v = float(bias_v)
        self.codes = self.analog_codec.encode_analog_targets(
            self.targets.laser_sync_threshold_mv,
            self.targets.pixel_sync_threshold_mv,
            self.targets.avalanche_threshold_mv,
            self.targets.bias_voltage_v,
        )
        frame = self.encoder.encode_set_ad5686(
            self.codes.laser_sync_code,
            self.codes.pixel_sync_code,
            self.codes.avalanche_code,
            self.codes.bias_code,
        )
        result = self.device_service.send_command_sync(
            protocol.CMD_AD5686,
            frame,
            timeout_ms,
            debug_details=(
                f"laser_mv={self.targets.laser_sync_threshold_mv:.2f}, "
                f"pixel_mv={self.targets.pixel_sync_threshold_mv:.2f}, "
                f"avalanche_mv={self.targets.avalanche_threshold_mv:.2f}, "
                f"bias_v={self.targets.bias_voltage_v:.3f} | "
                f"codes=ch1:{self.codes.laser_sync_code}(0x{self.codes.laser_sync_code:04X}) "
                f"ch2:{self.codes.pixel_sync_code}(0x{self.codes.pixel_sync_code:04X}) "
                f"ch3:{self.codes.avalanche_code}(0x{self.codes.avalanche_code:04X}) "
                f"ch4:{self.codes.bias_code}(0x{self.codes.bias_code:04X})"
            ),
        )
        if result.success:
            self.analog_targets_changed.emit(self.targets, self.codes)
        return result

    def set_threshold(self, channel: int, threshold_mv: float, timeout_ms: int = 1000) -> CommandResult:
        if channel == 1:
            return self.set_outputs(
                threshold_mv,
                self.targets.pixel_sync_threshold_mv,
                self.targets.avalanche_threshold_mv,
                self.targets.bias_voltage_v,
                timeout_ms,
            )
        if channel == 2:
            return self.set_outputs(
                self.targets.laser_sync_threshold_mv,
                threshold_mv,
                self.targets.avalanche_threshold_mv,
                self.targets.bias_voltage_v,
                timeout_ms,
            )
        if channel == 3:
            return self.set_outputs(
                self.targets.laser_sync_threshold_mv,
                self.targets.pixel_sync_threshold_mv,
                threshold_mv,
                self.targets.bias_voltage_v,
                timeout_ms,
            )
        raise ValueError("Threshold channel must be 1, 2, or 3")

    def set_bias(self, bias_v: float, timeout_ms: int = 1000) -> CommandResult:
        return self.set_outputs(
            self.targets.laser_sync_threshold_mv,
            self.targets.pixel_sync_threshold_mv,
            self.targets.avalanche_threshold_mv,
            bias_v,
            timeout_ms,
        )


class FpgaControlService(QtCore.QObject):
    gate_settings_changed = QtCore.pyqtSignal(object)

    def __init__(self, device_service: DeviceService, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.encoder = CommandEncoder()
        self.gate_settings = GateSettings()

    def configure_gpx2(
        self,
        timeout_ms: int = 1000,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> CommandResult:
        profile_id = int(profile) & 0xF
        sequence_id = int(sequence_mode) & 0xF
        profile_name = protocol.GPX2_CONFIG_PROFILES.get(profile_id, "reserved")
        sequence_name = protocol.GPX2_CONFIG_SEQUENCES.get(sequence_id, "reserved")
        return self.device_service.send_command_sync(
            protocol.CMD_GPX2_CFG,
            self.encoder.encode_gpx2_config(profile_id, sequence_id),
            timeout_ms,
            debug_details=(
                f"apply GPX2 configuration profile {profile_id}: {profile_name}; "
                f"sequence {sequence_id}: {sequence_name}"
            ),
        )

    def configure_gpx2_custom(
        self,
        config_bytes,
        timeout_ms: int = 1000,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> CommandResult:
        profile_id = int(profile) & 0xF
        sequence_id = int(sequence_mode) & 0xF
        regs = protocol.sanitize_gpx2_config_bytes(config_bytes)
        return self.device_service.send_command_sync(
            protocol.CMD_GPX2_CFG_CUSTOM,
            self.encoder.encode_gpx2_custom_config(regs, profile_id, sequence_id),
            timeout_ms,
            debug_details=(
                f"apply GPX2 custom config sequence {sequence_id}; "
                f"regs={' '.join(f'{value:02X}' for value in regs)}"
            ),
        )

    def configure_gpx2_lvds_test(self, timeout_ms: int = 1000) -> CommandResult:
        regs = protocol.gpx2_lvds_test_config_bytes()
        return self.configure_gpx2_custom(regs, timeout_ms=timeout_ms, sequence_mode=GPX2_DEFAULT_SEQUENCE)

    def reset_gpx2_power_opcode(self, timeout_ms: int = 1000) -> CommandResult:
        return self.configure_gpx2(timeout_ms=timeout_ms, profile=GPX2_DEFAULT_PROFILE, sequence_mode=2)

    def read_gpx2_config(self, timeout_ms: int = 1000) -> CommandResult:
        return self.configure_gpx2(timeout_ms=timeout_ms, profile=GPX2_DEFAULT_PROFILE, sequence_mode=1)

    def configure_tdc_test_upload(
        self,
        enable: bool,
        channel_mask: int,
        timeout_ms: int = 1000,
        synthetic: bool = False,
        spi: bool = False,
        extended_timestamp_raw: bool = False,
    ) -> CommandResult:
        mask = int(channel_mask) & 0xF
        use_ext = bool(extended_timestamp_raw) and not synthetic and not spi
        payload_word = (
            (1 if enable else 0)
            | ((1 if synthetic else 0) << 1)
            | ((1 if spi else 0) << 2)
            | ((1 if use_ext else 0) << 3)
            | (mask << 4)
        )
        frame = self.encoder.encode_tdc_test_upload(
            enable,
            mask,
            synthetic=synthetic,
            spi=spi,
            extended_timestamp_raw=use_ext,
        )
        return self.device_service.send_command_sync(
            protocol.CMD_TDC_TEST_UPLOAD,
            frame,
            timeout_ms,
            debug_details=(
                f"tdc_raw_test_enable={int(bool(enable))}, "
                f"synthetic={int(bool(synthetic))}, spi={int(bool(spi))}, "
                f"extended_timestamp_raw={int(use_ext)}, "
                f"channel_mask=0x{mask:X}, payload_word=0x{payload_word:08X}"
            ),
        )

    def configure_tdc_histogram(
        self,
        *,
        enable: bool,
        start_channel: int,
        stop_channel: int,
        bin_width_raw: int,
        bin_count: int,
        offset_raw: int = 0,
        reference_cleanup_raw: int = 0,
        tdc_deadtime_raw: int = 0,
        clear: bool = False,
        refclk_divisions: int = 12500,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        frame = self.encoder.encode_tdc_hist_config(
            enable=enable,
            start_channel=start_channel,
            stop_channel=stop_channel,
            bin_width_raw=bin_width_raw,
            bin_count=bin_count,
            offset_raw=offset_raw,
            reference_cleanup_raw=reference_cleanup_raw,
            tdc_deadtime_raw=tdc_deadtime_raw,
            clear=clear,
            refclk_divisions=refclk_divisions,
        )
        return self.device_service.send_command_sync(
            protocol.CMD_TDC_HIST_CONFIG,
            frame,
            timeout_ms,
            debug_details=(
                f"tdc_hist_enable={int(bool(enable))}, clear={int(bool(clear))}, "
                f"start_ch={int(start_channel)}, stop_ch={int(stop_channel)}, "
                f"bin_width_raw={int(bin_width_raw)}, bins={int(bin_count)}, "
                f"offset_raw={int(offset_raw)}, ref_cleanup_raw={int(reference_cleanup_raw)}, "
                f"tdc_deadtime_raw={int(tdc_deadtime_raw)}"
            ),
        )

    def request_tdc_histogram_readout(
        self,
        mask: int = 0x3,
        timeout_ms: int = 1000,
        chunk_index: int | None = None,
    ) -> CommandResult:
        frame = self.encoder.encode_tdc_hist_readout(mask, chunk_index=chunk_index)
        return self.device_service.send_command_sync(
            protocol.CMD_TDC_HIST_READOUT,
            frame,
            timeout_ms,
            debug_details=(
                f"tdc_hist_readout_mask=0x{int(mask) & 0x3:X}"
                + ("" if chunk_index is None else f", chunk_index={int(chunk_index)}")
            ),
        )

    def run_ddr3_bist(
        self,
        *,
        mode: int = 0,
        words: int = 1024,
        clear: bool = False,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        frame = self.encoder.encode_ddr3_bist(
            start=not bool(clear),
            clear=bool(clear),
            mode=mode,
            words=words,
        )
        return self.device_service.send_command_sync(
            protocol.CMD_DDR3_BIST,
            frame,
            timeout_ms,
            debug_details=(
                f"ddr3_bist_start={int(not bool(clear))}, clear={int(bool(clear))}, "
                f"mode={int(mode) & 0xF}, words={max(0, int(words))}"
            ),
        )

    def set_gate_holdoff(self, holdoff: int, timeout_ms: int = 1000) -> CommandResult:
        self.gate_settings.hold_off_time = int(holdoff)
        return self.device_service.send_command_sync(
            protocol.CMD_GATE,
            self.encoder.encode_gate_holdoff(holdoff),
            timeout_ms,
            debug_details=f"gate_holdoff={int(holdoff)}",
        )

    def set_nb6(self, delay_a: int, delay_b: int, enable: bool, timeout_ms: int = 1000) -> CommandResult:
        return self.device_service.send_command_sync(
            protocol.CMD_NB6L295,
            self.encoder.encode_nb6(delay_a, delay_b, enable),
            timeout_ms,
            debug_details=f"delay_a={int(delay_a)}, delay_b={int(delay_b)}, enable={int(bool(enable))}",
        )

    def set_gate_div(self, divider: int, timeout_ms: int = 1000) -> CommandResult:
        self.gate_settings.divider = int(divider)
        result = self.device_service.send_command_sync(
            protocol.CMD_GATE_DIV,
            self.encoder.encode_gate_div(divider),
            timeout_ms,
            debug_details=f"gate_divider={int(divider)}",
        )
        if result.success:
            self.gate_settings_changed.emit(self.gate_settings)
        return result

    def set_gate_signal(
        self,
        signal_id: int,
        delay_coarse: int,
        delay_fine: int,
        width_coarse: int,
        width_fine: int,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        result = self.device_service.send_command_sync(
            protocol.CMD_GATE_SIG2 if signal_id == 2 else protocol.CMD_GATE_SIG3,
            self.encoder.encode_gate_signal(signal_id, delay_coarse, delay_fine, width_coarse, width_fine),
            timeout_ms,
            debug_details=(
                f"signal={int(signal_id)}, delay_coarse={int(delay_coarse)}, delay_fine={int(delay_fine)}, "
                f"width_coarse={int(width_coarse)}, width_fine={int(width_fine)}"
            ),
        )
        if result.success:
            if signal_id == 2:
                self.gate_settings.sig2_delay_coarse = int(delay_coarse)
                self.gate_settings.sig2_delay_fine = int(delay_fine)
                self.gate_settings.sig2_width_coarse = int(width_coarse)
                self.gate_settings.sig2_width_fine = int(width_fine)
            else:
                self.gate_settings.sig3_delay_coarse = int(delay_coarse)
                self.gate_settings.sig3_delay_fine = int(delay_fine)
                self.gate_settings.sig3_width_coarse = int(width_coarse)
                self.gate_settings.sig3_width_fine = int(width_fine)
            self.gate_settings_changed.emit(self.gate_settings)
        return result

    def set_gate_sig3_long_delay(
        self,
        delay_ns: int,
        width_coarse: int,
        width_fine: int,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        delay_value = max(0, min(50000, int(delay_ns)))
        legacy_delay = min(delay_value, 159)
        delay_coarse = legacy_delay // 10
        delay_fine = legacy_delay % 10
        result = self.device_service.send_command_sync(
            protocol.CMD_GATE_SIG3_LONG,
            self.encoder.encode_gate_sig3_long_delay(delay_value, width_coarse, width_fine),
            timeout_ms,
            debug_details=(
                f"signal=3, delay_ns={delay_value}, "
                f"legacy_delay_coarse={delay_coarse}, legacy_delay_fine={delay_fine}, "
                f"width_coarse={int(width_coarse)}, width_fine={int(width_fine)}"
            ),
        )
        if result.success:
            self.gate_settings.sig3_delay_coarse = int(delay_coarse)
            self.gate_settings.sig3_delay_fine = int(delay_fine)
            self.gate_settings.sig3_width_coarse = int(width_coarse)
            self.gate_settings.sig3_width_fine = int(width_fine)
            self.gate_settings_changed.emit(self.gate_settings)
        return result

    def set_gate_enable(
        self,
        sig2_enable: bool,
        sig3_enable: bool,
        pixel_mode: bool,
        timeout_ms: int = 1000,
    ) -> CommandResult:
        self.gate_settings.sig2_enable = bool(sig2_enable)
        self.gate_settings.sig3_enable = bool(sig3_enable)
        self.gate_settings.pixel_mode = bool(pixel_mode)
        result = self.device_service.send_command_sync(
            protocol.CMD_GATE_ENABLE,
            self.encoder.encode_gate_enable(sig2_enable, sig3_enable, pixel_mode),
            timeout_ms,
            debug_details=(
                f"sig2_enable={int(bool(sig2_enable))}, "
                f"sig3_enable={int(bool(sig3_enable))}, "
                f"pixel_mode={int(bool(pixel_mode))}"
            ),
        )
        if result.success:
            self.gate_settings_changed.emit(self.gate_settings)
        return result


class PixelArrayService(QtCore.QObject):
    pixel_records_changed = QtCore.pyqtSignal(object)

    def __init__(self, device_service: DeviceService, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.encoder = CommandEncoder()
        self.records: List[PixelParamRecord] = []

    def set_records(self, records: List[PixelParamRecord]) -> None:
        self.records = list(records)
        self.pixel_records_changed.emit(self.records)

    def write_pixel_param(self, addr: int, value36: int, timeout_ms: int = 1000) -> CommandResult:
        frame = self.encoder.encode_gate_ram_write(addr, value36)
        return self.device_service.send_command_sync(
            protocol.CMD_GATE_RAM,
            frame,
            timeout_ms,
            debug_details=f"addr={int(addr)} (0x{int(addr):04X}), value36=0x{int(value36) & ((1 << 36) - 1):09X}",
        )

    def write_all_records(self, timeout_ms: int = 1000) -> list[CommandResult]:
        results: list[CommandResult] = []
        for record in self.records:
            results.append(self.write_pixel_param(record.addr, record.value36, timeout_ms))
        return results

    def reload_pixel_params(self, timeout_ms: int = 1000) -> CommandResult:
        return self.device_service.send_command_sync(
            protocol.CMD_GATE_PIXEL,
            self.encoder.encode_gate_pixel_reload(True),
            timeout_ms,
            debug_details="reload current pixel parameters (enable_bit=1)",
        )


class FlashService(QtCore.QObject):
    def __init__(self, device_service: DeviceService, parent=None) -> None:
        super().__init__(parent)
        self.device_service = device_service
        self.encoder = CommandEncoder()

    def flash_save(self, timeout_ms: int = 1000) -> CommandResult:
        return self.device_service.send_command_sync(
            protocol.CMD_FLASH_SAVE,
            self.encoder.encode_flash_save(),
            timeout_ms,
            debug_details="save live configuration and pixel image to flash",
        )

    def flash_load(self, timeout_ms: int = 1000) -> CommandResult:
        return self.device_service.send_command_sync(
            protocol.CMD_FLASH_LOAD,
            self.encoder.encode_flash_load(),
            timeout_ms,
            debug_details="load configuration and pixel image from flash",
        )

    def flash_save_analog(self, timeout_ms: int = 1000) -> CommandResult:
        return self.device_service.send_command_sync(
            protocol.CMD_FLASH_SAVE_ANALOG,
            self.encoder.encode_flash_save_analog(),
            timeout_ms,
            debug_details="save AD5686 threshold/bias and TEC target defaults to flash",
        )
