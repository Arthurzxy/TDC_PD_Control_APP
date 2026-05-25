from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.models import (
    Ddr3BistResult,
    PacketHeader,
    ParsedPacket,
    PhotonEvent,
    StatusPacket,
    TdcEvent,
    TdcHistogramChunk,
    TdcRawEventBatch,
)


SYNC_BYTE_DOWNLINK = 0xBB
SYNC_BYTE_UPLINK = 0xA5
PROTO_VERSION = 0x01
HEADER_WORDS = 4

PKT_TDC_RAW = 0x01
PKT_STATUS = 0x02
PKT_ACK = 0x03
PKT_PHOTON_EVENT = 0x04
PKT_TDC_RAW_EXT = 0x05
PKT_TDC_HIST = 0x06
PKT_DDR3_BIST = 0x07

TDC_HIST_STATUS_FLAG_NAMES = {
    0: "enable",
    1: "clear_busy",
    2: "readout_busy",
    3: "hist_overflow",
    4: "input_stall",
    5: "core_present",
    6: "tdc_deadtime_enabled",
    7: "ddr_backend",
    8: "ddr_calibrated",
    9: "ddr_ring_overflow",
    10: "ddr_storage_stall",
    11: "ddr_ring_read_stall",
    12: "ddr_ring_write_stall",
    13: "ddr_cdc_req_overflow",
    14: "ddr_cdc_rsp_overflow",
    15: "ddr_not_calibrated",
}


def decode_tdc_hist_status_flags(flags: int) -> dict[str, bool]:
    value = int(flags) & 0xFFFFFFFF
    return {
        name: bool(value & (1 << bit))
        for bit, name in TDC_HIST_STATUS_FLAG_NAMES.items()
    }


def describe_tdc_hist_status_flags(flags: int) -> str:
    decoded = decode_tdc_hist_status_flags(flags)
    active = [name for name, enabled in decoded.items() if enabled]
    return ",".join(active) if active else "-"

CMD_AD5686 = 0x01
CMD_GATE = 0x02
CMD_NB6L295 = 0x03
CMD_TEC_PID = 0x04
CMD_GPX2_CFG = 0x10
CMD_GPX2_CFG_CUSTOM = 0x11
CMD_GATE_DIV = 0x20
CMD_GATE_SIG2 = 0x21
CMD_GATE_SIG3 = 0x22
CMD_GATE_ENABLE = 0x23
CMD_GATE_PIXEL = 0x24
CMD_GATE_RAM = 0x25
CMD_TDC_TEST_UPLOAD = 0x26
CMD_TDC_HIST_CONFIG = 0x27
CMD_TDC_HIST_READOUT = 0x28
CMD_DDR3_BIST = 0x29
CMD_FLASH_SAVE = 0x30
CMD_FLASH_LOAD = 0x31
CMD_FLASH_SAVE_ANALOG = 0x32
CMD_GATE_SIG3_LONG = 0x33

GPX2_DEFAULT_PROFILE = 0
GPX2_DEFAULT_SEQUENCE = 10

CMD_NAMES = {
    CMD_AD5686: "CMD_AD5686",
    CMD_GATE: "CMD_GATE",
    CMD_NB6L295: "CMD_NB6L295",
    CMD_TEC_PID: "CMD_TEC_PID",
    CMD_GPX2_CFG: "CMD_GPX2_CFG",
    CMD_GPX2_CFG_CUSTOM: "CMD_GPX2_CFG_CUSTOM",
    CMD_GATE_DIV: "CMD_GATE_DIV",
    CMD_GATE_SIG2: "CMD_GATE_SIG2",
    CMD_GATE_SIG3: "CMD_GATE_SIG3",
    CMD_GATE_SIG3_LONG: "CMD_GATE_SIG3_LONG",
    CMD_GATE_ENABLE: "CMD_GATE_ENABLE",
    CMD_GATE_PIXEL: "CMD_GATE_PIXEL",
    CMD_GATE_RAM: "CMD_GATE_RAM",
    CMD_TDC_TEST_UPLOAD: "CMD_TDC_TEST_UPLOAD",
    CMD_TDC_HIST_CONFIG: "CMD_TDC_HIST_CONFIG",
    CMD_TDC_HIST_READOUT: "CMD_TDC_HIST_READOUT",
    CMD_DDR3_BIST: "CMD_DDR3_BIST",
    CMD_FLASH_SAVE: "CMD_FLASH_SAVE",
    CMD_FLASH_LOAD: "CMD_FLASH_LOAD",
    CMD_FLASH_SAVE_ANALOG: "CMD_FLASH_SAVE_ANALOG",
}

GPX2_DEFAULT_CONFIG_BYTES = (
    0x3F, 0x8F, 0x24, 0xD4, 0x30, 0x00, 0xC0, 0x53,
    0xA1, 0x13, 0x00, 0x0A, 0xCC, 0xCC, 0xF1, 0x7D, 0x00,
)

GPX2_FIXED_CONFIG_BYTES = {
    8: 0xA1,
    9: 0x13,
    10: 0x00,
    11: 0x0A,
    12: 0xCC,
    13: 0xCC,
    14: 0xF1,
    15: 0x7D,
}

GPX2_CONFIG_REGISTER_NAMES = (
    "PIN_ENA",
    "HIT/COMBINE/RES",
    "DATA_OUTPUT",
    "REFCLK_DIV_LO",
    "REFCLK_DIV_MID",
    "REFCLK_DIV_HI",
    "LVDS_TEST",
    "REFCLK/DVALID",
    "FIXED_A1",
    "FIXED_13",
    "FIXED_00",
    "FIXED_0A",
    "FIXED_CC",
    "FIXED_CC",
    "FIXED_F1",
    "FIXED_7D",
    "CMOS_INPUT",
)


@dataclass(frozen=True)
class Gpx2RegisterSpec:
    address: int
    name: str
    summary: str
    detail: str
    editable: bool = True


GPX2_REGISTER_SPECS = (
    Gpx2RegisterSpec(
        0,
        "PIN_ENA",
        "Pin enables for STOP/REF/LVDS related paths.",
        "Project profiles use reg0 to choose which GPX2 input/output groups are active. "
        "Common values: 0x00 idle, 0x10 REF only, 0x11 STOP1+REF, 0x1F STOP1-4+REF, "
        "0x20 LVDS output only, 0x3F full normal.",
    ),
    Gpx2RegisterSpec(
        1,
        "HIT_COMBINE_RES",
        "Hit enable and high-resolution mode selection.",
        "Project default: 0x8F full normal with 4x high resolution. "
        "Older 0x4F is full normal with 2x high resolution. 0x40 pins on but hits off, "
        "0x01 minimal STOP1+REF, 0x00 idle/REF-only.",
    ),
    Gpx2RegisterSpec(
        2,
        "DATA_OUTPUT",
        "Output format and framing mode.",
        "0x24 is the normal DDR output format used in this design. "
        "0x3D is the LVDS test-pattern mode. 0x00 is used by minimal or raw-input profiles.",
    ),
    Gpx2RegisterSpec(
        3,
        "REFCLK_DIV_LO",
        "REFCLK_DIVISIONS lower byte.",
        "Combined with reg4/reg5 to form the 20-bit REFCLK_DIVISIONS value. "
        "Normal profile uses D4 30 00 (= 12500).",
    ),
    Gpx2RegisterSpec(
        4,
        "REFCLK_DIV_MID",
        "REFCLK_DIVISIONS middle byte.",
        "Combined with reg3/reg5 to form the 20-bit REFCLK_DIVISIONS value. "
        "The example 5 MHz path uses 40 0D 03.",
    ),
    Gpx2RegisterSpec(
        5,
        "REFCLK_DIV_HI",
        "REFCLK_DIVISIONS upper nibble.",
        "Only the low nibble is writable in this project. The upper nibble is forced low by host/FPGA masking.",
    ),
    Gpx2RegisterSpec(
        6,
        "LVDS_TEST",
        "LVDS output and test-pattern control.",
        "bit4 enables LVDS_TEST_PATTERN. Bits [7:5] are forced to 110 by the host/FPGA masks. "
        "Normal = 0xC0, LVDS test = 0xD0.",
    ),
    Gpx2RegisterSpec(
        7,
        "REFCLK_DVALID",
        "REFCLK source and LVDS DATA_VALID adjust.",
        "Project normal uses 0x53 for external REFCLK with 0 ps data-valid adjust. "
        "The low nibble is forced to 0x3; only the upper control bits are editable.",
    ),
    Gpx2RegisterSpec(
        8,
        "FIXED_A1",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0xA1 by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        9,
        "FIXED_13",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0x13 by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        10,
        "FIXED_00",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0x00 by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        11,
        "FIXED_0A",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0x0A by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        12,
        "FIXED_CC",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0xCC by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        13,
        "FIXED_CC",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0xCC by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        14,
        "FIXED_F1",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0xF1 by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        15,
        "FIXED_7D",
        "Fixed byte required by the current board configuration.",
        "This register is held at 0x7D by both the host and FPGA custom-config masks.",
        editable=False,
    ),
    Gpx2RegisterSpec(
        16,
        "CMOS_INPUT",
        "Input mode select.",
        "bit2 selects CMOS input mode. 0x00 keeps LVDS input mode, 0x04 enables the CMOS-reference profile.",
    ),
)

GPX2_CONFIG_PROFILES = {
    0: "full normal",
    1: "power/core only",
    2: "REF only",
    3: "SPI raw all STOP+REF LVDS input",
    4: "LVDS out only",
    5: "pins on, hits off",
    6: "LVDS test pattern",
    7: "legacy NLOS 10 ps",
    8: "all-zero registers",
    9: "CSDN single-channel CMOS reference",
    10: "LVDS CH1+REF no LVDS out",
    11: "LVDS CH1+REF+LVDS out",
    12: "all STOP+REF, hits off",
    13: "minimal legal idle",
    14: "minimal legal REF only",
    15: "minimal legal STOP1+REF",
}

GPX2_CONFIG_SEQUENCES = {
    0: "diagnostic fast full sequence, readback-gated INIT",
    1: "read config only",
    2: "POWER only",
    3: "write/read no POWER or INIT",
    4: "INIT only",
    5: "POWER + write/read, no INIT",
    6: "POWER + INIT, no write",
    7: "write/read + INIT, no POWER",
    8: "write + INIT, no readback",
    9: "manual slow 0x30 + 0x40 read",
    10: "manual slow full config + INIT, no readback",
    11: "manual slow config write only",
    12: "manual slow POWER + INIT only",
    13: "manual slow POWER + 1s wait + INIT",
    14: "manual slow INIT only, no POWER",
    15: "manual slow POWER + read + INIT",
}


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def pack_u32_le(word: int) -> bytes:
    return int(word & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False)


def sanitize_gpx2_config_bytes(values: List[int] | tuple[int, ...] | bytes | bytearray | None = None) -> tuple[int, ...]:
    regs = list(GPX2_DEFAULT_CONFIG_BYTES)
    if values is not None:
        for idx, value in enumerate(values):
            if idx >= len(regs):
                break
            regs[idx] = int(value) & 0xFF

    regs[5] &= 0x0F
    regs[6] = (regs[6] & 0x10) | 0xC0
    regs[7] = (regs[7] & 0xB0) | 0x43
    for idx, value in GPX2_FIXED_CONFIG_BYTES.items():
        regs[idx] = value
    regs[16] &= 0x04
    return tuple(regs)


def gpx2_lvds_test_config_bytes() -> tuple[int, ...]:
    regs = list(GPX2_DEFAULT_CONFIG_BYTES)
    regs[0] |= 0x20
    regs[6] |= 0x10
    return sanitize_gpx2_config_bytes(regs)


def pack_gpx2_config_readback_words(regs: List[int] | tuple[int, ...]) -> list[int]:
    values = list(regs)[:17]
    while len(values) < 17:
        values.append(0)
    return [
        ((values[0] & 0xFF) << 24) | ((values[1] & 0xFF) << 16) | ((values[2] & 0xFF) << 8) | (values[3] & 0xFF),
        ((values[4] & 0xFF) << 24) | ((values[5] & 0xFF) << 16) | ((values[6] & 0xFF) << 8) | (values[7] & 0xFF),
        ((values[8] & 0xFF) << 24) | ((values[9] & 0xFF) << 16) | ((values[10] & 0xFF) << 8) | (values[11] & 0xFF),
        ((values[12] & 0xFF) << 24) | ((values[13] & 0xFF) << 16) | ((values[14] & 0xFF) << 8) | (values[15] & 0xFF),
        (values[16] & 0xFF) << 24,
    ]


def unpack_gpx2_config_readback_words(words: List[int] | tuple[int, ...]) -> tuple[int, ...]:
    if len(words) < 5:
        return ()
    regs: list[int] = []
    for word in words[:4]:
        regs.extend([(word >> 24) & 0xFF, (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF])
    regs.append((words[4] >> 24) & 0xFF)
    return tuple(regs)


def gpx2_refclk_divisions(regs: List[int] | tuple[int, ...]) -> int:
    values = list(regs)[:17]
    while len(values) < 17:
        values.append(0)
    return (values[3] & 0xFF) | ((values[4] & 0xFF) << 8) | ((values[5] & 0x0F) << 16)


def describe_gpx2_config_bytes(regs: List[int] | tuple[int, ...]) -> dict[str, str]:
    values = list(regs)[:17]
    while len(values) < 17:
        values.append(0)
    decoded = {
        f"reg{spec.address:02d}_{spec.name}": f"0x{values[spec.address] & 0xFF:02X}"
        for spec in GPX2_REGISTER_SPECS
    }
    decoded["refclk_divisions"] = str(gpx2_refclk_divisions(values))
    return decoded


class CommandEncoder:
    """Encode host-to-FPGA commands using the current 0xBB word protocol."""

    @staticmethod
    def build_frame(cmd_id: int, payload_words: List[int] | None = None) -> bytes:
        payload_words = payload_words or []
        header = ((SYNC_BYTE_DOWNLINK & 0xFF) << 24) | ((cmd_id & 0xFF) << 16) | (len(payload_words) & 0xF)
        frame = bytearray(pack_u32_le(header))
        for word in payload_words:
            frame.extend(pack_u32_le(word))
        return bytes(frame)

    def encode_set_ad5686(self, ch1_code: int, ch2_code: int, ch3_code: int, ch4_code: int) -> bytes:
        word0 = ((ch1_code & 0xFFFF) << 16) | (ch2_code & 0xFFFF)
        word1 = ((ch3_code & 0xFFFF) << 16) | (ch4_code & 0xFFFF)
        return self.build_frame(CMD_AD5686, [word0, word1])

    def encode_set_temperature_code(self, temp_code: int) -> bytes:
        return self.build_frame(CMD_TEC_PID, [temp_code & 0xFFFF])

    def encode_gpx2_config(
        self,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> bytes:
        word = (int(profile) & 0xF) | ((int(sequence_mode) & 0xF) << 4)
        return self.build_frame(CMD_GPX2_CFG, [word])

    def encode_gpx2_custom_config(
        self,
        config_bytes: List[int] | tuple[int, ...] | bytes | bytearray | None = None,
        profile: int = GPX2_DEFAULT_PROFILE,
        sequence_mode: int = GPX2_DEFAULT_SEQUENCE,
    ) -> bytes:
        regs = sanitize_gpx2_config_bytes(config_bytes)
        control = (int(profile) & 0xF) | ((int(sequence_mode) & 0xF) << 4)
        return self.build_frame(CMD_GPX2_CFG_CUSTOM, [control] + pack_gpx2_config_readback_words(regs))

    def encode_gpx2_lvds_test_config(self, sequence_mode: int = GPX2_DEFAULT_SEQUENCE) -> bytes:
        return self.encode_gpx2_custom_config(gpx2_lvds_test_config_bytes(), sequence_mode=sequence_mode)

    def encode_gate_holdoff(self, holdoff: int) -> bytes:
        return self.build_frame(CMD_GATE, [holdoff & 0xFFFFFF])

    def encode_nb6(self, delay_a: int, delay_b: int, enable: bool) -> bytes:
        word = (delay_a & 0x1FF) | ((delay_b & 0x1FF) << 9) | ((1 if enable else 0) << 18)
        return self.build_frame(CMD_NB6L295, [word])

    def encode_gate_div(self, divider: int) -> bytes:
        value = int(divider)
        encoded = 0 if value >= 4096 else clamp(value, 1, 4095)
        return self.build_frame(CMD_GATE_DIV, [encoded & 0xFFF])

    def encode_gate_signal(
        self,
        signal_id: int,
        delay_coarse: int,
        delay_fine: int,
        width_coarse: int,
        width_fine: int,
    ) -> bytes:
        word = (
            (delay_coarse & 0xF)
            | ((delay_fine & 0x1F) << 4)
            | ((width_coarse & 0x7) << 9)
            | ((width_fine & 0x1F) << 12)
        )
        cmd = CMD_GATE_SIG2 if signal_id == 2 else CMD_GATE_SIG3
        return self.build_frame(cmd, [word])

    def encode_gate_sig3_long_delay(
        self,
        delay_ns: int,
        width_coarse: int,
        width_fine: int,
    ) -> bytes:
        delay_value = clamp(int(delay_ns), 0, 50000)
        legacy_delay = min(delay_value, 159)
        delay_coarse = legacy_delay // 10
        delay_fine = legacy_delay % 10
        word0 = (
            (delay_coarse & 0xF)
            | ((delay_fine & 0x1F) << 4)
            | ((int(width_coarse) & 0x7) << 9)
            | ((int(width_fine) & 0x1F) << 12)
        )
        return self.build_frame(CMD_GATE_SIG3_LONG, [word0, delay_value & 0xFFFF])

    def encode_gate_enable(self, sig2_enable: bool, sig3_enable: bool, pixel_mode: bool) -> bytes:
        word = (1 if sig2_enable else 0) | ((1 if sig3_enable else 0) << 1) | ((1 if pixel_mode else 0) << 2)
        return self.build_frame(CMD_GATE_ENABLE, [word])

    def encode_gate_pixel_reload(self, enable_bit: bool = True) -> bytes:
        return self.build_frame(CMD_GATE_PIXEL, [1 if enable_bit else 0])

    def encode_gate_ram_write(self, addr: int, value36: int) -> bytes:
        combined = ((addr & 0x3FFF) << 36) | (value36 & ((1 << 36) - 1))
        word0 = (combined >> 32) & 0xFFFFFFFF
        word1 = combined & 0xFFFFFFFF
        return self.build_frame(CMD_GATE_RAM, [word0, word1])

    def encode_tdc_test_upload(
        self,
        enable: bool,
        channel_mask: int,
        synthetic: bool = False,
        spi: bool = False,
        extended_timestamp_raw: bool = False,
    ) -> bytes:
        mask = channel_mask & 0xF
        use_ext = bool(extended_timestamp_raw) and not synthetic and not spi
        word = (
            (1 if enable else 0)
            | ((1 if synthetic else 0) << 1)
            | ((1 if spi else 0) << 2)
            | ((1 if use_ext else 0) << 3)
            | (mask << 4)
        )
        return self.build_frame(CMD_TDC_TEST_UPLOAD, [word])

    def encode_tdc_hist_config(
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
        refclk_divisions: int = GPX2_DEFAULT_CONFIG_BYTES[3]
        | (GPX2_DEFAULT_CONFIG_BYTES[4] << 8)
        | ((GPX2_DEFAULT_CONFIG_BYTES[5] & 0x0F) << 16),
    ) -> bytes:
        control = (
            (1 if enable else 0)
            | ((1 if clear else 0) << 1)
            | ((int(start_channel) & 0x3) << 4)
            | ((int(stop_channel) & 0x3) << 8)
        )
        return self.build_frame(
            CMD_TDC_HIST_CONFIG,
            [
                control,
                int(refclk_divisions) & 0xFFFFFFFF,
                max(1, int(bin_width_raw)) & 0xFFFFFFFF,
                int(offset_raw) & 0xFFFFFFFF,
                max(1, int(bin_count)) & 0xFFFFFFFF,
                max(0, int(reference_cleanup_raw)) & 0xFFFFFFFF,
                max(0, int(tdc_deadtime_raw)) & 0xFFFFFFFF,
            ],
        )

    def encode_tdc_hist_readout(self, mask: int = 0x3, chunk_index: int | None = None) -> bytes:
        word = int(mask) & 0x3
        if chunk_index is not None:
            word |= 0x80000000
            word |= (int(chunk_index) & 0xFFFF) << 2
        return self.build_frame(CMD_TDC_HIST_READOUT, [word])

    def encode_ddr3_bist(
        self,
        *,
        start: bool = True,
        clear: bool = False,
        mode: int = 0,
        words: int = 1024,
    ) -> bytes:
        word_count = clamp(int(words), 0, 0xFFFFFF)
        word = (
            (1 if start else 0)
            | ((1 if clear else 0) << 1)
            | ((int(mode) & 0xF) << 4)
            | (word_count << 8)
        )
        return self.build_frame(CMD_DDR3_BIST, [word])

    def encode_flash_save(self) -> bytes:
        return self.build_frame(CMD_FLASH_SAVE, [])

    def encode_flash_load(self) -> bytes:
        return self.build_frame(CMD_FLASH_LOAD, [])

    def encode_flash_save_analog(self) -> bytes:
        return self.build_frame(CMD_FLASH_SAVE_ANALOG, [])


@dataclass
class AnalogCodes:
    laser_sync_code: int
    pixel_sync_code: int
    avalanche_code: int
    bias_code: int


class LegacyAnalogCodec:
    """Compatibility encoder derived from the old main_1.py formulas."""

    def __init__(self, threshold_range_mv: float = 2500.0, bias_scale: float = 433.0) -> None:
        self.threshold_range_mv = threshold_range_mv
        self.bias_scale = bias_scale

    def threshold_mv_to_code(self, threshold_mv: float) -> int:
        scaled = round((float(threshold_mv) / self.threshold_range_mv) * 65535.0)
        return clamp(scaled, 0, 0xFFFF)

    def bias_v_to_code(self, bias_v: float) -> int:
        scaled = round(float(bias_v) * self.bias_scale)
        return clamp(scaled, 0, 0xFFFF)

    def temperature_c_to_code(self, temp_c: float) -> int:
        t = float(temp_c)
        raw = round(
            16497.62491
            - 664.96558 * t
            + 10.82931 * (t ** 2)
            + 0.02139 * (t ** 3)
            - 0.00252 * (t ** 4)
        )
        return clamp(raw, 0, 0xFFFF)

    def encode_analog_targets(
        self,
        laser_sync_threshold_mv: float,
        pixel_sync_threshold_mv: float,
        avalanche_threshold_mv: float,
        bias_voltage_v: float,
    ) -> AnalogCodes:
        return AnalogCodes(
            laser_sync_code=self.threshold_mv_to_code(laser_sync_threshold_mv),
            pixel_sync_code=self.threshold_mv_to_code(pixel_sync_threshold_mv),
            avalanche_code=self.threshold_mv_to_code(avalanche_threshold_mv),
            bias_code=self.bias_v_to_code(bias_voltage_v),
        )


class ByteRingBuffer:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def append(self, data: bytes) -> None:
        self._buffer.extend(data)

    def __len__(self) -> int:
        return len(self._buffer)

    def peek(self, size: int) -> bytes:
        return bytes(self._buffer[:size])

    def consume(self, size: int) -> bytes:
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def discard(self, size: int) -> None:
        del self._buffer[:size]


class PacketParser:
    """Reassemble FPGA uplink packets from the raw FT601 byte stream."""

    GPX2_REFCLK_DIVISIONS = 12500
    MAX_EXT_TIMESTAMP = ((1 << 48) - 1) * GPX2_REFCLK_DIVISIONS + 0x3FFF
    _MAX_PAYLOAD_WORDS_BY_TYPE = {
        PKT_TDC_RAW: 252,
        PKT_STATUS: 20,
        PKT_ACK: 3,
        PKT_PHOTON_EVENT: 80,
        PKT_TDC_RAW_EXT: 252,
        PKT_TDC_HIST: 272,
        PKT_DDR3_BIST: 16,
    }

    def __init__(self, decode_tdc_events: bool = True) -> None:
        self.buffer = ByteRingBuffer()
        self.decode_tdc_events = bool(decode_tdc_events)
        self.reject_counts: dict[str, int] = {}
        self.reject_samples: list[str] = []

    @staticmethod
    def _u32_at(data: bytes, offset: int) -> int:
        return int.from_bytes(data[offset : offset + 4], byteorder="little", signed=False)

    def _record_reject(self, reason: str, detail: str = "") -> None:
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1
        if len(self.reject_samples) >= 16:
            return
        preview = self.buffer.peek(min(len(self.buffer), 96)).hex(" ")
        suffix = f" {detail}" if detail else ""
        self.reject_samples.append(f"{reason}{suffix} preview={preview}")

    def feed(self, data: bytes) -> List[ParsedPacket]:
        self.buffer.append(data)
        packets: List[ParsedPacket] = []

        while len(self.buffer) >= 16:
            header_bytes = self.buffer.peek(16)
            word0 = self._u32_at(header_bytes, 0)
            sync = (word0 >> 24) & 0xFF
            pkt_type = (word0 >> 16) & 0xFF
            version = (word0 >> 8) & 0xFF
            hdr_words = word0 & 0xFF

            if (
                sync != SYNC_BYTE_UPLINK
                or hdr_words != HEADER_WORDS
                or version != PROTO_VERSION
                or pkt_type not in self._MAX_PAYLOAD_WORDS_BY_TYPE
            ):
                if sync == SYNC_BYTE_UPLINK:
                    self._record_reject(
                        "bad_header",
                        f"type=0x{pkt_type:02X} ver={version} hdr={hdr_words} word0=0x{word0:08X}",
                    )
                self.buffer.discard(1)
                continue

            word1 = self._u32_at(header_bytes, 4)
            word2 = self._u32_at(header_bytes, 8)
            word3 = self._u32_at(header_bytes, 12)
            seq = (word1 >> 16) & 0xFFFF
            payload_words = word1 & 0xFFFF
            item_count = (word2 >> 16) & 0xFFFF
            flags = word2 & 0xFFFF
            max_payload_words = self._MAX_PAYLOAD_WORDS_BY_TYPE[pkt_type]
            if (
                payload_words > max_payload_words
                or (pkt_type == PKT_TDC_RAW and payload_words != item_count * 2)
                or (pkt_type == PKT_TDC_RAW_EXT and payload_words != item_count * 3)
                or (pkt_type == PKT_PHOTON_EVENT and payload_words != item_count * 5)
                or (pkt_type == PKT_STATUS and payload_words not in (16, 20))
                or (pkt_type == PKT_ACK and payload_words != 3)
                or (pkt_type == PKT_DDR3_BIST and (payload_words != 16 or item_count != 16))
            ):
                self._record_reject(
                    "bad_length",
                    f"seq={seq} type=0x{pkt_type:02X} payload={payload_words} items={item_count} flags=0x{flags:04X}",
                )
                self.buffer.discard(1)
                continue
            total_words = hdr_words + payload_words
            total_bytes = total_words * 4
            if len(self.buffer) < total_bytes:
                break

            packet_bytes = self.buffer.peek(total_bytes)
            payload = [self._u32_at(packet_bytes, 16 + idx * 4) for idx in range(payload_words)]
            payload_reject = self._payload_reject_reason(pkt_type, payload)
            if payload_reject is not None:
                self._record_reject(
                    payload_reject,
                    f"seq={seq} type=0x{pkt_type:02X} payload={payload_words} items={item_count}",
                )
                # The header and length fields were self-consistent, so treat
                # this as one malformed packet and resume at the next boundary.
                # Byte-wise resync here tends to rediscover header-like words
                # inside the bad payload and creates a cascade of fake rejects.
                self.buffer.discard(total_bytes)
                continue

            packet_bytes = self.buffer.consume(total_bytes)
            header = PacketHeader(
                sync=sync,
                pkt_type=pkt_type,
                version=version,
                hdr_words=hdr_words,
                seq=seq,
                payload_words=payload_words,
                item_count=item_count,
                flags=flags,
                timestamp_us=word3,
            )
            packets.append(self._decode_packet(header, payload, packet_bytes))

        return packets

    @classmethod
    def _payload_matches_header(cls, pkt_type: int, payload_words: List[int]) -> bool:
        return cls._payload_reject_reason(pkt_type, payload_words) is None

    @classmethod
    def _payload_reject_reason(cls, pkt_type: int, payload_words: List[int]) -> str | None:
        if pkt_type == PKT_TDC_HIST:
            if len(payload_words) < 16:
                return "bad_hist_short"
            chunk_count = int(payload_words[3] & 0xFFFF)
            total_bins = int(payload_words[4])
            bin_start = int(payload_words[2])
            if chunk_count > 256:
                return "bad_hist_chunk"
            if len(payload_words) != 16 + chunk_count:
                return "bad_hist_length"
            if bin_start + chunk_count > total_bins:
                return "bad_hist_range"
        if pkt_type == PKT_DDR3_BIST:
            if len(payload_words) != 16:
                return "bad_bist_length"
            if ((int(payload_words[0]) >> 16) & 0xFFFF) != 0xDDB1:
                return "bad_bist_magic"
        if pkt_type == PKT_TDC_RAW_EXT:
            if len(payload_words) % 3 != 0:
                return "bad_ext_mod"
            last_timestamp: int | None = None
            for idx in range(0, len(payload_words), 3):
                timestamp = (int(payload_words[idx + 1]) << 32) | int(payload_words[idx])
                meta = int(payload_words[idx + 2])
                tstop = (meta >> 16) & 0x3FFF
                if timestamp > cls.MAX_EXT_TIMESTAMP:
                    return "bad_ext_timestamp"
                if tstop >= cls.GPX2_REFCLK_DIVISIONS:
                    return "bad_ext_tstop"
                # Timestamp merge is best-effort across independent GPX2 channel
                # FIFOs. A rare late event must not make us discard the whole USB
                # packet; analysis code counts/sorts regressions explicitly.
                last_timestamp = timestamp
        return None

    def _decode_packet(self, header: PacketHeader, payload_words: List[int], raw_bytes: bytes = b"") -> ParsedPacket:
        if header.pkt_type == PKT_STATUS:
            status = StatusPacket(
                header=header,
                flags=payload_words[0] & 0xFFFF if len(payload_words) > 0 else 0,
                uptime_seconds=payload_words[1] if len(payload_words) > 1 else 0,
                temp_avg_raw=payload_words[2] & 0xFFFF if len(payload_words) > 2 else 0,
                counter_1s=payload_words[3] if len(payload_words) > 3 else 0,
                tdc_drop_count=payload_words[4] if len(payload_words) > 4 else 0,
                usb_drop_count=payload_words[5] if len(payload_words) > 5 else 0,
                gpx2_raw_count_ch1=payload_words[6] if len(payload_words) > 6 else 0,
                gpx2_raw_count_ch2=payload_words[7] if len(payload_words) > 7 else 0,
                gpx2_raw_count_ch3=payload_words[8] if len(payload_words) > 8 else 0,
                gpx2_raw_count_ch4=payload_words[9] if len(payload_words) > 9 else 0,
                gpx2_cfg_diag=payload_words[10] if len(payload_words) > 10 else 0,
                gpx2_cfg_readback=unpack_gpx2_config_readback_words(payload_words[11:16]) if len(payload_words) >= 16 else (),
                gate_trigger_count=payload_words[16] if len(payload_words) > 16 else 0,
                gate_direct_pulse_count=payload_words[17] if len(payload_words) > 17 else 0,
                gate_divided_pulse_count=payload_words[18] if len(payload_words) > 18 else 0,
                gate_output_pulse_count=payload_words[19] if len(payload_words) > 19 else 0,
            )
            return ParsedPacket(header=header, payload_words=payload_words, status=status, raw_bytes=raw_bytes)

        if header.pkt_type == PKT_TDC_RAW:
            batch = self._decode_tdc_event_batch(header, payload_words)
            events = self._decode_tdc_events_from_batch(batch) if self.decode_tdc_events else None
            return ParsedPacket(
                header=header,
                payload_words=payload_words,
                tdc_events=events,
                tdc_event_batch=batch,
                raw_bytes=raw_bytes,
            )

        if header.pkt_type == PKT_TDC_RAW_EXT:
            batch = self._decode_tdc_event_ext_batch(header, payload_words)
            events = self._decode_tdc_events_from_batch(batch) if self.decode_tdc_events else None
            return ParsedPacket(
                header=header,
                payload_words=payload_words,
                tdc_events=events,
                tdc_event_batch=batch,
                raw_bytes=raw_bytes,
            )

        if header.pkt_type == PKT_PHOTON_EVENT:
            events = self._decode_photon_events(header, payload_words)
            return ParsedPacket(header=header, payload_words=payload_words, photon_events=events, raw_bytes=raw_bytes)

        if header.pkt_type == PKT_TDC_HIST:
            chunk = self._decode_tdc_histogram_chunk(header, payload_words)
            return ParsedPacket(
                header=header,
                payload_words=payload_words,
                histogram_chunk=chunk,
                raw_bytes=raw_bytes,
            )

        if header.pkt_type == PKT_DDR3_BIST:
            result = self._decode_ddr3_bist_result(header, payload_words)
            return ParsedPacket(
                header=header,
                payload_words=payload_words,
                ddr3_bist_result=result,
                raw_bytes=raw_bytes,
            )

        return ParsedPacket(header=header, payload_words=payload_words, raw_bytes=raw_bytes)

    @staticmethod
    def _decode_tdc_histogram_chunk(header: PacketHeader, payload_words: List[int]) -> TdcHistogramChunk:
        chunk_count = int(payload_words[3] & 0xFFFF) if len(payload_words) > 3 else 0
        aux = int(payload_words[15]) if len(payload_words) > 15 else 0
        bins = np.asarray(payload_words[16 : 16 + chunk_count], dtype=np.uint32)
        return TdcHistogramChunk(
            header=header,
            snapshot_id=int(payload_words[0]) if len(payload_words) > 0 else 0,
            hist_id=int(payload_words[1] & 0x3) if len(payload_words) > 1 else 0,
            bin_start=int(payload_words[2]) if len(payload_words) > 2 else 0,
            bin_count=chunk_count,
            total_bins=int(payload_words[4]) if len(payload_words) > 4 else 0,
            bin_width_raw=int(payload_words[5]) if len(payload_words) > 5 else 1,
            offset_raw=int(payload_words[6]) if len(payload_words) > 6 else 0,
            status_flags=int(payload_words[7]) if len(payload_words) > 7 else 0,
            start_count=int(payload_words[8]) if len(payload_words) > 8 else 0,
            stop_count=int(payload_words[9]) if len(payload_words) > 9 else 0,
            accepted_count=int(payload_words[10]) if len(payload_words) > 10 else 0,
            out_of_window_count=int(payload_words[11]) if len(payload_words) > 11 else 0,
            no_start_count=int(payload_words[12]) if len(payload_words) > 12 else 0,
            first_stop_suppressed_count=int(payload_words[13]) if len(payload_words) > 13 else 0,
            last_dt_raw=int(payload_words[14]) if len(payload_words) > 14 else 0,
            reference_cleanup_count=aux & 0xFFFF,
            tdc_deadtime_filtered_count=(aux >> 16) & 0xFFFF,
            bins=bins,
        )

    @staticmethod
    def _decode_ddr3_bist_result(header: PacketHeader, payload_words: List[int]) -> Ddr3BistResult:
        word0 = int(payload_words[0]) if payload_words else 0
        flags = word0 & 0xFF
        expected = (
            (int(payload_words[5]) if len(payload_words) > 5 else 0)
            | ((int(payload_words[6]) if len(payload_words) > 6 else 0) << 32)
            | ((int(payload_words[7]) if len(payload_words) > 7 else 0) << 64)
            | ((int(payload_words[8]) if len(payload_words) > 8 else 0) << 96)
        )
        actual = (
            (int(payload_words[9]) if len(payload_words) > 9 else 0)
            | ((int(payload_words[10]) if len(payload_words) > 10 else 0) << 32)
            | ((int(payload_words[11]) if len(payload_words) > 11 else 0) << 64)
            | ((int(payload_words[12]) if len(payload_words) > 12 else 0) << 96)
        )
        width_word = int(payload_words[15]) if len(payload_words) > 15 else 0
        return Ddr3BistResult(
            header=header,
            mode=(word0 >> 12) & 0xF,
            calibrated=bool(flags & (1 << 0)),
            running=bool(flags & (1 << 1)),
            done=bool(flags & (1 << 2)),
            passed=bool(flags & (1 << 3)),
            failed=bool(flags & (1 << 4)),
            requested_words=int(payload_words[1]) if len(payload_words) > 1 else 0,
            tested_words=int(payload_words[2]) if len(payload_words) > 2 else 0,
            error_count=int(payload_words[3]) if len(payload_words) > 3 else 0,
            first_error_addr=int(payload_words[4]) if len(payload_words) > 4 else 0,
            expected=expected,
            actual=actual,
            write_cycles=int(payload_words[13]) if len(payload_words) > 13 else 0,
            read_cycles=int(payload_words[14]) if len(payload_words) > 14 else 0,
            app_addr_width=(width_word >> 8) & 0xFF,
            app_data_width=width_word & 0xFF,
        )

    @staticmethod
    def _decode_tdc_events(header: PacketHeader, payload_words: List[int]) -> List[TdcEvent]:
        return PacketParser._decode_tdc_events_from_batch(
            PacketParser._decode_tdc_event_batch(header, payload_words)
        )

    @staticmethod
    def _decode_tdc_event_batch(header: PacketHeader, payload_words: List[int]) -> TdcRawEventBatch:
        if len(payload_words) < 2:
            empty_u8 = np.array([], dtype=np.uint8)
            empty_u32 = np.array([], dtype=np.uint32)
            return TdcRawEventBatch(
                header=header,
                channels=empty_u8,
                refids=empty_u32,
                tstops=empty_u32,
                rec_types=empty_u8,
                event_classes=empty_u8,
                reserved=empty_u32,
            )

        word_count = len(payload_words) & ~1
        words = np.asarray(payload_words[:word_count], dtype=np.uint32)
        lows = words[0::2].astype(np.uint64, copy=False)
        highs = words[1::2].astype(np.uint64, copy=False)
        return TdcRawEventBatch(
            header=header,
            channels=((highs >> np.uint64(26)) & np.uint64(0x3)).astype(np.uint8),
            refids=(highs & np.uint64(0xFFFF)).astype(np.uint32),
            tstops=((lows >> np.uint64(12)) & np.uint64(0x3FFF)).astype(np.uint32),
            rec_types=((highs >> np.uint64(28)) & np.uint64(0xF)).astype(np.uint8),
            event_classes=((highs >> np.uint64(24)) & np.uint64(0x3)).astype(np.uint8),
            reserved=(lows & np.uint64(0xFFF)).astype(np.uint32),
        )

    @staticmethod
    def _decode_tdc_event_ext_batch(header: PacketHeader, payload_words: List[int]) -> TdcRawEventBatch:
        if len(payload_words) < 3:
            empty_u8 = np.array([], dtype=np.uint8)
            empty_u32 = np.array([], dtype=np.uint32)
            return TdcRawEventBatch(
                header=header,
                channels=empty_u8,
                refids=empty_u32,
                tstops=empty_u32,
                rec_types=empty_u8,
                event_classes=empty_u8,
                reserved=empty_u32,
                timestamps=np.array([], dtype=np.uint64),
            )

        word_count = len(payload_words) - (len(payload_words) % 3)
        words = np.asarray(payload_words[:word_count], dtype=np.uint32)
        lows = words[0::3].astype(np.uint64, copy=False)
        highs = words[1::3].astype(np.uint64, copy=False)
        metas = words[2::3].astype(np.uint64, copy=False)
        event_count = metas.size
        return TdcRawEventBatch(
            header=header,
            channels=((metas >> np.uint64(30)) & np.uint64(0x3)).astype(np.uint8),
            refids=(metas & np.uint64(0xFFFF)).astype(np.uint32),
            tstops=((metas >> np.uint64(16)) & np.uint64(0x3FFF)).astype(np.uint32),
            rec_types=np.full(event_count, 1, dtype=np.uint8),
            event_classes=np.zeros(event_count, dtype=np.uint8),
            reserved=np.zeros(event_count, dtype=np.uint32),
            timestamps=(lows | (highs << np.uint64(32))).astype(np.uint64, copy=False),
        )

    @staticmethod
    def _decode_tdc_events_from_batch(batch: TdcRawEventBatch) -> List[TdcEvent]:
        events: List[TdcEvent] = []
        for idx in range(len(batch)):
            events.append(
                TdcEvent(
                    header=batch.header,
                    rec_type=int(batch.rec_types[idx]),
                    channel=int(batch.channels[idx]),
                    event_class=int(batch.event_classes[idx]),
                    refid=int(batch.refids[idx]),
                    tstop=int(batch.tstops[idx]),
                    reserved=int(batch.reserved[idx]),
                    packet_seq=batch.header.seq,
                    packet_timestamp_us=batch.header.timestamp_us,
                )
            )
        return events

    @staticmethod
    def _decode_photon_events(header: PacketHeader, payload_words: List[int]) -> List[PhotonEvent]:
        events: List[PhotonEvent] = []
        for idx in range(0, len(payload_words), 5):
            if idx + 4 >= len(payload_words):
                break
            marker = payload_words[idx]
            if marker != 0xA55A_F00D:
                continue
            word1 = payload_words[idx + 1]
            word2 = payload_words[idx + 2]
            events.append(
                PhotonEvent(
                    header=header,
                    packet_type=(word1 >> 24) & 0xFF,
                    frame_id=(word1 >> 16) & 0xFF,
                    line_id=word1 & 0xFFFF,
                    pixel_id=(word2 >> 16) & 0xFFFF,
                    bin_index=word2 & 0xFFFF,
                    dt_8ps=payload_words[idx + 3],
                    detector_timestamp_low=payload_words[idx + 4],
                    packet_seq=header.seq,
                    packet_timestamp_us=header.timestamp_us,
                )
            )
        return events


def build_test_tdc_raw_packet(
    ch0_tstop: int = 0x100,
    ch0_refid: int = 0xAAAA,
    ch1_tstop: int = 0x180,
    ch1_refid: int = 0xAAAA,
) -> bytes:
    """Build a synthetic TDC_RAW uplink packet for parser self-test.

    Mimics the FPGA gpx2_raw_event_streamer 2-word-per-event format:
      word0 = {6'd0, tstop[13:0], 12'd0}
      word1 = {4'h1, ch[1:0], 2'b00, 8'd0, refid[15:0]}
    """
    def _event_words(ch: int, tstop: int, refid: int) -> tuple[int, int]:
        w0 = ((tstop & 0x3FFF) << 12)
        w1 = (0x1 << 28) | ((ch & 0x3) << 26) | (refid & 0xFFFF)
        return w0, w1

    w0_start, w1_start = _event_words(0, ch0_tstop & 0x3FFF, ch0_refid & 0xFFFF)
    w0_stop, w1_stop = _event_words(1, ch1_tstop & 0x3FFF, ch1_refid & 0xFFFF)

    payload_words = [w0_start, w1_start, w0_stop, w1_stop]
    payload_bytes = b"".join(w.to_bytes(4, "little") for w in payload_words)

    header_word0 = (SYNC_BYTE_UPLINK << 24) | (PKT_TDC_RAW << 16) | (PROTO_VERSION << 8) | HEADER_WORDS
    header_word1 = (0x0001 << 16) | len(payload_words)
    header_word2 = (2 << 16) | 0
    header_word3 = 12345
    header_bytes = b"".join(
        w.to_bytes(4, "little") for w in [header_word0, header_word1, header_word2, header_word3]
    )
    return header_bytes + payload_bytes


def run_parser_self_test() -> dict:
    """Verify the parser correctly decodes a GPX2 raw TDC packet.

    Returns a dict with test results and decoded event details.
    """
    test_pkt = build_test_tdc_raw_packet(ch0_tstop=0x100, ch0_refid=0xAAAA, ch1_tstop=0x180, ch1_refid=0xAAAA)
    parser = PacketParser()
    packets = parser.feed(test_pkt)

    result = {
        "raw_packet_hex": test_pkt.hex(),
        "raw_packet_len": len(test_pkt),
        "parsed_packet_count": len(packets),
        "ok": False,
        "events": [],
        "error": None,
    }

    if not packets:
        result["error"] = "No packets parsed"
        return result

    pkt = packets[0]
    if pkt.tdc_events is None:
        result["error"] = f"Expected TDC_RAW packet, got pkt_type=0x{pkt.header.pkt_type:02X}"
        return result

    for evt in pkt.tdc_events:
        result["events"].append({
            "channel": evt.channel,
            "event_class": evt.event_class,
            "refid": evt.refid,
            "tstop": evt.tstop,
            "rec_type": evt.rec_type,
        })

    ok = (
        len(pkt.tdc_events) == 2
        and pkt.tdc_events[0].channel == 0
        and pkt.tdc_events[0].tstop == 0x100
        and pkt.tdc_events[0].refid == 0xAAAA
        and pkt.tdc_events[1].channel == 1
        and pkt.tdc_events[1].tstop == 0x180
        and pkt.tdc_events[1].refid == 0xAAAA
    )
    result["ok"] = ok
    if not ok:
        result["error"] = "Decoded event values mismatch"
    return result
