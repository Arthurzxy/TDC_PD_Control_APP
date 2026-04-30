from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.models import PacketHeader, ParsedPacket, PhotonEvent, StatusPacket, TdcEvent


SYNC_BYTE_DOWNLINK = 0xBB
SYNC_BYTE_UPLINK = 0xA5
PROTO_VERSION = 0x01
HEADER_WORDS = 4

PKT_TDC_RAW = 0x01
PKT_STATUS = 0x02
PKT_PHOTON_EVENT = 0x04

CMD_AD5686 = 0x01
CMD_GATE = 0x02
CMD_NB6L295 = 0x03
CMD_TEC_PID = 0x04
CMD_GPX2_CFG = 0x10
CMD_GATE_DIV = 0x20
CMD_GATE_SIG2 = 0x21
CMD_GATE_SIG3 = 0x22
CMD_GATE_ENABLE = 0x23
CMD_GATE_PIXEL = 0x24
CMD_GATE_RAM = 0x25
CMD_FLASH_SAVE = 0x30
CMD_FLASH_LOAD = 0x31

CMD_NAMES = {
    CMD_AD5686: "CMD_AD5686",
    CMD_GATE: "CMD_GATE",
    CMD_NB6L295: "CMD_NB6L295",
    CMD_TEC_PID: "CMD_TEC_PID",
    CMD_GPX2_CFG: "CMD_GPX2_CFG",
    CMD_GATE_DIV: "CMD_GATE_DIV",
    CMD_GATE_SIG2: "CMD_GATE_SIG2",
    CMD_GATE_SIG3: "CMD_GATE_SIG3",
    CMD_GATE_ENABLE: "CMD_GATE_ENABLE",
    CMD_GATE_PIXEL: "CMD_GATE_PIXEL",
    CMD_GATE_RAM: "CMD_GATE_RAM",
    CMD_FLASH_SAVE: "CMD_FLASH_SAVE",
    CMD_FLASH_LOAD: "CMD_FLASH_LOAD",
}


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def pack_u32_le(word: int) -> bytes:
    return int(word & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False)


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

    def encode_gpx2_config(self) -> bytes:
        return self.build_frame(CMD_GPX2_CFG, [])

    def encode_gate_holdoff(self, holdoff: int) -> bytes:
        return self.build_frame(CMD_GATE, [holdoff & 0xFFFFFF])

    def encode_nb6(self, delay_a: int, delay_b: int, enable: bool) -> bytes:
        word = (delay_a & 0x1FF) | ((delay_b & 0x1FF) << 9) | ((1 if enable else 0) << 18)
        return self.build_frame(CMD_NB6L295, [word])

    def encode_gate_div(self, divider: int) -> bytes:
        return self.build_frame(CMD_GATE_DIV, [divider & 0xFFF])

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

    def encode_flash_save(self) -> bytes:
        return self.build_frame(CMD_FLASH_SAVE, [])

    def encode_flash_load(self) -> bytes:
        return self.build_frame(CMD_FLASH_LOAD, [])


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

    def __init__(self) -> None:
        self.buffer = ByteRingBuffer()

    @staticmethod
    def _u32_at(data: bytes, offset: int) -> int:
        return int.from_bytes(data[offset : offset + 4], byteorder="little", signed=False)

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

            if sync != SYNC_BYTE_UPLINK or hdr_words != HEADER_WORDS:
                self.buffer.discard(1)
                continue

            word1 = self._u32_at(header_bytes, 4)
            word2 = self._u32_at(header_bytes, 8)
            word3 = self._u32_at(header_bytes, 12)
            seq = (word1 >> 16) & 0xFFFF
            payload_words = word1 & 0xFFFF
            item_count = (word2 >> 16) & 0xFFFF
            flags = word2 & 0xFFFF
            total_words = hdr_words + payload_words
            total_bytes = total_words * 4
            if len(self.buffer) < total_bytes:
                break

            packet_bytes = self.buffer.consume(total_bytes)
            payload = [self._u32_at(packet_bytes, 16 + idx * 4) for idx in range(payload_words)]
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
            )
            return ParsedPacket(header=header, payload_words=payload_words, status=status, raw_bytes=raw_bytes)

        if header.pkt_type == PKT_TDC_RAW:
            events = self._decode_tdc_events(header, payload_words)
            return ParsedPacket(header=header, payload_words=payload_words, tdc_events=events, raw_bytes=raw_bytes)

        if header.pkt_type == PKT_PHOTON_EVENT:
            events = self._decode_photon_events(header, payload_words)
            return ParsedPacket(header=header, payload_words=payload_words, photon_events=events, raw_bytes=raw_bytes)

        return ParsedPacket(header=header, payload_words=payload_words, raw_bytes=raw_bytes)

    @staticmethod
    def _decode_tdc_events(header: PacketHeader, payload_words: List[int]) -> List[TdcEvent]:
        events: List[TdcEvent] = []
        for idx in range(0, len(payload_words), 2):
            if idx + 1 >= len(payload_words):
                break
            low = payload_words[idx]
            high = payload_words[idx + 1]
            record = low | (high << 32)
            events.append(
                TdcEvent(
                    header=header,
                    rec_type=(record >> 60) & 0xF,
                    channel=(record >> 58) & 0x3,
                    event_class=(record >> 56) & 0x3,
                    refid=(record >> 32) & 0xFFFFFF,
                    tstop=(record >> 12) & 0xFFFFF,
                    reserved=record & 0xFFF,
                    packet_seq=header.seq,
                    packet_timestamp_us=header.timestamp_us,
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
