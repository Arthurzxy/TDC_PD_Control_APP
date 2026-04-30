from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MarkerMapping:
    line_event_class: int = 0x1
    pixel_event_class: int = 0x2


@dataclass
class AnalogTargets:
    laser_sync_threshold_mv: float = 100.0
    pixel_sync_threshold_mv: float = 100.0
    avalanche_threshold_mv: float = 100.0
    bias_voltage_v: float = 45.0


@dataclass
class GateSettings:
    hold_off_time: int = 0
    divider: int = 1
    sig2_enable: bool = False
    sig3_enable: bool = False
    pixel_mode: bool = False
    sig2_delay_coarse: int = 0
    sig2_delay_fine: int = 0
    sig2_width_coarse: int = 0
    sig2_width_fine: int = 10
    sig3_delay_coarse: int = 0
    sig3_delay_fine: int = 0
    sig3_width_coarse: int = 0
    sig3_width_fine: int = 10


@dataclass
class HistogramSettings:
    bin_width_raw: int = 1
    bin_offset: int = 0
    bin_count: int = 4096


@dataclass
class AppConfig:
    target_temperature_c: float = 25.0
    analog_targets: AnalogTargets = field(default_factory=AnalogTargets)
    gate_settings: GateSettings = field(default_factory=GateSettings)
    marker_mapping: MarkerMapping = field(default_factory=MarkerMapping)
    histogram_settings: HistogramSettings = field(default_factory=HistogramSettings)
    save_directory: str = "./sessions"
    device_index: int = 0
    read_pipe: int = 0x82
    write_pipe: int = 0x02
    read_block_size: int = 262144
    read_enabled: bool = True
    threshold_range_mv: float = 2500.0
    bias_scale: float = 433.0
    temperature_encode_formula_version: str = "legacy_main_1_py_v1"
    protocol_version: str = "uplink_v1_downlink_legacy_bb"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        analog = AnalogTargets(**data.get("analog_targets", {}))
        gate = GateSettings(**data.get("gate_settings", {}))
        mapping = MarkerMapping(**data.get("marker_mapping", {}))
        hist = HistogramSettings(**data.get("histogram_settings", {}))
        return cls(
            target_temperature_c=data.get("target_temperature_c", 25.0),
            analog_targets=analog,
            gate_settings=gate,
            marker_mapping=mapping,
            histogram_settings=hist,
            save_directory=data.get("save_directory", "./sessions"),
            device_index=data.get("device_index", 0),
            read_pipe=data.get("read_pipe", 0x82),
            write_pipe=data.get("write_pipe", 0x02),
            read_block_size=data.get("read_block_size", 262144),
            read_enabled=data.get("read_enabled", True),
            threshold_range_mv=data.get("threshold_range_mv", 2500.0),
            bias_scale=data.get("bias_scale", 433.0),
            temperature_encode_formula_version=data.get(
                "temperature_encode_formula_version", "legacy_main_1_py_v1"
            ),
            protocol_version=data.get("protocol_version", "uplink_v1_downlink_legacy_bb"),
        )


@dataclass
class PixelParamRecord:
    addr: int
    value36: int
    decoded_fields: Dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    version: str = "v1"


@dataclass
class SessionMetadata:
    session_name: str
    created_at: str
    config_snapshot: Dict[str, Any]
    notes: str = ""
    raw_file_name: str = "capture.tdcpack"
    analysis_file_name: str = "analysis.npz"
    packet_count: int = 0
    tdc_event_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PacketHeader:
    sync: int
    pkt_type: int
    version: int
    hdr_words: int
    seq: int
    payload_words: int
    item_count: int
    flags: int
    timestamp_us: int


@dataclass(frozen=True)
class StatusPacket:
    header: PacketHeader
    flags: int
    uptime_seconds: int
    temp_avg_raw: int
    counter_1s: int
    tdc_drop_count: int
    usb_drop_count: int


@dataclass(frozen=True)
class TdcEvent:
    header: PacketHeader
    rec_type: int
    channel: int
    event_class: int
    refid: int
    tstop: int
    reserved: int
    packet_seq: int
    packet_timestamp_us: int


@dataclass(frozen=True)
class PhotonEvent:
    header: PacketHeader
    packet_type: int
    frame_id: int
    line_id: int
    pixel_id: int
    bin_index: int
    dt_8ps: int
    detector_timestamp_low: int
    packet_seq: int
    packet_timestamp_us: int


@dataclass(frozen=True)
class ParsedPacket:
    header: PacketHeader
    payload_words: List[int]
    status: Optional[StatusPacket] = None
    tdc_events: Optional[List[TdcEvent]] = None
    photon_events: Optional[List[PhotonEvent]] = None
    raw_bytes: bytes = b""


@dataclass(frozen=True)
class CommandResult:
    success: bool
    cmd_id: int
    message: str = ""


@dataclass
class RuntimeStats:
    rx_bytes: int = 0
    tx_bytes: int = 0
    packet_count: int = 0
    tdc_event_count: int = 0
    status_count: int = 0


@dataclass
class PacketRates:
    rx_bytes_per_sec: float = 0.0
    tx_bytes_per_sec: float = 0.0
    packets_per_sec: float = 0.0
    tdc_events_per_sec: float = 0.0


@dataclass
class CommandHistoryItem:
    cmd_id: int
    name: str
    timestamp_us: int
    success: bool
    detail: str = ""


@dataclass
class StatusFlagsDecoded:
    flash_busy: bool = False
    gpx2_lclk_locked: bool = False
    gate_clk_locked: bool = False
    flash_error: bool = False
    usb_tx_backpressure: bool = False
    gpx2_event_overflow: bool = False
    gpx2_cfg_error: bool = False
    gpx2_cfg_done: bool = False


@dataclass
class HistogramSnapshot:
    histograms: Dict[str, List[int]] = field(default_factory=dict)
    image_projection: List[List[int]] = field(default_factory=list)
    current_row: int = 0
    current_col: int = 0
