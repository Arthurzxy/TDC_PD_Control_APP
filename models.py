from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


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
class TdcTestSettings:
    enabled: bool = False
    start_channel: int = 0
    stop_channel: int = 1
    source: str = "gpx2_fpga"
    pairing_mode: str = "prev_start"
    time_formula: str = "forward"
    refclk_divisions: int = 12500
    bin_width_raw: int = 1
    bin_offset: int = 0
    bin_count: int = 4096
    acquisition_time_s: float = 1.0
    tdc_deadtime_ps: float = 0.0
    reference_deadtime_ns: float = 0.0
    detector_deadtime_ns: float = 100.0
    raw_epoch: int = 0


@dataclass
class AppConfig:
    target_temperature_c: float = 25.0
    analog_targets: AnalogTargets = field(default_factory=AnalogTargets)
    gate_settings: GateSettings = field(default_factory=GateSettings)
    marker_mapping: MarkerMapping = field(default_factory=MarkerMapping)
    histogram_settings: HistogramSettings = field(default_factory=HistogramSettings)
    tdc_test_settings: TdcTestSettings = field(default_factory=TdcTestSettings)
    save_directory: str = "./sessions"
    device_index: int = 0
    read_pipe: int = 0x82
    write_pipe: int = 0x02
    read_block_size: int = 262144
    read_enabled: bool = False
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
        tdc_test_data = dict(data.get("tdc_test_settings", {}))
        if tdc_test_data.get("pairing_mode") == "all_start" and "tdc_deadtime_ps" not in tdc_test_data:
            tdc_test_data["pairing_mode"] = "prev_start"
        tdc_test = TdcTestSettings(**tdc_test_data)
        return cls(
            target_temperature_c=data.get("target_temperature_c", 25.0),
            analog_targets=analog,
            gate_settings=gate,
            marker_mapping=mapping,
            histogram_settings=hist,
            tdc_test_settings=tdc_test,
            save_directory=data.get("save_directory", "./sessions"),
            device_index=data.get("device_index", 0),
            read_pipe=data.get("read_pipe", 0x82),
            write_pipe=data.get("write_pipe", 0x02),
            read_block_size=data.get("read_block_size", 262144),
            read_enabled=data.get("read_enabled", False),
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
    photon_event_count: int = 0

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
    gpx2_raw_count_ch1: int = 0
    gpx2_raw_count_ch2: int = 0
    gpx2_raw_count_ch3: int = 0
    gpx2_raw_count_ch4: int = 0
    gpx2_cfg_diag: int = 0
    gpx2_cfg_readback: tuple[int, ...] = ()
    gate_trigger_count: int = 0
    gate_direct_pulse_count: int = 0
    gate_divided_pulse_count: int = 0
    gate_output_pulse_count: int = 0


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
class TdcRawEventBatch:
    header: PacketHeader
    channels: np.ndarray
    refids: np.ndarray
    tstops: np.ndarray
    rec_types: np.ndarray
    event_classes: np.ndarray
    reserved: np.ndarray
    timestamps: Optional[np.ndarray] = None
    raw_epoch: int = 0
    raw_sequence: int = 0
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None

    def __len__(self) -> int:
        return int(self.channels.size)


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
class TdcHistogramChunk:
    header: PacketHeader
    snapshot_id: int
    hist_id: int
    bin_start: int
    bin_count: int
    total_bins: int
    bin_width_raw: int
    offset_raw: int
    status_flags: int
    start_count: int
    stop_count: int
    accepted_count: int
    out_of_window_count: int
    no_start_count: int
    first_stop_suppressed_count: int
    last_dt_raw: int
    reference_cleanup_count: int
    tdc_deadtime_filtered_count: int
    bins: np.ndarray


@dataclass(frozen=True)
class Ddr3BistResult:
    header: PacketHeader
    mode: int
    calibrated: bool
    running: bool
    done: bool
    passed: bool
    failed: bool
    requested_words: int
    tested_words: int
    error_count: int
    first_error_addr: int
    expected: int
    actual: int
    write_cycles: int
    read_cycles: int
    app_addr_width: int
    app_data_width: int


@dataclass(frozen=True)
class ParsedPacket:
    header: PacketHeader
    payload_words: List[int]
    status: Optional[StatusPacket] = None
    tdc_events: Optional[List[TdcEvent]] = None
    tdc_event_batch: Optional[TdcRawEventBatch] = None
    photon_events: Optional[List[PhotonEvent]] = None
    histogram_chunk: Optional[TdcHistogramChunk] = None
    ddr3_bist_result: Optional[Ddr3BistResult] = None
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
    tdc_raw_packet_count: int = 0
    tdc_raw_event_count: int = 0
    last_tdc_raw_packet_age_ms: float = -1.0
    last_tdc_raw_channels: tuple[int, int, int, int] = (0, 0, 0, 0)
    photon_event_count: int = 0
    status_count: int = 0
    rx_reader_enabled: bool = False
    rx_reader_running: bool = False


@dataclass
class PacketRates:
    rx_bytes_per_sec: float = 0.0
    tx_bytes_per_sec: float = 0.0
    packets_per_sec: float = 0.0
    tdc_events_per_sec: float = 0.0
    photon_events_per_sec: float = 0.0


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
    histograms: Dict[str, Any] = field(default_factory=dict)
    image_projection: List[List[int]] = field(default_factory=list)
    current_row: int = 0
    current_col: int = 0
