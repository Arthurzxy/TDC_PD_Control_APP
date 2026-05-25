from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from app import protocol
from app.data_processing import TcspcCommercialAnalyzer, TcspcTimeReconstructor
from app.models import HistogramSnapshot, PixelParamRecord, StatusPacket, TdcTestSettings


class MainWindow(QtWidgets.QMainWindow):
    """现代化 PyQt5 主窗口。

    目标：
    - 保持现有 FPGA 控制链和采集链不变
    - 升级为更适合实验室联调的现代化界面
    - 增加温度与 CPS 的实时监控卡片和小型趋势图
    """

    def __init__(self) -> None:
        super().__init__()
        self.controller = None
        self._device_entries: list[tuple[int, str]] = []
        self._monitor_tick = 0
        self._gpx2_register_edits: list[QtWidgets.QLineEdit] = []
        self._gpx2_readback_items: list[QtWidgets.QTableWidgetItem] = []
        self._gpx2_status_items: list[QtWidgets.QTableWidgetItem] = []
        self._gpx2_readback_pending = False
        self._last_gpx2_readback: tuple[int, ...] = ()
        self._last_gpx2_render_signature = None
        self._last_status_ui_update_monotonic = 0.0
        self._last_status_packet: StatusPacket | None = None
        self._last_runtime_stats = None
        self._last_runtime_rates: dict = {}
        self._last_tdc_test_label_core = ""
        self._tdc_test_usb_drop_baseline: int | None = None
        self._tdc_test_upload_payload_word: int | None = None
        self._gpx2_readback_status_count_at_request = -1
        self._gpx2_readback_temp_rx = False
        self.temp_history: deque[float] = deque(maxlen=120)
        self.cps_history: deque[int] = deque(maxlen=120)
        self.monitor_index: deque[int] = deque(maxlen=120)
        self._tdc_test_timer = QtCore.QTimer(self)
        self._tdc_test_timer.setSingleShot(True)
        self._tdc_test_timer.timeout.connect(self.on_stop_tdc_test)
        self._tdc_test_plot_max_points = 6000
        self._gpx2_readback_timer = QtCore.QTimer(self)
        self._gpx2_readback_timer.setSingleShot(True)
        self._gpx2_readback_timer.timeout.connect(self.on_gpx2_readback_timeout)
        self._sig3_sweep_timer = QtCore.QTimer(self)
        self._sig3_sweep_timer.setInterval(50)
        self._sig3_sweep_timer.timeout.connect(self._on_sig3_sweep_timer)
        self._sig3_sweep_active = False
        self._sig3_sweep_points: list[int] = []
        self._sig3_sweep_rows: list[dict[str, int]] = []
        self._sig3_sweep_index = 0
        self._sig3_sweep_phase = "idle"
        self._sig3_sweep_config_seq: int | None = None
        self._sig3_sweep_baseline: StatusPacket | None = None
        self._sig3_sweep_deadline = 0.0
        self._sig3_sweep_timeout = 0.0
        self._sig3_sweep_restore_state: tuple[int, int, bool] | None = None
        self.sig3_sweep_window: QtWidgets.QDialog | None = None

        pg.setConfigOptions(antialias=False, background="#0f172a", foreground="#e2e8f0")

        self.setWindowTitle("TDC Host V1")
        self.setMinimumSize(1500, 940)
        self.resize(1760, 1080)
        self._apply_modern_style()
        self._build_ui()

    def bind_controller(self, controller) -> None:
        self.controller = controller
        self._bind_signals()
        self._load_defaults_from_config()
        QtCore.QTimer.singleShot(0, self.on_refresh_devices)

    def _apply_modern_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #08111d;
                color: #e7edf7;
                font-family: "Microsoft YaHei UI", "Segoe UI";
                font-size: 14px;
            }
            QLabel {
                background: transparent;
            }
            QLabel#WindowTitle {
                font-size: 30px;
                font-weight: 700;
                color: #f8fafc;
            }
            QLabel#WindowSubtitle {
                color: #a0afc4;
                font-size: 14px;
            }
            QLabel#CardTitle {
                color: #f8fafc;
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#InfoText {
                color: #9fb0c8;
            }
            QStatusBar {
                background: #0b1322;
                border-top: 1px solid #243144;
                color: #9fb0c8;
            }
            QLabel#StatusChip {
                background: #3f1d1d;
                color: #fecaca;
                border: 1px solid #7f1d1d;
                border-radius: 14px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
                min-width: 120px;
            }
            QLabel#StatusChip[connected="true"] {
                background: #123223;
                color: #bbf7d0;
                border: 1px solid #166534;
            }
            QTabWidget::pane {
                border: 1px solid #22314a;
                border-radius: 18px;
                background: #0d1728;
                top: -1px;
            }
            QTabBar::tab {
                background: #101a2b;
                color: #94a3b8;
                border: 1px solid #22314a;
                border-bottom: none;
                padding: 12px 20px;
                margin-right: 8px;
                min-width: 124px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #1d4ed8;
                color: #eff6ff;
            }
            QTabBar::tab:hover {
                color: #f8fafc;
            }
            QGroupBox {
                border: 1px solid #243144;
                border-radius: 18px;
                margin-top: 16px;
                padding: 18px 18px 20px 18px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #101827, stop:1 #162133);
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                top: 5px;
                padding: 0 8px;
                color: #d7e6fb;
                font-size: 15px;
            }
            QFrame#Card {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #101827, stop:1 #141f31);
                border: 1px solid #243144;
                border-radius: 18px;
            }
            QFrame#HeroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #12223c, stop:0.55 #0f1b31, stop:1 #0d1628);
                border: 1px solid #2c4263;
                border-radius: 20px;
            }
            QFrame#MetricCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #111a2b, stop:1 #1b2942);
                border: 1px solid #2b4262;
                border-radius: 20px;
            }
            QLabel#MetricTitle {
                color: #9ac8ff;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#MetricValue {
                color: #f8fafc;
                font-size: 34px;
                font-weight: 700;
            }
            QLabel#MetricDetail {
                color: #a0afc4;
                font-size: 13px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #2563eb, stop:1 #1d4ed8);
                border: 1px solid #3b82f6;
                border-radius: 12px;
                padding: 10px 16px;
                min-height: 40px;
                color: white;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #3b82f6;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
            QPushButton#Secondary {
                background: #192536;
                border: 1px solid #334155;
                color: #e2e8f0;
            }
            QPushButton#Secondary:hover {
                background: #273449;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 7px 10px;
                color: #f8fafc;
                selection-background-color: #2563eb;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 38px;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QPlainTextEdit#StatusPanel, QPlainTextEdit#LogPanel {
                font-family: "Cascadia Mono", "Consolas";
                font-size: 14px;
                background: #0b1322;
                border-color: #30445f;
            }
            QHeaderView::section {
                background: #172033;
                color: #cbd5e1;
                border: none;
                padding: 10px 12px;
                font-size: 13px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #243144;
                alternate-background-color: #0c1525;
                border-radius: 14px;
            }
            QCheckBox {
                spacing: 10px;
                color: #dbe5f4;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 4px;
                border: 1px solid #475569;
                background: #0f172a;
            }
            QCheckBox::indicator:checked {
                border-radius: 4px;
                border: 1px solid #2563eb;
                background: #2563eb;
            }
            QSplitter::handle {
                background: #162236;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background: #0c1525;
                width: 12px;
                margin: 4px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar:horizontal {
                background: #0c1525;
                height: 12px;
                margin: 4px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #334155;
                min-width: 30px;
                border-radius: 6px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(16)

        header = self._build_header()
        root.addWidget(header)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        root.addWidget(self.tabs, 1)

        self._init_gpx2_controls()
        self.connection_tab = self._build_connection_tab()
        self.control_tab = self._build_control_tab()
        self.gpx2_tab = self._build_gpx2_tab()
        self.pixel_tab = self._build_pixel_tab()
        self.tdc_test_tab = self._build_tdc_test_tab()
        self.acquisition_tab = self._build_acquisition_tab()
        self.offline_tab = self._build_offline_tab()
        self.log_tab = self._build_log_tab()

        self.tabs.addTab(self.connection_tab, "设备连接")
        self.tabs.addTab(self.control_tab, "FPGA 控制")
        self.tabs.addTab(self.gpx2_tab, "GPX2 调试")
        self.tabs.addTab(self.pixel_tab, "像素阵列")
        self.tabs.addTab(self.tdc_test_tab, "TCSPC")
        self.tabs.addTab(self.acquisition_tab, "实时采集")
        self.tabs.addTab(self.offline_tab, "离线分析")
        self.tabs.addTab(self.log_tab, "日志调试")
        self._build_global_usb_status_bar()

    def _build_global_usb_status_bar(self) -> None:
        bar = QtWidgets.QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)
        self.global_rx_rate_label = QtWidgets.QLabel("USB RX: 0.0 kB/s")
        self.global_tx_rate_label = QtWidgets.QLabel("USB TX: 0.0 kB/s")
        self.global_packet_rate_label = QtWidgets.QLabel("Packets: 0.0/s")
        self.global_raw_event_rate_label = QtWidgets.QLabel("Raw events: 0.0/s")
        for label in (
            self.global_rx_rate_label,
            self.global_tx_rate_label,
            self.global_packet_rate_label,
            self.global_raw_event_rate_label,
        ):
            label.setObjectName("InfoText")
            label.setMinimumWidth(150)
            bar.addPermanentWidget(label)

    def _build_header(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("HeroCard")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(18)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(6)
        title = QtWidgets.QLabel("TDC Host Control Center")
        title.setObjectName("WindowTitle")
        subtitle = QtWidgets.QLabel("FT601 通信、FPGA 控制、实时采集与离线回放")
        subtitle.setObjectName("WindowSubtitle")
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        layout.addLayout(title_col)
        layout.addStretch(1)

        self.header_status_chip = QtWidgets.QLabel("未连接")
        self.header_status_chip.setObjectName("StatusChip")
        self.header_status_chip.setAlignment(QtCore.Qt.AlignCenter)
        self.header_status_chip.setMinimumWidth(120)
        layout.addWidget(self.header_status_chip)
        self._set_header_connection_state(False)
        return frame

    def _make_card_title(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("CardTitle")
        return label

    def _set_header_connection_state(self, connected: bool) -> None:
        self.header_status_chip.setText("ONLINE" if connected else "DISCONNECTED")
        self.header_status_chip.setProperty("connected", connected)
        style = self.header_status_chip.style()
        style.unpolish(self.header_status_chip)
        style.polish(self.header_status_chip)
        self.header_status_chip.update()

    def _style_plot_widget(self, plot: pg.PlotWidget, bottom_label: str) -> None:
        plot.setBackground("#0b1322")
        plot.showGrid(x=True, y=True, alpha=0.18)
        item = plot.getPlotItem()
        item.hideButtons()
        item.setMenuEnabled(False)
        item.setLabel("bottom", bottom_label)
        for axis_name in ("left", "bottom"):
            axis = item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#50627e"))
            axis.setTextPen(pg.mkPen("#cbd5e1"))

    def _build_connection_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(14)
        top_row.addWidget(self._build_connection_config_card(), 3)
        top_row.addWidget(self._build_link_stats_card(), 2)
        layout.addLayout(top_row)

        bottom_row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.status_flags_view = QtWidgets.QPlainTextEdit()
        self.status_flags_view.setObjectName("StatusPanel")
        self.status_flags_view.setReadOnly(True)
        self.status_flags_view.setPlaceholderText("最近 STATUS 摘要")
        self.self_test_view = QtWidgets.QPlainTextEdit()
        self.self_test_view.setObjectName("StatusPanel")
        self.self_test_view.setReadOnly(True)
        self.self_test_view.setPlaceholderText("联板自检结果")
        bottom_row.addWidget(self._wrap_card("状态摘要", self.status_flags_view))
        bottom_row.addWidget(self._wrap_card("联板自检", self.self_test_view))
        bottom_row.setSizes([820, 520])
        layout.addWidget(bottom_row, 1)
        return tab

    def _build_connection_config_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("FT601 连接参数")
        title.setObjectName("CardTitle")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)

        self.refresh_devices_btn = QtWidgets.QPushButton("刷新设备")
        self.refresh_devices_btn.setObjectName("Secondary")
        self.device_combo = QtWidgets.QComboBox()
        self.connect_btn = QtWidgets.QPushButton("连接 FT601")
        self.disconnect_btn = QtWidgets.QPushButton("断开设备")
        self.disconnect_btn.setObjectName("Secondary")
        self.device_status_label = QtWidgets.QLabel("未连接")

        self.read_pipe_edit = QtWidgets.QLineEdit("0x82")
        self.write_pipe_edit = QtWidgets.QLineEdit("0x02")
        self.read_block_spin = QtWidgets.QSpinBox()
        self.read_block_spin.setRange(4 * 1024, 4 * 1024 * 1024)
        self.read_block_spin.setSingleStep(4 * 1024)
        self.read_block_spin.setValue(256 * 1024)
        self.read_block_spin.setSuffix(" B")
        self.read_enable_check = QtWidgets.QCheckBox("读取数据")
        self.read_enable_check.setChecked(False)

        self.self_test_btn = QtWidgets.QPushButton("运行联板自检")
        self.save_config_btn = QtWidgets.QPushButton("保存当前配置")
        self.save_config_btn.setObjectName("Secondary")

        grid.addWidget(QtWidgets.QLabel("设备"), 0, 0)
        grid.addWidget(self.device_combo, 0, 1, 1, 2)
        grid.addWidget(self.refresh_devices_btn, 0, 3)
        grid.addWidget(self.connect_btn, 0, 4)
        grid.addWidget(self.disconnect_btn, 0, 5)

        grid.addWidget(QtWidgets.QLabel("读 Pipe"), 1, 0)
        grid.addWidget(self.read_pipe_edit, 1, 1)
        grid.addWidget(QtWidgets.QLabel("写 Pipe"), 1, 2)
        grid.addWidget(self.write_pipe_edit, 1, 3)
        grid.addWidget(QtWidgets.QLabel("读块大小"), 1, 4)
        grid.addWidget(self.read_block_spin, 1, 5)

        grid.addWidget(QtWidgets.QLabel("状态"), 2, 0)
        grid.addWidget(self.device_status_label, 2, 1)
        grid.addWidget(self.read_enable_check, 2, 2)
        grid.addWidget(self.self_test_btn, 2, 4)
        grid.addWidget(self.save_config_btn, 2, 5)
        layout.addLayout(grid)
        return frame

    def _build_link_stats_card(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("链路速率")
        title.setObjectName("CardTitle")
        layout.addWidget(title)

        stats_grid = QtWidgets.QGridLayout()
        stats_grid.setHorizontalSpacing(12)
        stats_grid.setVerticalSpacing(12)
        stats_grid.setColumnStretch(0, 1)
        stats_grid.setColumnStretch(1, 1)
        self.rx_rate_label = self._make_stat_pill("USB RX", "0.0 kB/s")
        self.tx_rate_label = self._make_stat_pill("USB TX", "0.0 kB/s")
        self.packet_rate_label = self._make_stat_pill("PACKETS", "0 /s")
        self.event_rate_label = self._make_stat_pill("EVENTS", "0 /s")
        stats_grid.addWidget(self.rx_rate_label.parentWidget(), 0, 0)
        stats_grid.addWidget(self.tx_rate_label.parentWidget(), 0, 1)
        stats_grid.addWidget(self.packet_rate_label.parentWidget(), 1, 0)
        stats_grid.addWidget(self.event_rate_label.parentWidget(), 1, 1)
        layout.addLayout(stats_grid)
        layout.addStretch(1)
        return frame

    def _init_gpx2_controls(self) -> None:
        self.gpx2_apply_hex_btn = QtWidgets.QPushButton("从 HEX 填充")
        self.gpx2_load_default_btn = QtWidgets.QPushButton("恢复默认")
        self.gpx2_copy_readback_btn = QtWidgets.QPushButton("回读覆盖编辑")
        self.gpx2_run_profile_btn = QtWidgets.QPushButton("运行预设序列")
        self.gpx2_run_custom_sequence_btn = QtWidgets.QPushButton("运行当前自定义序列")
        for button in (
            self.gpx2_apply_hex_btn,
            self.gpx2_load_default_btn,
            self.gpx2_copy_readback_btn,
            self.gpx2_run_profile_btn,
        ):
            button.setObjectName("Secondary")

        self.gpx2_result_label = QtWidgets.QLabel("GPX2: -")
        self.gpx2_result_label.setObjectName("InfoText")
        self.gpx2_diag_label = QtWidgets.QLabel("最近回读: -")
        self.gpx2_diag_label.setObjectName("InfoText")
        self.gpx2_readback_hex_view = QtWidgets.QPlainTextEdit()
        self.gpx2_readback_hex_view.setObjectName("StatusPanel")
        self.gpx2_readback_hex_view.setReadOnly(True)
        self.gpx2_readback_hex_view.setPlaceholderText("最近一次 0x40 配置回读会显示在这里")

        self.gpx2_register_table = QtWidgets.QTableWidget(len(protocol.GPX2_REGISTER_SPECS), 6)
        self.gpx2_register_table.setHorizontalHeaderLabels(
            ["Addr", "Register", "Planned", "Readback", "Status", "Manual note"]
        )
        self.gpx2_register_table.verticalHeader().setVisible(False)
        self.gpx2_register_table.verticalHeader().setDefaultSectionSize(42)
        self.gpx2_register_table.setAlternatingRowColors(True)
        self.gpx2_register_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.gpx2_register_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.gpx2_register_table.setFocusPolicy(QtCore.Qt.NoFocus)
        header = self.gpx2_register_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)

    def _build_gpx2_tab(self) -> QtWidgets.QWidget:
        self.gpx2_init_btn.setText("INIT 0x18")
        self.gpx2_custom_apply_btn.setText("写当前配置 0x80")
        self.gpx2_lvds_test_btn.setText("填充 LVDS Test")
        self.gpx2_reset_btn.setText("POWER/Reset 0x30")
        self.gpx2_read_cfg_btn.setText("读取配置 0x40")
        self._set_gpx2_register_values(protocol.GPX2_DEFAULT_CONFIG_BYTES)

        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        ops_card = QtWidgets.QFrame()
        ops_card.setObjectName("Card")
        ops_layout = QtWidgets.QGridLayout(ops_card)
        ops_layout.setContentsMargins(14, 14, 14, 14)
        ops_layout.setHorizontalSpacing(12)
        ops_layout.setVerticalSpacing(12)

        ops_layout.addWidget(QtWidgets.QLabel("原始 17 字节"), 0, 0)
        ops_layout.addWidget(self.gpx2_custom_hex_edit, 0, 1, 1, 5)
        ops_layout.addWidget(self.gpx2_apply_hex_btn, 0, 6)

        ops_layout.addWidget(self.gpx2_load_default_btn, 1, 0)
        ops_layout.addWidget(self.gpx2_lvds_test_btn, 1, 1)
        ops_layout.addWidget(self.gpx2_copy_readback_btn, 1, 2)
        ops_layout.addWidget(self.gpx2_custom_apply_btn, 1, 3)
        ops_layout.addWidget(self.gpx2_read_cfg_btn, 1, 4)
        ops_layout.addWidget(self.gpx2_reset_btn, 1, 5)
        ops_layout.addWidget(self.gpx2_init_btn, 1, 6)

        ops_layout.addWidget(QtWidgets.QLabel("预设 profile"), 2, 0)
        ops_layout.addWidget(self.gpx2_profile_combo, 2, 1, 1, 2)
        ops_layout.addWidget(QtWidgets.QLabel("高级 sequence"), 2, 3)
        ops_layout.addWidget(self.gpx2_sequence_combo, 2, 4, 1, 2)
        ops_layout.addWidget(self.gpx2_run_profile_btn, 2, 6)
        ops_layout.addWidget(self.gpx2_diag_label, 3, 0, 1, 3)
        ops_layout.addWidget(self.gpx2_result_label, 3, 3, 1, 3)
        ops_layout.addWidget(self.gpx2_run_custom_sequence_btn, 3, 6)
        layout.addWidget(ops_card)

        table_card = QtWidgets.QFrame()
        table_card.setObjectName("Card")
        table_layout = QtWidgets.QVBoxLayout(table_card)
        table_layout.setContentsMargins(14, 14, 14, 14)
        table_layout.setSpacing(10)
        table_layout.addWidget(self._make_card_title("寄存器编辑"))
        table_layout.addWidget(self.gpx2_register_table)
        layout.addWidget(table_card, 1)

        readback_card = QtWidgets.QFrame()
        readback_card.setObjectName("Card")
        readback_layout = QtWidgets.QVBoxLayout(readback_card)
        readback_layout.setContentsMargins(14, 14, 14, 14)
        readback_layout.setSpacing(10)
        readback_layout.addWidget(self._make_card_title("配置回读"))
        readback_layout.addWidget(self.gpx2_readback_hex_view, 1)
        layout.addWidget(readback_card)
        return tab

    def _build_control_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        analog_group = QtWidgets.QGroupBox("温控 / 偏压 / 甄别阈值")
        analog_grid = QtWidgets.QGridLayout(analog_group)
        analog_grid.setHorizontalSpacing(12)
        analog_grid.setVerticalSpacing(12)

        self.temp_target_spin = QtWidgets.QDoubleSpinBox()
        self.temp_target_spin.setRange(-50.0, 80.0)
        self.temp_target_spin.setDecimals(2)
        self.temp_target_spin.setSuffix(" °C")
        self.temp_set_btn = QtWidgets.QPushButton("设置目标温度")
        self.temp_code_label = QtWidgets.QLabel("raw: 0")
        self.temp_current_label = QtWidgets.QLabel("当前温度: -- °C")

        self.laser_thr_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_thr_spin = QtWidgets.QDoubleSpinBox()
        self.avalanche_thr_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.laser_thr_spin, self.pixel_thr_spin, self.avalanche_thr_spin):
            spin.setRange(0.0, 2500.0)
            spin.setDecimals(2)
            spin.setSuffix(" mV")

        self.bias_spin = QtWidgets.QDoubleSpinBox()
        self.bias_spin.setRange(0.0, 100.0)
        self.bias_spin.setDecimals(3)
        self.bias_spin.setSuffix(" V")
        self.save_analog_flash_btn = QtWidgets.QPushButton("下发并保存温度/阈值/偏压到 Flash")
        self.apply_analog_btn = QtWidgets.QPushButton("下发四路 DAC")
        self.analog_code_label = QtWidgets.QLabel("codes: ch1=0 ch2=0 ch3=0 ch4=0")

        analog_grid.addWidget(QtWidgets.QLabel("目标温度"), 0, 0)
        analog_grid.addWidget(self.temp_target_spin, 0, 1)
        analog_grid.addWidget(self.temp_set_btn, 0, 2)
        analog_grid.addWidget(self.temp_code_label, 0, 3)
        analog_grid.addWidget(self.temp_current_label, 0, 4)

        analog_grid.addWidget(QtWidgets.QLabel("激光同步阈值"), 1, 0)
        analog_grid.addWidget(self.laser_thr_spin, 1, 1)
        analog_grid.addWidget(QtWidgets.QLabel("像素同步阈值"), 1, 2)
        analog_grid.addWidget(self.pixel_thr_spin, 1, 3)
        analog_grid.addWidget(QtWidgets.QLabel("雪崩阈值"), 2, 0)
        analog_grid.addWidget(self.avalanche_thr_spin, 2, 1)
        analog_grid.addWidget(QtWidgets.QLabel("偏压控制"), 2, 2)
        analog_grid.addWidget(self.bias_spin, 2, 3)
        analog_grid.addWidget(self.apply_analog_btn, 3, 0, 1, 2)
        analog_grid.addWidget(self.analog_code_label, 3, 2, 1, 3)
        analog_grid.addWidget(self.save_analog_flash_btn, 4, 0, 1, 3)
        layout.addWidget(analog_group)

        fpga_group = QtWidgets.QGroupBox("核心控制")
        fpga_grid = QtWidgets.QGridLayout(fpga_group)
        fpga_grid.setHorizontalSpacing(12)
        fpga_grid.setVerticalSpacing(12)

        self.gpx2_init_btn = QtWidgets.QPushButton("初始化 GPX2")
        self.gpx2_profile_combo = QtWidgets.QComboBox()
        self.gpx2_profile_combo.addItem("0 Full normal", 0)
        self.gpx2_profile_combo.addItem("1 Power/core only", 1)
        self.gpx2_profile_combo.addItem("2 REF only", 2)
        self.gpx2_profile_combo.addItem("3 STOP+REF no LVDS", 3)
        self.gpx2_profile_combo.addItem("4 LVDS out only", 4)
        self.gpx2_profile_combo.addItem("5 Pins on, hits off", 5)
        self.gpx2_profile_combo.addItem("6 LVDS test pattern", 6)
        self.gpx2_profile_combo.addItem("7 Legacy NLOS 10 ps", 7)
        self.gpx2_profile_combo.addItem("8 All-zero registers", 8)
        self.gpx2_profile_combo.addItem("9 CSDN 1ch CMOS ref", 9)
        self.gpx2_profile_combo.addItem("10 LVDS CH1+REF no out", 10)
        self.gpx2_profile_combo.addItem("11 LVDS CH1+REF+out", 11)
        self.gpx2_profile_combo.addItem("12 STOP+REF hits off", 12)
        self.gpx2_profile_combo.addItem("13 Minimal legal idle", 13)
        self.gpx2_profile_combo.addItem("14 Minimal legal REF only", 14)
        self.gpx2_profile_combo.addItem("15 Minimal legal STOP1+REF", 15)
        self.gpx2_sequence_combo = QtWidgets.QComboBox()
        self.gpx2_sequence_combo.addItem("0 Diagnostic fast full seq, readback-gated INIT", 0)
        self.gpx2_sequence_combo.addItem("1 Read config only", 1)
        self.gpx2_sequence_combo.addItem("2 POWER only", 2)
        self.gpx2_sequence_combo.addItem("3 Write/read no POWER or INIT", 3)
        self.gpx2_sequence_combo.addItem("4 INIT only", 4)
        self.gpx2_sequence_combo.addItem("5 POWER + write/read, no INIT", 5)
        self.gpx2_sequence_combo.addItem("6 POWER + INIT, no write", 6)
        self.gpx2_sequence_combo.addItem("7 Write/read + INIT, no POWER", 7)
        self.gpx2_sequence_combo.addItem("8 Write + INIT, no readback", 8)
        self.gpx2_sequence_combo.addItem("9 Manual slow 0x30 + 0x40 read", 9)
        self.gpx2_sequence_combo.addItem("10 Manual slow full config + INIT, no readback", 10)
        self.gpx2_sequence_combo.addItem("11 Manual slow config write only", 11)
        self.gpx2_sequence_combo.addItem("12 Manual slow POWER + INIT only", 12)
        self.gpx2_sequence_combo.addItem("13 Manual slow POWER + 1s + INIT", 13)
        self.gpx2_sequence_combo.addItem("14 Manual slow INIT only", 14)
        self.gpx2_sequence_combo.addItem("15 Manual slow POWER + read + INIT", 15)
        self.gpx2_sequence_combo.setCurrentIndex(self.gpx2_sequence_combo.findData(10))
        self.gpx2_custom_hex_edit = QtWidgets.QLineEdit(
            " ".join(f"{value:02X}" for value in protocol.GPX2_DEFAULT_CONFIG_BYTES)
        )
        self.gpx2_custom_hex_edit.setPlaceholderText("17 bytes: reg0 ... reg16")
        self.gpx2_custom_apply_btn = QtWidgets.QPushButton("GPX2 custom write")
        self.gpx2_lvds_test_btn = QtWidgets.QPushButton("LVDS test pattern")
        self.gpx2_reset_btn = QtWidgets.QPushButton("GPX2 0x30 reset")
        self.gpx2_read_cfg_btn = QtWidgets.QPushButton("Read GPX2 config")
        self.flash_save_btn = QtWidgets.QPushButton("FLASH_SAVE")
        self.flash_load_btn = QtWidgets.QPushButton("FLASH_LOAD")

        self.gate_holdoff_spin = QtWidgets.QSpinBox()
        self.gate_holdoff_spin.setRange(0, (1 << 24) - 1)
        self.gate_holdoff_btn = QtWidgets.QPushButton("设置 Holdoff")

        self.gate_div_spin = QtWidgets.QSpinBox()
        self.gate_div_spin.setRange(1, 4096)
        self.gate_div_btn = QtWidgets.QPushButton("设置 Gate 分频")

        self.nb6_a_spin = QtWidgets.QSpinBox()
        self.nb6_b_spin = QtWidgets.QSpinBox()
        for spin in (self.nb6_a_spin, self.nb6_b_spin):
            spin.setRange(0, 511)
        self.nb6_enable_check = QtWidgets.QCheckBox("NB6 使能")
        self.nb6_btn = QtWidgets.QPushButton("设置 NB6")

        self.sig2_enable_check = QtWidgets.QCheckBox("SIG2 使能")
        self.sig3_enable_check = QtWidgets.QCheckBox("SIG3 使能")
        self.pixel_mode_check = QtWidgets.QCheckBox("像素模式")
        self.gate_enable_btn = QtWidgets.QPushButton("下发 Gate 使能")

        self.sig2_delay_coarse = QtWidgets.QSpinBox()
        self.sig2_delay_fine = QtWidgets.QSpinBox()
        self.sig2_width_coarse = QtWidgets.QSpinBox()
        self.sig2_width_fine = QtWidgets.QSpinBox()
        self.sig3_delay_coarse = QtWidgets.QSpinBox()
        self.sig3_delay_fine = QtWidgets.QSpinBox()
        self.sig3_width_coarse = QtWidgets.QSpinBox()
        self.sig3_width_fine = QtWidgets.QSpinBox()

        for spin in (self.sig2_delay_coarse, self.sig3_delay_coarse):
            spin.setRange(0, 15)
        for spin in (self.sig2_delay_fine, self.sig3_delay_fine):
            spin.setRange(0, 9)
        for spin in (self.sig2_width_fine, self.sig3_width_fine):
            spin.setRange(0, 31)
        for spin in (self.sig2_width_coarse, self.sig3_width_coarse):
            spin.setRange(0, 7)
        self.sig2_width_fine.setValue(10)
        self.sig3_width_fine.setValue(10)

        self.sig2_apply_btn = QtWidgets.QPushButton("设置 SIG2")
        self.sig3_apply_btn = QtWidgets.QPushButton("设置 SIG3")
        self.control_result_label = QtWidgets.QLabel("最近命令: -")
        self.control_result_label.setObjectName("InfoText")
        self.sig3_sweep_open_btn = QtWidgets.QPushButton("打开 SIG3 扫门窗口")

        self.sig3_sweep_start_ns_spin = QtWidgets.QSpinBox()
        self.sig3_sweep_stop_ns_spin = QtWidgets.QSpinBox()
        for spin in (self.sig3_sweep_start_ns_spin, self.sig3_sweep_stop_ns_spin):
            spin.setRange(0, 50000)
            spin.setSuffix(" ns")
        self.sig3_sweep_stop_ns_spin.setValue(50000)
        self.sig3_sweep_step_ns_spin = QtWidgets.QSpinBox()
        self.sig3_sweep_step_ns_spin.setRange(1, 50000)
        self.sig3_sweep_step_ns_spin.setValue(100)
        self.sig3_sweep_step_ns_spin.setSuffix(" ns")
        self.sig3_sweep_dwell_spin = QtWidgets.QDoubleSpinBox()
        self.sig3_sweep_dwell_spin.setRange(0.10, 30.00)
        self.sig3_sweep_dwell_spin.setDecimals(2)
        self.sig3_sweep_dwell_spin.setValue(1.00)
        self.sig3_sweep_dwell_spin.setSuffix(" s")
        self.sig3_sweep_auto_enable_check = QtWidgets.QCheckBox("扫描时使能 SIG3")
        self.sig3_sweep_auto_enable_check.setChecked(True)
        self.sig3_sweep_restore_check = QtWidgets.QCheckBox("结束后恢复原延时")
        self.sig3_sweep_restore_check.setChecked(True)
        self.sig3_sweep_start_btn = QtWidgets.QPushButton("开始 SIG3 扫门")
        self.sig3_sweep_stop_btn = QtWidgets.QPushButton("停止")
        self.sig3_sweep_stop_btn.setEnabled(False)
        self.sig3_sweep_copy_btn = QtWidgets.QPushButton("复制 CSV")
        self.sig3_sweep_copy_btn.setEnabled(False)
        self.sig3_sweep_apply_best_btn = QtWidgets.QPushButton("下发最大计数延时")
        self.sig3_sweep_apply_best_btn.setEnabled(False)
        self.sig3_sweep_status_label = QtWidgets.QLabel("SIG3 扫门: -")
        self.sig3_sweep_status_label.setObjectName("InfoText")

        self.sig3_sweep_table = QtWidgets.QTableWidget(0, 6)
        self.sig3_sweep_table.setHorizontalHeaderLabels(
            ["延时(ns)", "counter_1s", "SIG3增量", "触发增量", "输出增量", "seq"]
        )
        self.sig3_sweep_table.verticalHeader().setVisible(False)
        self.sig3_sweep_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sig3_sweep_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sig3_sweep_table.setAlternatingRowColors(True)
        self.sig3_sweep_table.setMinimumHeight(190)
        self.sig3_sweep_table.horizontalHeader().setStretchLastSection(True)

        self.sig3_sweep_plot = pg.PlotWidget(title="SIG3 扫门")
        self.sig3_sweep_plot.setMinimumHeight(230)
        self._style_plot_widget(self.sig3_sweep_plot, "SIG3 delay (ns)")
        self.sig3_sweep_plot.getPlotItem().setLabel("left", "counts")
        self.sig3_sweep_plot.addLegend(offset=(10, 10))
        self.sig3_sweep_cps_curve = self.sig3_sweep_plot.plot(
            pen=pg.mkPen("#38bdf8", width=2),
            symbol="o",
            symbolBrush="#38bdf8",
            symbolPen=pg.mkPen("#38bdf8"),
            name="counter_1s",
        )
        self.sig3_sweep_gate_curve = self.sig3_sweep_plot.plot(
            pen=pg.mkPen("#f59e0b", width=2),
            symbol="t",
            symbolBrush="#f59e0b",
            symbolPen=pg.mkPen("#f59e0b"),
            name="SIG3 delta",
        )

        fpga_grid.addWidget(self.flash_save_btn, 0, 0)
        fpga_grid.addWidget(self.flash_load_btn, 0, 1)

        fpga_grid.addWidget(QtWidgets.QLabel("Gate Holdoff"), 1, 0)
        fpga_grid.addWidget(self.gate_holdoff_spin, 1, 1)
        fpga_grid.addWidget(self.gate_holdoff_btn, 1, 2)

        fpga_grid.addWidget(QtWidgets.QLabel("Gate Divider"), 2, 0)
        fpga_grid.addWidget(self.gate_div_spin, 2, 1)
        fpga_grid.addWidget(self.gate_div_btn, 2, 2)

        fpga_grid.addWidget(QtWidgets.QLabel("NB6 A/B"), 3, 0)
        fpga_grid.addWidget(self.nb6_a_spin, 3, 1)
        fpga_grid.addWidget(self.nb6_b_spin, 3, 2)
        fpga_grid.addWidget(self.nb6_enable_check, 3, 3)
        fpga_grid.addWidget(self.nb6_btn, 3, 4)

        fpga_grid.addWidget(self.sig2_enable_check, 4, 0)
        fpga_grid.addWidget(self.sig3_enable_check, 4, 1)
        fpga_grid.addWidget(self.pixel_mode_check, 4, 2)
        fpga_grid.addWidget(self.gate_enable_btn, 4, 3)

        fpga_grid.addWidget(QtWidgets.QLabel("门信号"), 5, 0)
        fpga_grid.addWidget(QtWidgets.QLabel("延时粗调(10ns)"), 5, 1)
        fpga_grid.addWidget(QtWidgets.QLabel("延时细调(1ns)"), 5, 2)
        fpga_grid.addWidget(QtWidgets.QLabel("宽度粗调(10ns)"), 5, 3)
        fpga_grid.addWidget(QtWidgets.QLabel("宽度细调(1ns)"), 5, 4)

        fpga_grid.addWidget(QtWidgets.QLabel("SIG2 直通门"), 6, 0)
        fpga_grid.addWidget(self.sig2_delay_coarse, 6, 1)
        fpga_grid.addWidget(self.sig2_delay_fine, 6, 2)
        fpga_grid.addWidget(self.sig2_width_coarse, 6, 3)
        fpga_grid.addWidget(self.sig2_width_fine, 6, 4)
        fpga_grid.addWidget(self.sig2_apply_btn, 6, 5)

        fpga_grid.addWidget(QtWidgets.QLabel("SIG3 分频门"), 7, 0)
        fpga_grid.addWidget(self.sig3_delay_coarse, 7, 1)
        fpga_grid.addWidget(self.sig3_delay_fine, 7, 2)
        fpga_grid.addWidget(self.sig3_width_coarse, 7, 3)
        fpga_grid.addWidget(self.sig3_width_fine, 7, 4)
        fpga_grid.addWidget(self.sig3_apply_btn, 7, 5)
        fpga_grid.addWidget(self.sig3_sweep_open_btn, 8, 0, 1, 2)
        fpga_grid.addWidget(self.control_result_label, 9, 0, 1, 6)
        layout.addWidget(fpga_group)
        layout.addStretch(1)
        return tab

    def _build_pixel_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("Card")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 14, 14, 14)
        toolbar_layout.setSpacing(10)

        self.pixel_import_json_btn = QtWidgets.QPushButton("导入 JSON")
        self.pixel_export_json_btn = QtWidgets.QPushButton("导出 JSON")
        self.pixel_import_csv_btn = QtWidgets.QPushButton("导入 CSV")
        self.pixel_export_csv_btn = QtWidgets.QPushButton("导出 CSV")
        self.pixel_add_row_btn = QtWidgets.QPushButton("新增一行")
        self.pixel_write_selected_btn = QtWidgets.QPushButton("写选中行")
        self.pixel_write_all_btn = QtWidgets.QPushButton("批量写全部")
        self.pixel_reload_btn = QtWidgets.QPushButton("重载像素参数")

        for widget in (
            self.pixel_import_json_btn,
            self.pixel_export_json_btn,
            self.pixel_import_csv_btn,
            self.pixel_export_csv_btn,
            self.pixel_add_row_btn,
            self.pixel_write_selected_btn,
            self.pixel_write_all_btn,
            self.pixel_reload_btn,
        ):
            if widget in (self.pixel_export_json_btn, self.pixel_export_csv_btn):
                widget.setObjectName("Secondary")
            toolbar_layout.addWidget(widget)
        toolbar_layout.addStretch(1)
        layout.addWidget(toolbar)

        table_card = QtWidgets.QFrame()
        table_card.setObjectName("Card")
        table_layout = QtWidgets.QVBoxLayout(table_card)
        table_layout.setContentsMargins(14, 14, 14, 14)
        self.pixel_table = QtWidgets.QTableWidget(0, 4)
        self.pixel_table.setHorizontalHeaderLabels(["addr", "value36", "version", "comment"])
        self.pixel_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.pixel_table.verticalHeader().setVisible(False)
        self.pixel_table.verticalHeader().setDefaultSectionSize(38)
        self.pixel_table.setAlternatingRowColors(True)
        self.pixel_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pixel_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table_layout.addWidget(self.pixel_table)
        layout.addWidget(table_card, 1)
        return tab

    def _build_tdc_test_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QtWidgets.QGridLayout(controls_card)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(12)

        self.tdc_test_enable_check = QtWidgets.QCheckBox("Running")
        self.tdc_test_enable_check.setEnabled(False)
        self.tdc_test_enable_check.setToolTip("Start controls upload; Stop turns it off.")
        self.tdc_source_combo = QtWidgets.QComboBox()
        self.tdc_source_combo.addItem("GPX2 FPGA histogram", "gpx2_fpga")
        self.tdc_source_combo.addItem("GPX2 raw", "gpx2")
        self.tdc_source_combo.addItem("GPX2 example SPI", "gpx2_spi")
        self.tdc_source_combo.addItem("FPGA synthetic", "synthetic")
        self.tdc_pairing_combo = QtWidgets.QComboBox()
        self.tdc_pairing_combo.addItem("Commercial nearest-start", "prev_start")
        self.tdc_pairing_combo.addItem("Correlation all-start", "all_start")
        self.tdc_start_combo = QtWidgets.QComboBox()
        self.tdc_stop_combo = QtWidgets.QComboBox()
        for combo in (self.tdc_start_combo, self.tdc_stop_combo):
            for ch in range(1, 5):
                combo.addItem(f"CH{ch}", ch)

        self.tdc_refdiv_spin = QtWidgets.QSpinBox()
        self.tdc_refdiv_spin.setRange(1, 1_000_000)
        self.tdc_refdiv_spin.setValue(12500)
        self.tdc_bin_width_spin = QtWidgets.QSpinBox()
        self.tdc_bin_width_spin.setRange(1, 1_000_000)
        self.tdc_bin_width_spin.setValue(1)
        self.tdc_bin_offset_spin = QtWidgets.QSpinBox()
        self.tdc_bin_offset_spin.setRange(0, 1_000_000_000)
        self.tdc_bin_count_spin = QtWidgets.QSpinBox()
        self.tdc_bin_count_spin.setRange(1, 1_000_000)
        self.tdc_bin_count_spin.setValue(4096)
        self.tdc_acq_time_spin = QtWidgets.QDoubleSpinBox()
        self.tdc_acq_time_spin.setRange(0.0, 86400.0)
        self.tdc_acq_time_spin.setDecimals(3)
        self.tdc_acq_time_spin.setSingleStep(0.1)
        self.tdc_acq_time_spin.setSuffix(" s")
        self.tdc_acq_time_spin.setValue(1.0)
        self.tdc_display_min_bin_spin = QtWidgets.QSpinBox()
        self.tdc_display_min_bin_spin.setRange(0, 1_000_000)
        self.tdc_display_min_bin_spin.setValue(0)
        self.tdc_display_min_bin_spin.setToolTip("Lowest histogram bin used for plotting and peak/afterpulse analysis.")
        self.tdc_prompt_threshold_spin = QtWidgets.QSpinBox()
        self.tdc_prompt_threshold_spin.setRange(0, 1_000_000_000)
        self.tdc_prompt_threshold_spin.setValue(100)
        self.tdc_prompt_threshold_spin.setToolTip("Prompt peak integration threshold. Set 0 to use half of the peak.")
        self.tdc_afterpulse_delay_spin = QtWidgets.QDoubleSpinBox()
        self.tdc_afterpulse_delay_spin.setRange(0.0, 1_000_000.0)
        self.tdc_afterpulse_delay_spin.setDecimals(3)
        self.tdc_afterpulse_delay_spin.setSingleStep(10.0)
        self.tdc_afterpulse_delay_spin.setSuffix(" ns")
        self.tdc_afterpulse_delay_spin.setValue(100.0)
        self.tdc_afterpulse_delay_spin.setToolTip("Afterpulse integration starts at peak time plus this delay.")
        self.tdc_deadtime_ps_spin = QtWidgets.QDoubleSpinBox()
        self.tdc_deadtime_ps_spin.setRange(0.0, 1_000_000_000.0)
        self.tdc_deadtime_ps_spin.setDecimals(1)
        self.tdc_deadtime_ps_spin.setSingleStep(100.0)
        self.tdc_deadtime_ps_spin.setSuffix(" ps")
        self.tdc_deadtime_ps_spin.setValue(0.0)
        self.tdc_deadtime_ps_spin.setToolTip("Optional same-channel TDC deadtime; 0 disables filtering.")
        self.tdc_reference_deadtime_ns_spin = QtWidgets.QDoubleSpinBox()
        self.tdc_reference_deadtime_ns_spin.setRange(0.0, 10_000.0)
        self.tdc_reference_deadtime_ns_spin.setDecimals(1)
        self.tdc_reference_deadtime_ns_spin.setSingleStep(10.0)
        self.tdc_reference_deadtime_ns_spin.setSuffix(" ns")
        self.tdc_reference_deadtime_ns_spin.setValue(800.0)
        self.tdc_reference_deadtime_ns_spin.setToolTip(
            "Reference-channel cleanup holdoff. The 800 ns default keeps one start per 625 kHz laser period; raw repeat diagnostics remain visible."
        )

        self.tdc_test_apply_btn = QtWidgets.QPushButton("Start")
        self.tdc_test_stop_btn = QtWidgets.QPushButton("Stop")
        self.tdc_test_stop_btn.setObjectName("Secondary")
        self.tdc_test_clear_btn = QtWidgets.QPushButton("Clear")
        self.tdc_test_clear_btn.setObjectName("Secondary")
        self.tdc_test_status_label = QtWidgets.QLabel("sync: 0 / photon: 0 / pairs: 0 / peak: -")
        self.tdc_test_status_label.setObjectName("InfoText")

        controls_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        controls_layout.addWidget(self.tdc_test_enable_check, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Source"), 0, 2)
        controls_layout.addWidget(self.tdc_source_combo, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Reference"), 0, 4)
        controls_layout.addWidget(self.tdc_start_combo, 0, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Photon"), 1, 4)
        controls_layout.addWidget(self.tdc_stop_combo, 1, 5)
        controls_layout.addWidget(QtWidgets.QLabel("REF divisions"), 1, 0)
        controls_layout.addWidget(self.tdc_refdiv_spin, 1, 1)
        bin_width_label = QtWidgets.QLabel("Bin width (ticks)")
        bin_width_label.setToolTip("GPX2 raw tick is 8 ps; 5 ticks = 40 ps.")
        controls_layout.addWidget(bin_width_label, 1, 2)
        controls_layout.addWidget(self.tdc_bin_width_spin, 1, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Bins"), 2, 0)
        controls_layout.addWidget(self.tdc_bin_count_spin, 2, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Offset"), 2, 2)
        controls_layout.addWidget(self.tdc_bin_offset_spin, 2, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Acq time"), 2, 4)
        controls_layout.addWidget(self.tdc_acq_time_spin, 2, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Min bin"), 3, 0)
        controls_layout.addWidget(self.tdc_display_min_bin_spin, 3, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Prompt min"), 3, 2)
        controls_layout.addWidget(self.tdc_prompt_threshold_spin, 3, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Afterpulse"), 3, 4)
        controls_layout.addWidget(self.tdc_afterpulse_delay_spin, 3, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Pairing"), 4, 0)
        controls_layout.addWidget(self.tdc_pairing_combo, 4, 1)
        controls_layout.addWidget(QtWidgets.QLabel("TDC deadtime"), 4, 2)
        controls_layout.addWidget(self.tdc_deadtime_ps_spin, 4, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Ref cleanup"), 4, 4)
        controls_layout.addWidget(self.tdc_reference_deadtime_ns_spin, 4, 5)
        controls_layout.addWidget(self.tdc_test_apply_btn, 5, 1)
        controls_layout.addWidget(self.tdc_test_stop_btn, 5, 3)
        controls_layout.addWidget(self.tdc_test_clear_btn, 5, 4)
        controls_layout.addWidget(self.tdc_test_status_label, 5, 5)
        layout.addWidget(controls_card)

        self.tdc_test_plot = pg.PlotWidget(title="TCSPC Multi-hit Histogram")
        self._style_plot_widget(self.tdc_test_plot, "dt from reference (ns)")
        self.tdc_test_plot.getPlotItem().setLabel("left", "log10(counts/bin)")
        self.tdc_test_curve = self.tdc_test_plot.plot(pen=pg.mkPen("#22c55e", width=1))
        layout.addWidget(self._wrap_card("TCSPC Log Overview", self.tdc_test_plot), 2)
        self.tdc_peak_plot = pg.PlotWidget(title="TCSPC Peak Shape")
        self._style_plot_widget(self.tdc_peak_plot, "dt near peak (ns)")
        self.tdc_peak_plot.getPlotItem().setLabel("left", "counts/bin")
        self.tdc_peak_curve = self.tdc_peak_plot.plot(pen=pg.mkPen("#38bdf8", width=1))
        self.tdc_peak_half_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#f59e0b", width=1, style=QtCore.Qt.DashLine),
        )
        self.tdc_peak_left_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#f97316", width=1, style=QtCore.Qt.DashLine),
        )
        self.tdc_peak_right_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#f97316", width=1, style=QtCore.Qt.DashLine),
        )
        for marker in (self.tdc_peak_half_line, self.tdc_peak_left_line, self.tdc_peak_right_line):
            marker.setVisible(False)
            self.tdc_peak_plot.addItem(marker)
        layout.addWidget(self._wrap_card("TCSPC Peak Shape", self.tdc_peak_plot), 1)
        return tab

    def _build_acquisition_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(14)
        temp_card, self.temp_value_big_label, self.temp_detail_label = self._create_metric_card(
            "实时温度", "-- °C", "等待 STATUS"
        )
        cps_card, self.cps_value_big_label, self.cps_detail_label = self._create_metric_card(
            "实时计数", "-- cps", "计数率未更新"
        )
        rec_card, self.record_value_big_label, self.record_detail_label = self._create_metric_card(
            "采集状态", "Idle", "未开始录制"
        )
        metrics_row.addWidget(temp_card)
        metrics_row.addWidget(cps_card)
        metrics_row.addWidget(rec_card)
        layout.addLayout(metrics_row)

        trend_row = QtWidgets.QHBoxLayout()
        trend_row.setSpacing(14)
        self.temp_trend_plot = self._create_trend_plot("温度趋势")
        self.cps_trend_plot = self._create_trend_plot("CPS 趋势")
        trend_row.addWidget(self._wrap_card("温度小图", self.temp_trend_plot), 1)
        trend_row.addWidget(self._wrap_card("计数小图", self.cps_trend_plot), 1)
        layout.addLayout(trend_row)

        self.temp_trend_curve = self.temp_trend_plot.plot(pen=pg.mkPen("#22c55e", width=2))
        self.cps_trend_curve = self.cps_trend_plot.plot(pen=pg.mkPen("#38bdf8", width=2))

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QtWidgets.QHBoxLayout(controls_card)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(10)

        self.session_name_edit = QtWidgets.QLineEdit("capture")
        self.session_notes_edit = QtWidgets.QLineEdit()
        self.start_recording_btn = QtWidgets.QPushButton("开始采集并落盘")
        self.stop_recording_btn = QtWidgets.QPushButton("停止采集")
        self.stop_recording_btn.setObjectName("Secondary")
        self.acq_status_label = QtWidgets.QLabel("未采集")
        self.counter_label = QtWidgets.QLabel("counter_1s: 0")
        self.drop_label = QtWidgets.QLabel("tdc_drop: 0 / usb_drop: 0")

        controls_layout.addWidget(QtWidgets.QLabel("Session"))
        controls_layout.addWidget(self.session_name_edit)
        controls_layout.addWidget(QtWidgets.QLabel("备注"))
        controls_layout.addWidget(self.session_notes_edit, 1)
        controls_layout.addWidget(self.start_recording_btn)
        controls_layout.addWidget(self.stop_recording_btn)
        controls_layout.addWidget(self.acq_status_label)
        controls_layout.addWidget(self.counter_label)
        controls_layout.addWidget(self.drop_label)
        layout.addWidget(controls_card)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.hist_plot = pg.PlotWidget(title="当前像素 Histogram")
        self._style_plot_widget(self.hist_plot, "Bin")
        self.hist_plot.getPlotItem().setLabel("bottom", "Bin")
        self.hist_plot.getPlotItem().setLabel("left", "Counts")
        self.hist_curve = self.hist_plot.plot(pen=pg.mkPen("#f59e0b", width=2))
        self.image_view = pg.ImageView()
        splitter.addWidget(self._wrap_card("当前像素 Histogram", self.hist_plot))
        splitter.addWidget(self._wrap_card("像素投影", self.image_view))
        splitter.setSizes([760, 760])
        layout.addWidget(splitter, 1)
        return tab

    def _build_offline_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(14)

        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("Card")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 14, 14, 14)
        toolbar_layout.setSpacing(10)

        self.open_session_btn = QtWidgets.QPushButton("打开 Session")
        self.export_hist_csv_btn = QtWidgets.QPushButton("导出当前 Histogram CSV")
        self.export_hist_csv_btn.setObjectName("Secondary")
        self.offline_status_label = QtWidgets.QLabel("未加载")
        toolbar_layout.addWidget(self.open_session_btn)
        toolbar_layout.addWidget(self.export_hist_csv_btn)
        toolbar_layout.addWidget(self.offline_status_label)
        toolbar_layout.addStretch(1)
        layout.addWidget(toolbar)

        self.offline_hist_plot = pg.PlotWidget(title="离线 Histogram")
        self._style_plot_widget(self.offline_hist_plot, "Bin")
        self.offline_hist_plot.getPlotItem().setLabel("left", "Counts")
        self.offline_hist_curve = self.offline_hist_plot.plot(pen=pg.mkPen("#fb7185", width=2))
        self.offline_image_view = pg.ImageView()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._wrap_card("离线 Histogram", self.offline_hist_plot))
        splitter.addWidget(self._wrap_card("离线像素投影", self.offline_image_view))
        splitter.setSizes([760, 760])
        layout.addWidget(splitter, 1)
        return tab

    def _build_log_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("LogPanel")
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(2000)
        layout.addWidget(self._wrap_card("运行日志", self.log_view))
        return tab

    def _make_stat_pill(self, title: str, value: str) -> QtWidgets.QLabel:
        frame = QtWidgets.QFrame()
        frame.setObjectName("MetricCard")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(4)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("MetricTitle")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("MetricValue")
        value_label.setStyleSheet("font-size:22px;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        value_label._pill_frame = frame  # type: ignore[attr-defined]
        return value_label

    def _create_metric_card(self, title: str, value: str, detail: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel]:
        frame = QtWidgets.QFrame()
        frame.setObjectName("MetricCard")
        frame.setMinimumHeight(118)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("MetricTitle")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("MetricValue")
        detail_label = QtWidgets.QLabel(detail)
        detail_label.setObjectName("MetricDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        layout.addStretch(1)
        return frame, value_label, detail_label

    def _create_trend_plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setMinimumHeight(170)
        self._style_plot_widget(plot, "Samples")
        return plot

    def _wrap_card(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_label = self._make_card_title(title)
        layout.addWidget(title_label)
        layout.addWidget(widget, 1)
        return frame

    def _build_sig3_sweep_window(self) -> QtWidgets.QDialog:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("SIG3 扫门")
        dialog.setModal(False)
        dialog.resize(1180, 720)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        controls = QtWidgets.QGroupBox("扫描设置")
        controls_grid = QtWidgets.QGridLayout(controls)
        controls_grid.setHorizontalSpacing(12)
        controls_grid.setVerticalSpacing(10)
        controls_grid.addWidget(QtWidgets.QLabel("起始延时"), 0, 0)
        controls_grid.addWidget(self.sig3_sweep_start_ns_spin, 0, 1)
        controls_grid.addWidget(QtWidgets.QLabel("结束延时"), 0, 2)
        controls_grid.addWidget(self.sig3_sweep_stop_ns_spin, 0, 3)
        controls_grid.addWidget(QtWidgets.QLabel("步进"), 0, 4)
        controls_grid.addWidget(self.sig3_sweep_step_ns_spin, 0, 5)
        controls_grid.addWidget(QtWidgets.QLabel("每点采样"), 1, 0)
        controls_grid.addWidget(self.sig3_sweep_dwell_spin, 1, 1)
        controls_grid.addWidget(self.sig3_sweep_auto_enable_check, 1, 2)
        controls_grid.addWidget(self.sig3_sweep_restore_check, 1, 3)
        controls_grid.addWidget(self.sig3_sweep_start_btn, 1, 4)
        controls_grid.addWidget(self.sig3_sweep_stop_btn, 1, 5)
        controls_grid.addWidget(self.sig3_sweep_copy_btn, 2, 4)
        controls_grid.addWidget(self.sig3_sweep_apply_best_btn, 2, 5)
        controls_grid.addWidget(self.sig3_sweep_status_label, 3, 0, 1, 7)
        layout.addWidget(controls)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.sig3_sweep_plot, 3)
        body.addWidget(self.sig3_sweep_table, 2)
        layout.addLayout(body, 1)
        return dialog

    def on_open_sig3_sweep_window(self) -> None:
        if self.sig3_sweep_window is None:
            self.sig3_sweep_window = self._build_sig3_sweep_window()
        self.sig3_sweep_window.show()
        self.sig3_sweep_window.raise_()
        self.sig3_sweep_window.activateWindow()

    def _bind_signals(self) -> None:
        c = self.controller

        # Connection tab
        self.refresh_devices_btn.clicked.connect(self.on_refresh_devices)
        self.connect_btn.clicked.connect(self.on_connect_device)
        self.disconnect_btn.clicked.connect(c.disconnect_device)
        self.read_enable_check.toggled.connect(self.on_read_enable_toggled)
        self.self_test_btn.clicked.connect(self.on_run_self_test)
        self.save_config_btn.clicked.connect(self.on_save_config)

        # Control tab
        self.temp_set_btn.clicked.connect(self.on_set_temperature)
        self.apply_analog_btn.clicked.connect(self.on_apply_analog)
        self.save_analog_flash_btn.clicked.connect(self.on_save_analog_defaults_to_flash)
        self.gpx2_init_btn.clicked.connect(self.on_gpx2_init_opcode)
        self.gpx2_custom_apply_btn.clicked.connect(self.on_apply_gpx2_custom)
        self.gpx2_lvds_test_btn.clicked.connect(self.on_fill_gpx2_lvds_test)
        self.gpx2_reset_btn.clicked.connect(self.on_gpx2_power_opcode)
        self.gpx2_read_cfg_btn.clicked.connect(self.on_read_gpx2_config)
        self.gpx2_apply_hex_btn.clicked.connect(self.on_fill_gpx2_from_hex)
        self.gpx2_load_default_btn.clicked.connect(self.on_load_gpx2_defaults)
        self.gpx2_copy_readback_btn.clicked.connect(self.on_copy_gpx2_readback)
        self.gpx2_run_profile_btn.clicked.connect(self.on_run_gpx2_profile_sequence)
        self.gpx2_run_custom_sequence_btn.clicked.connect(self.on_run_gpx2_custom_sequence)
        self.flash_save_btn.clicked.connect(lambda: self._show_command_result(c.flash_save(3000)))
        self.flash_load_btn.clicked.connect(lambda: self._show_command_result(c.flash_load(3000)))
        self.gate_holdoff_btn.clicked.connect(
            lambda: self._show_command_result(c.set_gate_holdoff(self.gate_holdoff_spin.value()))
        )
        self.gate_div_btn.clicked.connect(lambda: self._show_command_result(c.set_gate_div(self.gate_div_spin.value())))
        self.nb6_btn.clicked.connect(
            lambda: self._show_command_result(
                c.set_nb6(
                    self.nb6_a_spin.value(),
                    self.nb6_b_spin.value(),
                    self.nb6_enable_check.isChecked(),
                )
            )
        )
        self.gate_enable_btn.clicked.connect(
            lambda: self._show_command_result(
                c.set_gate_enable(
                    self.sig2_enable_check.isChecked(),
                    self.sig3_enable_check.isChecked(),
                    self.pixel_mode_check.isChecked(),
                )
            )
        )
        self.sig2_apply_btn.clicked.connect(
            lambda: self._show_command_result(
                c.set_gate_signal(
                    2,
                    self.sig2_delay_coarse.value(),
                    self.sig2_delay_fine.value(),
                    self.sig2_width_coarse.value(),
                    self.sig2_width_fine.value(),
                )
            )
        )
        self.sig3_apply_btn.clicked.connect(
            lambda: self._show_command_result(
                c.set_gate_signal(
                    3,
                    self.sig3_delay_coarse.value(),
                    self.sig3_delay_fine.value(),
                    self.sig3_width_coarse.value(),
                    self.sig3_width_fine.value(),
                )
            )
        )
        self.sig3_sweep_open_btn.clicked.connect(self.on_open_sig3_sweep_window)
        self.sig3_sweep_start_btn.clicked.connect(self.on_start_sig3_sweep)
        self.sig3_sweep_stop_btn.clicked.connect(self.on_stop_sig3_sweep)
        self.sig3_sweep_copy_btn.clicked.connect(self.on_copy_sig3_sweep_csv)
        self.sig3_sweep_apply_best_btn.clicked.connect(self.on_apply_best_sig3_sweep_delay)

        # Pixel tab
        self.pixel_add_row_btn.clicked.connect(self.on_add_pixel_row)
        self.pixel_import_json_btn.clicked.connect(lambda: self._import_pixel_records("json"))
        self.pixel_export_json_btn.clicked.connect(lambda: self._export_pixel_records("json"))
        self.pixel_import_csv_btn.clicked.connect(lambda: self._import_pixel_records("csv"))
        self.pixel_export_csv_btn.clicked.connect(lambda: self._export_pixel_records("csv"))
        self.pixel_write_selected_btn.clicked.connect(self.on_write_selected_pixel_row)
        self.pixel_write_all_btn.clicked.connect(self.on_write_all_pixels)
        self.pixel_reload_btn.clicked.connect(lambda: self._show_command_result(c.reload_pixel_params(2000)))

        # Acquisition / offline
        self.start_recording_btn.clicked.connect(self.on_start_recording)
        self.stop_recording_btn.clicked.connect(c.stop_recording)
        self.tdc_test_apply_btn.clicked.connect(self.on_apply_tdc_test)
        self.tdc_test_stop_btn.clicked.connect(self.on_stop_tdc_test)
        self.tdc_test_clear_btn.clicked.connect(self.on_clear_tdc_test)
        self.open_session_btn.clicked.connect(self.on_open_session)
        self.export_hist_csv_btn.clicked.connect(self.on_export_hist_csv)

        # Runtime updates
        c.device_state_changed.connect(self.on_device_state_changed)
        c.log_message.connect(self.append_log)
        c.stats_updated.connect(self.on_stats_updated)
        c.read_state_changed.connect(self.on_read_state_changed)
        c.status_received.connect(self.on_status_packet)
        c.temperature_target_changed.connect(self.on_temperature_target_changed)
        c.analog_targets_changed.connect(self.on_analog_targets_changed)
        c.histogram_updated.connect(self.on_histogram_snapshot)
        c.tdc_test_histogram_updated.connect(self.on_tdc_test_snapshot)
        c.recording_state_changed.connect(self.on_recording_state_changed)
        c.replay_completed.connect(self.on_offline_snapshot)

    def _load_defaults_from_config(self) -> None:
        cfg = self.controller.config
        self.temp_target_spin.setValue(cfg.target_temperature_c)
        self.laser_thr_spin.setValue(cfg.analog_targets.laser_sync_threshold_mv)
        self.pixel_thr_spin.setValue(cfg.analog_targets.pixel_sync_threshold_mv)
        self.avalanche_thr_spin.setValue(cfg.analog_targets.avalanche_threshold_mv)
        self.bias_spin.setValue(cfg.analog_targets.bias_voltage_v)
        self.gate_holdoff_spin.setValue(cfg.gate_settings.hold_off_time)
        self.gate_div_spin.setValue(cfg.gate_settings.divider)
        self.sig2_enable_check.setChecked(cfg.gate_settings.sig2_enable)
        self.sig3_enable_check.setChecked(cfg.gate_settings.sig3_enable)
        self.pixel_mode_check.setChecked(cfg.gate_settings.pixel_mode)
        self.sig2_delay_coarse.setValue(cfg.gate_settings.sig2_delay_coarse)
        self.sig2_delay_fine.setValue(cfg.gate_settings.sig2_delay_fine)
        self.sig2_width_coarse.setValue(cfg.gate_settings.sig2_width_coarse)
        self.sig2_width_fine.setValue(cfg.gate_settings.sig2_width_fine)
        self.sig3_delay_coarse.setValue(cfg.gate_settings.sig3_delay_coarse)
        self.sig3_delay_fine.setValue(cfg.gate_settings.sig3_delay_fine)
        self.sig3_width_coarse.setValue(cfg.gate_settings.sig3_width_coarse)
        self.sig3_width_fine.setValue(cfg.gate_settings.sig3_width_fine)

        self.read_pipe_edit.setText(f"0x{cfg.read_pipe:02X}")
        self.write_pipe_edit.setText(f"0x{cfg.write_pipe:02X}")
        self.read_block_spin.setValue(cfg.read_block_size)
        self.read_enable_check.blockSignals(True)
        self.read_enable_check.setChecked(cfg.read_enabled)
        self.read_enable_check.blockSignals(False)
        tdc = cfg.tdc_test_settings
        self.tdc_test_enable_check.setChecked(tdc.enabled)
        source_index = self.tdc_source_combo.findData(getattr(tdc, "source", "gpx2"))
        self.tdc_source_combo.setCurrentIndex(max(0, source_index))
        pairing_index = self.tdc_pairing_combo.findData(getattr(tdc, "pairing_mode", "prev_start"))
        self.tdc_pairing_combo.setCurrentIndex(max(0, pairing_index))
        self.tdc_start_combo.setCurrentIndex(max(0, min(3, tdc.start_channel)))
        self.tdc_stop_combo.setCurrentIndex(max(0, min(3, tdc.stop_channel)))
        self.tdc_refdiv_spin.setValue(tdc.refclk_divisions)
        self.tdc_bin_width_spin.setValue(tdc.bin_width_raw)
        self.tdc_bin_offset_spin.setValue(tdc.bin_offset)
        self.tdc_bin_count_spin.setValue(tdc.bin_count)
        self.tdc_acq_time_spin.setValue(float(getattr(tdc, "acquisition_time_s", 1.0)))
        self.tdc_deadtime_ps_spin.setValue(float(getattr(tdc, "tdc_deadtime_ps", 0.0)))
        self.tdc_reference_deadtime_ns_spin.setValue(
            float(getattr(tdc, "reference_deadtime_ns", 800.0))
        )

    def _guard_action(self, func) -> None:
        try:
            func()
        except Exception as exc:
            self.append_log(f"Error: {exc}")
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))

    def _parse_int(self, text: str) -> int:
        return int(text.strip(), 0)

    @staticmethod
    def _format_gpx2_config_bytes(values: List[int] | tuple[int, ...]) -> str:
        return " ".join(f"{int(value) & 0xFF:02X}" for value in list(values)[:17])

    @staticmethod
    def _parse_gpx2_hex_value(text: str) -> int:
        token = text.strip().replace("_", "")
        if not token:
            raise ValueError("GPX2 register value cannot be empty.")
        try:
            return int(token, 0) & 0xFF
        except ValueError:
            return int(token, 16) & 0xFF

    def _parse_gpx2_custom_config_bytes(self) -> tuple[int, ...]:
        raw_text = self.gpx2_custom_hex_edit.text().strip().replace(",", " ")
        tokens = [token for token in raw_text.split() if token]
        if len(tokens) == 1:
            compact = tokens[0].replace("_", "")
            if len(compact) == 34 and all(ch in "0123456789abcdefABCDEF" for ch in compact):
                tokens = [compact[idx : idx + 2] for idx in range(0, 34, 2)]
        if len(tokens) != 17:
            raise ValueError("GPX2 custom config expects exactly 17 bytes (reg0..reg16).")

        values: list[int] = []
        for token in tokens:
            values.append(self._parse_gpx2_hex_value(token))
        regs = protocol.sanitize_gpx2_config_bytes(values)
        self._set_gpx2_register_values(regs)
        return regs

    def _collect_gpx2_register_values(self) -> tuple[int, ...]:
        values = [self._parse_gpx2_hex_value(edit.text()) for edit in self._gpx2_register_edits]
        regs = protocol.sanitize_gpx2_config_bytes(values)
        self._set_gpx2_register_values(regs)
        return regs

    def _set_gpx2_register_values(self, regs: List[int] | tuple[int, ...]) -> None:
        values = protocol.sanitize_gpx2_config_bytes(regs)
        self.gpx2_custom_hex_edit.setText(self._format_gpx2_config_bytes(values))
        while len(self._gpx2_register_edits) < len(values):
            row = len(self._gpx2_register_edits)
            spec = protocol.GPX2_REGISTER_SPECS[row]
            addr_item = QtWidgets.QTableWidgetItem(f"0x{spec.address:02X}")
            name_item = QtWidgets.QTableWidgetItem(spec.name)
            readback_item = QtWidgets.QTableWidgetItem("--")
            status_item = QtWidgets.QTableWidgetItem("planned")
            detail_item = QtWidgets.QTableWidgetItem(spec.summary)
            for item in (addr_item, name_item, readback_item, status_item, detail_item):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                item.setToolTip(spec.detail)
            self.gpx2_register_table.setItem(row, 0, addr_item)
            self.gpx2_register_table.setItem(row, 1, name_item)
            self.gpx2_register_table.setItem(row, 3, readback_item)
            self.gpx2_register_table.setItem(row, 4, status_item)
            self.gpx2_register_table.setItem(row, 5, detail_item)
            edit = QtWidgets.QLineEdit()
            edit.setAlignment(QtCore.Qt.AlignCenter)
            edit.setMaxLength(4)
            edit.setToolTip(spec.detail)
            if not spec.editable:
                edit.setReadOnly(True)
            edit.editingFinished.connect(self.on_gpx2_registers_changed)
            self.gpx2_register_table.setCellWidget(row, 2, edit)
            self._gpx2_register_edits.append(edit)
            self._gpx2_readback_items.append(readback_item)
            self._gpx2_status_items.append(status_item)
        for row, value in enumerate(values):
            self._gpx2_register_edits[row].setText(f"{value:02X}")
        self._refresh_gpx2_compare_view()

    def _set_gpx2_status_item(self, row: int, text: str, color_hex: str) -> None:
        item = self._gpx2_status_items[row]
        item.setText(text)
        item.setBackground(QtGui.QColor(color_hex))

    def _refresh_gpx2_compare_view(self) -> None:
        planned = []
        for edit in self._gpx2_register_edits:
            try:
                planned.append(self._parse_gpx2_hex_value(edit.text()))
            except ValueError:
                planned.append(None)
        readback = list(self._last_gpx2_readback[:17])
        for row, spec in enumerate(protocol.GPX2_REGISTER_SPECS):
            planned_value = planned[row] if row < len(planned) else None
            if planned_value is None:
                self._set_gpx2_status_item(row, "invalid", "#7f1d1d")
                continue
            if row < len(readback):
                readback_value = readback[row] & 0xFF
                self._gpx2_readback_items[row].setText(f"{readback_value:02X}")
                if readback_value == (planned_value & 0xFF):
                    self._set_gpx2_status_item(row, "match", "#123223")
                else:
                    self._set_gpx2_status_item(row, "diff", "#4a2f0b")
            else:
                self._gpx2_readback_items[row].setText("--")
                self._set_gpx2_status_item(row, "fixed" if not spec.editable else "planned", "#172033")

    def _show_gpx2_command_result(self, result, detail: str | None = None) -> None:
        status_text = "SENT" if result.success else "FAILED"
        detail_text = detail or result.message or status_text
        self.gpx2_result_label.setText(f"GPX2 cmd=0x{result.cmd_id:02X} status={status_text} detail={detail_text}")
        self.append_log(self.gpx2_result_label.text())

    def _current_status_count(self) -> int:
        if self.controller is None:
            return -1
        device_service = getattr(self.controller, "_device_service", None)
        if device_service is None:
            return -1
        return int(getattr(device_service.runtime_stats, "status_count", -1))

    def _set_rx_reader_checked(self, enabled: bool) -> None:
        self.read_enable_check.blockSignals(True)
        self.read_enable_check.setChecked(bool(enabled))
        self.read_enable_check.blockSignals(False)
        if self.controller is not None:
            self.controller.set_read_enabled(bool(enabled))

    def _restore_gpx2_temp_rx(self) -> None:
        if not self._gpx2_readback_temp_rx:
            return
        self._gpx2_readback_temp_rx = False
        self._set_rx_reader_checked(False)
        self.append_log("GPX2 config readback temporary RX reader disabled.")

    def _update_gpx2_readback_status(self, status: StatusPacket) -> None:
        if not status.gpx2_cfg_readback:
            return
        signature = (int(status.gpx2_cfg_diag), tuple(status.gpx2_cfg_readback[:17]))
        if (not self._gpx2_readback_pending) and signature == self._last_gpx2_render_signature:
            return
        self._last_gpx2_render_signature = signature
        self._last_gpx2_readback = signature[1]
        profile = (status.gpx2_cfg_diag >> 19) & 0xF
        sequence = (status.gpx2_cfg_diag >> 23) & 0xF
        mismatch_count = (status.gpx2_cfg_diag >> 13) & 0x1F
        mismatch_flag = bool((status.gpx2_cfg_diag >> 18) & 0x1)
        self.gpx2_diag_label.setText(
            f"最近回读 profile={profile} sequence={sequence} mismatch={int(mismatch_flag)} count={mismatch_count}"
        )
        decoded = protocol.describe_gpx2_config_bytes(self._last_gpx2_readback)
        lines = [self._format_gpx2_config_bytes(self._last_gpx2_readback), ""]
        lines.extend(f"{name}: {value}" for name, value in decoded.items())
        self.gpx2_readback_hex_view.setPlainText("\n".join(lines))
        self._refresh_gpx2_compare_view()
        status_count = self._current_status_count()
        is_fresh_status = (
            self._gpx2_readback_status_count_at_request < 0
            or status_count > self._gpx2_readback_status_count_at_request
        )
        if self._gpx2_readback_pending and is_fresh_status:
            self._gpx2_readback_pending = False
            self._gpx2_readback_status_count_at_request = -1
            self._gpx2_readback_timer.stop()
            self._restore_gpx2_temp_rx()
            self.gpx2_result_label.setText(
                f"GPX2 readback updated | status_count={status_count} mismatch={int(mismatch_flag)} count={mismatch_count}"
            )
            self.append_log(self.gpx2_result_label.text())

    def _current_device_index(self) -> int:
        data = self.device_combo.currentData()
        return 0 if data is None else int(data)

    def _show_command_result(self, result) -> None:
        status_text = "已发送" if result.success else "发送失败"
        detail = "" if result.message in ("", "SENT", "OK") else f" | {result.message}"
        command_name = self._command_display_name(result.cmd_id)
        self.control_result_label.setText(f"最近命令: {command_name} | {status_text}{detail}")
        self.append_log(self.control_result_label.text())

    @staticmethod
    def _command_display_name(cmd_id: int) -> str:
        names = {
            protocol.CMD_FLASH_SAVE: "保存全部板级配置到 Flash",
            protocol.CMD_FLASH_LOAD: "从 Flash 读取全部板级配置",
            protocol.CMD_FLASH_SAVE_ANALOG: "保存温度/阈值/偏压到 Flash",
            protocol.CMD_GATE: "设置 Gate Holdoff",
            protocol.CMD_NB6L295: "设置 NB6 延时",
            protocol.CMD_GATE_DIV: "设置 Gate 分频",
            protocol.CMD_GATE_SIG2: "设置 SIG2 直通门",
            protocol.CMD_GATE_SIG3: "设置 SIG3 分频门",
            protocol.CMD_GATE_SIG3_LONG: "设置 SIG3 扫门延时",
            protocol.CMD_GATE_ENABLE: "设置 Gate 使能",
            protocol.CMD_GATE_PIXEL: "刷新像素门参数",
            protocol.CMD_TEC_PID: "设置目标温度",
            protocol.CMD_AD5686: "设置阈值/偏压 DAC",
        }
        return names.get(cmd_id, f"命令 0x{cmd_id:02X}")

    @staticmethod
    def _split_gate_delay_ns(delay_ns: int) -> tuple[int, int]:
        value = max(0, min(159, int(delay_ns)))
        return value // 10, value % 10

    @staticmethod
    def _gate_delay_ns(delay_coarse: int, delay_fine: int) -> int:
        return (int(delay_coarse) * 10) + min(9, int(delay_fine))

    @staticmethod
    def _u32_delta_int(current: int, baseline: int) -> int:
        return (int(current) - int(baseline)) & 0xFFFFFFFF

    def _build_sig3_sweep_points(self) -> list[int]:
        start = self.sig3_sweep_start_ns_spin.value()
        stop = self.sig3_sweep_stop_ns_spin.value()
        step = max(1, self.sig3_sweep_step_ns_spin.value())
        if start <= stop:
            return list(range(start, stop + 1, step))
        return list(range(start, stop - 1, -step))

    def _set_sig3_sweep_running(self, running: bool) -> None:
        self._sig3_sweep_active = bool(running)
        self.sig3_sweep_start_btn.setEnabled(not running)
        self.sig3_sweep_stop_btn.setEnabled(running)
        for widget in (
            self.sig3_sweep_start_ns_spin,
            self.sig3_sweep_stop_ns_spin,
            self.sig3_sweep_step_ns_spin,
            self.sig3_sweep_dwell_spin,
            self.sig3_sweep_auto_enable_check,
            self.sig3_sweep_restore_check,
        ):
            widget.setEnabled(not running)
        self.sig3_sweep_apply_best_btn.setEnabled((not running) and bool(self._sig3_sweep_rows))

    def _clear_sig3_sweep_results(self) -> None:
        self._sig3_sweep_rows = []
        self.sig3_sweep_table.setRowCount(0)
        self.sig3_sweep_cps_curve.setData([], [])
        self.sig3_sweep_gate_curve.setData([], [])
        self.sig3_sweep_copy_btn.setEnabled(False)
        self.sig3_sweep_apply_best_btn.setEnabled(False)

    def _set_sig3_delay_controls(self, delay_ns: int) -> tuple[int, int]:
        coarse, fine = self._split_gate_delay_ns(delay_ns)
        self.sig3_delay_coarse.setValue(coarse)
        self.sig3_delay_fine.setValue(fine)
        return coarse, fine

    def on_start_sig3_sweep(self) -> None:
        self._guard_action(self._start_sig3_sweep)

    def _start_sig3_sweep(self) -> None:
        if self.controller is None:
            raise RuntimeError("Controller is not bound.")
        if self._sig3_sweep_active:
            return

        self._sig3_sweep_points = self._build_sig3_sweep_points()
        if not self._sig3_sweep_points:
            raise ValueError("SIG3 sweep range is empty.")
        self._sig3_sweep_index = 0
        self._sig3_sweep_phase = "idle"
        self._sig3_sweep_restore_state = (
            self.sig3_delay_coarse.value(),
            self.sig3_delay_fine.value(),
            self.sig3_enable_check.isChecked(),
        )
        self._clear_sig3_sweep_results()
        self._set_sig3_sweep_running(True)

        if self.sig3_sweep_auto_enable_check.isChecked() and not self.sig3_enable_check.isChecked():
            result = self.controller.set_gate_enable(
                self.sig2_enable_check.isChecked(),
                True,
                self.pixel_mode_check.isChecked(),
            )
            self._show_command_result(result)
            if not result.success:
                self._finish_sig3_sweep(f"SIG3 使能失败: {result.message}", restore=False)
                return
            self.sig3_enable_check.setChecked(True)

        self.sig3_sweep_status_label.setText(
            f"SIG3 扫门: 0/{len(self._sig3_sweep_points)}"
        )
        self._start_next_sig3_sweep_point()

    def on_stop_sig3_sweep(self) -> None:
        self._finish_sig3_sweep("SIG3 扫门已停止")

    def _start_next_sig3_sweep_point(self) -> None:
        if not self._sig3_sweep_active:
            return
        if self._sig3_sweep_index >= len(self._sig3_sweep_points):
            self._finish_sig3_sweep(
                f"SIG3 扫门完成: {len(self._sig3_sweep_rows)} 点"
            )
            return

        delay_ns = self._sig3_sweep_points[self._sig3_sweep_index]
        status = self._last_status_packet
        self._sig3_sweep_config_seq = int(status.header.seq) if status is not None else None
        self._sig3_sweep_baseline = None
        self._sig3_sweep_phase = "baseline"
        now = time.monotonic()
        dwell_s = float(self.sig3_sweep_dwell_spin.value())
        self._sig3_sweep_timeout = now + max(3.0, dwell_s + 3.0)
        result = self.controller.set_gate_sig3_long_delay(
            delay_ns,
            self.sig3_width_coarse.value(),
            self.sig3_width_fine.value(),
        )
        self._show_command_result(result)
        if not result.success:
            self._finish_sig3_sweep(f"SIG3 延时 {delay_ns} ns 下发失败: {result.message}")
            return

        self.sig3_sweep_status_label.setText(
            f"SIG3 扫门: {self._sig3_sweep_index + 1}/{len(self._sig3_sweep_points)} "
            f"delay={delay_ns} ns 等待基准 STATUS"
        )
        self._sig3_sweep_timer.start()

    def _on_sig3_sweep_timer(self) -> None:
        if not self._sig3_sweep_active:
            self._sig3_sweep_timer.stop()
            return

        now = time.monotonic()
        status = self._last_status_packet
        if status is None:
            if now >= self._sig3_sweep_timeout:
                self._finish_sig3_sweep("SIG3 扫门超时: 未收到 STATUS")
            return

        seq = int(status.header.seq)
        if self._sig3_sweep_phase == "baseline":
            if self._sig3_sweep_config_seq is None or seq != self._sig3_sweep_config_seq:
                self._sig3_sweep_baseline = status
                self._sig3_sweep_phase = "sample"
                dwell_s = float(self.sig3_sweep_dwell_spin.value())
                self._sig3_sweep_deadline = now + dwell_s
                self._sig3_sweep_timeout = self._sig3_sweep_deadline + max(3.0, dwell_s + 1.0)
                delay_ns = self._sig3_sweep_points[self._sig3_sweep_index]
                self.sig3_sweep_status_label.setText(
                    f"SIG3 扫门: {self._sig3_sweep_index + 1}/{len(self._sig3_sweep_points)} "
                    f"delay={delay_ns} ns 采样中"
                )
            elif now >= self._sig3_sweep_timeout:
                self._finish_sig3_sweep("SIG3 扫门超时: 等待基准 STATUS")
            return

        if self._sig3_sweep_phase != "sample":
            return
        baseline = self._sig3_sweep_baseline
        if baseline is None:
            self._sig3_sweep_phase = "baseline"
            return
        baseline_seq = int(baseline.header.seq)
        if now >= self._sig3_sweep_deadline and seq != baseline_seq:
            self._record_sig3_sweep_sample(status, baseline)
            return
        if now >= self._sig3_sweep_timeout:
            self._finish_sig3_sweep("SIG3 扫门超时: 等待采样 STATUS")

    def _record_sig3_sweep_sample(self, status: StatusPacket, baseline: StatusPacket) -> None:
        self._sig3_sweep_timer.stop()
        delay_ns = self._sig3_sweep_points[self._sig3_sweep_index]
        row = {
            "delay_ns": int(delay_ns),
            "counter_1s": int(status.counter_1s),
            "sig3_delta": self._u32_delta_int(
                status.gate_divided_pulse_count,
                baseline.gate_divided_pulse_count,
            ),
            "trigger_delta": self._u32_delta_int(
                status.gate_trigger_count,
                baseline.gate_trigger_count,
            ),
            "output_delta": self._u32_delta_int(
                status.gate_output_pulse_count,
                baseline.gate_output_pulse_count,
            ),
            "status_seq": int(status.header.seq),
        }
        self._sig3_sweep_rows.append(row)
        self._append_sig3_sweep_row(row)
        self._update_sig3_sweep_plot()
        self._sig3_sweep_index += 1
        self._start_next_sig3_sweep_point()

    def _append_sig3_sweep_row(self, row: dict[str, int]) -> None:
        table_row = self.sig3_sweep_table.rowCount()
        self.sig3_sweep_table.setRowCount(table_row + 1)
        keys = ("delay_ns", "counter_1s", "sig3_delta", "trigger_delta", "output_delta", "status_seq")
        for col, key in enumerate(keys):
            item = QtWidgets.QTableWidgetItem(str(row[key]))
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.sig3_sweep_table.setItem(table_row, col, item)
        self.sig3_sweep_table.scrollToBottom()
        self.sig3_sweep_copy_btn.setEnabled(True)
        self.sig3_sweep_apply_best_btn.setEnabled(not self._sig3_sweep_active)

    def _update_sig3_sweep_plot(self) -> None:
        if not self._sig3_sweep_rows:
            self.sig3_sweep_cps_curve.setData([], [])
            self.sig3_sweep_gate_curve.setData([], [])
            return
        x = np.array([row["delay_ns"] for row in self._sig3_sweep_rows], dtype=float)
        cps = np.array([row["counter_1s"] for row in self._sig3_sweep_rows], dtype=float)
        sig3 = np.array([row["sig3_delta"] for row in self._sig3_sweep_rows], dtype=float)
        self.sig3_sweep_cps_curve.setData(x, cps)
        self.sig3_sweep_gate_curve.setData(x, sig3)

    def _finish_sig3_sweep(self, message: str, restore: bool = True) -> None:
        self._sig3_sweep_timer.stop()
        was_active = self._sig3_sweep_active
        self._set_sig3_sweep_running(False)
        self._sig3_sweep_phase = "idle"
        if was_active and restore and self.sig3_sweep_restore_check.isChecked():
            self._restore_sig3_sweep_state()
        self.sig3_sweep_status_label.setText(message)
        self.append_log(message)

    def _restore_sig3_sweep_state(self) -> None:
        if self._sig3_sweep_restore_state is None or self.controller is None:
            return
        delay_coarse, delay_fine, sig3_enable = self._sig3_sweep_restore_state
        result = self.controller.set_gate_signal(
            3,
            delay_coarse,
            delay_fine,
            self.sig3_width_coarse.value(),
            self.sig3_width_fine.value(),
        )
        self._show_command_result(result)
        if result.success:
            self.sig3_delay_coarse.setValue(delay_coarse)
            self.sig3_delay_fine.setValue(delay_fine)
        if self.sig3_sweep_auto_enable_check.isChecked() and self.sig3_enable_check.isChecked() != sig3_enable:
            enable_result = self.controller.set_gate_enable(
                self.sig2_enable_check.isChecked(),
                sig3_enable,
                self.pixel_mode_check.isChecked(),
            )
            self._show_command_result(enable_result)
            if enable_result.success:
                self.sig3_enable_check.setChecked(sig3_enable)

    def on_copy_sig3_sweep_csv(self) -> None:
        if not self._sig3_sweep_rows:
            return
        lines = ["delay_ns,counter_1s,sig3_delta,trigger_delta,output_delta,status_seq"]
        for row in self._sig3_sweep_rows:
            lines.append(
                f"{row['delay_ns']},{row['counter_1s']},{row['sig3_delta']},"
                f"{row['trigger_delta']},{row['output_delta']},{row['status_seq']}"
            )
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.sig3_sweep_status_label.setText(f"SIG3 扫门 CSV 已复制: {len(self._sig3_sweep_rows)} 点")

    def on_apply_best_sig3_sweep_delay(self) -> None:
        self._guard_action(self._apply_best_sig3_sweep_delay)

    def _apply_best_sig3_sweep_delay(self) -> None:
        if self.controller is None:
            raise RuntimeError("Controller is not bound.")
        if self._sig3_sweep_active:
            return
        if not self._sig3_sweep_rows:
            self.sig3_sweep_status_label.setText("SIG3 扫门: 没有可下发的结果")
            self.sig3_sweep_apply_best_btn.setEnabled(False)
            return

        best_index, best_row = max(
            enumerate(self._sig3_sweep_rows),
            key=lambda item: item[1]["counter_1s"],
        )
        delay_ns = int(best_row["delay_ns"])
        counter_1s = int(best_row["counter_1s"])
        result = self.controller.set_gate_sig3_long_delay(
            delay_ns,
            self.sig3_width_coarse.value(),
            self.sig3_width_fine.value(),
        )
        self._show_command_result(result)
        if result.success:
            self.sig3_sweep_table.selectRow(best_index)
            if delay_ns <= 159:
                self._set_sig3_delay_controls(delay_ns)
            self.sig3_sweep_status_label.setText(
                f"已下发最大计数延时: {delay_ns} ns (counter_1s={counter_1s})"
            )
        else:
            self.sig3_sweep_status_label.setText(f"最大计数延时下发失败: {result.message}")

    @staticmethod
    def _decode_temperature_c(raw_value: int) -> float:
        temp_c = (
            44.2244
            - 0.0038 * raw_value
            + 8.1502e-8 * (raw_value ** 2)
            - 7.4953e-13 * (raw_value ** 3)
        )
        return round(temp_c, 2)

    def _push_monitor_sample(self, temp_c: float, cps: int) -> None:
        self._monitor_tick += 1
        self.monitor_index.append(self._monitor_tick)
        self.temp_history.append(temp_c)
        self.cps_history.append(cps)
        x = np.array(self.monitor_index, dtype=float)
        self.temp_trend_curve.setData(x, np.array(self.temp_history, dtype=float))
        self.cps_trend_curve.setData(x, np.array(self.cps_history, dtype=float))

    def _update_config_from_controls(self) -> None:
        cfg = self.controller.config
        cfg.target_temperature_c = self.temp_target_spin.value()
        cfg.analog_targets.laser_sync_threshold_mv = self.laser_thr_spin.value()
        cfg.analog_targets.pixel_sync_threshold_mv = self.pixel_thr_spin.value()
        cfg.analog_targets.avalanche_threshold_mv = self.avalanche_thr_spin.value()
        cfg.analog_targets.bias_voltage_v = self.bias_spin.value()
        cfg.gate_settings.hold_off_time = self.gate_holdoff_spin.value()
        cfg.gate_settings.divider = self.gate_div_spin.value()
        cfg.gate_settings.sig2_enable = self.sig2_enable_check.isChecked()
        cfg.gate_settings.sig3_enable = self.sig3_enable_check.isChecked()
        cfg.gate_settings.pixel_mode = self.pixel_mode_check.isChecked()
        cfg.gate_settings.sig2_delay_coarse = self.sig2_delay_coarse.value()
        cfg.gate_settings.sig2_delay_fine = self.sig2_delay_fine.value()
        cfg.gate_settings.sig2_width_coarse = self.sig2_width_coarse.value()
        cfg.gate_settings.sig2_width_fine = self.sig2_width_fine.value()
        cfg.gate_settings.sig3_delay_coarse = self.sig3_delay_coarse.value()
        cfg.gate_settings.sig3_delay_fine = self.sig3_delay_fine.value()
        cfg.gate_settings.sig3_width_coarse = self.sig3_width_coarse.value()
        cfg.gate_settings.sig3_width_fine = self.sig3_width_fine.value()
        cfg.device_index = self._current_device_index()
        cfg.read_pipe = self._parse_int(self.read_pipe_edit.text())
        cfg.write_pipe = self._parse_int(self.write_pipe_edit.text())
        cfg.read_block_size = self.read_block_spin.value()
        cfg.read_enabled = self.read_enable_check.isChecked()

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def on_refresh_devices(self) -> None:
        devices = self.controller.refresh_devices()
        self.device_combo.clear()
        self._device_entries = []
        for device in devices:
            label = f"{device.index}: {device.description or 'FT60x Device'}"
            self._device_entries.append((device.index, label))
            self.device_combo.addItem(label, device.index)

        if not devices:
            self.device_combo.addItem("No FT60x devices found", 0)
            self.append_log("Device scan finished: no FT60x devices found.")
        else:
            self.append_log(f"Device scan finished: found {len(devices)} FT60x device(s).")
            current_index = self.controller.config.device_index
            combo_index = self.device_combo.findData(current_index)
            if combo_index >= 0:
                self.device_combo.setCurrentIndex(combo_index)

    def on_connect_device(self) -> None:
        def action() -> None:
            self._update_config_from_controls()
            self.controller.connect_device(
                device_index=self.controller.config.device_index,
                read_pipe=self.controller.config.read_pipe,
                write_pipe=self.controller.config.write_pipe,
                read_block_size=self.controller.config.read_block_size,
                read_enabled=self.controller.config.read_enabled,
            )
            self.controller.save_current_config()

        self._guard_action(action)

    def on_read_enable_toggled(self, enabled: bool) -> None:
        if self.controller is None:
            return

        def action() -> None:
            self.controller.set_read_enabled(enabled)

        self._guard_action(action)

    def on_read_state_changed(self, enabled: bool) -> None:
        if self.read_enable_check.isChecked() != enabled:
            self.read_enable_check.blockSignals(True)
            self.read_enable_check.setChecked(enabled)
            self.read_enable_check.blockSignals(False)
        self.append_log(f"RX reader state: {'enabled' if enabled else 'disabled'}")

    def on_run_self_test(self) -> None:
        def action() -> None:
            results = self.controller.run_basic_self_test()
            self.self_test_view.setPlainText("\n".join(results))
            for line in results:
                self.append_log(f"[SelfTest] {line}")

        self._guard_action(action)

    def on_save_config(self) -> None:
        def action() -> None:
            self._update_config_from_controls()
            self.controller.save_current_config()
            self.append_log(f"Config saved to {self.controller.default_config_path}")

        self._guard_action(action)

    def on_device_state_changed(self, connected: bool, text: str) -> None:
        self.device_status_label.setText("已连接" if connected else "未连接")
        self.header_status_chip.setText("在线" if connected else "未连接")
        self._set_header_connection_state(connected)
        if not connected:
            self._gpx2_readback_pending = False
            self._gpx2_readback_temp_rx = False
            self._gpx2_readback_status_count_at_request = -1
            self._gpx2_readback_timer.stop()
        self.append_log(f"Device state: {text}")

    @staticmethod
    def _format_usb_rate(bytes_per_second: float) -> str:
        rate = max(0.0, float(bytes_per_second))
        if rate >= 1_000_000.0:
            return f"{rate / 1_000_000.0:.2f} MB/s"
        return f"{rate / 1_000.0:.1f} kB/s"

    def on_stats_updated(self, stats, rates) -> None:
        self._last_runtime_stats = stats
        self._last_runtime_rates = dict(rates)
        rx_rate = self._format_usb_rate(rates["rx_bytes_per_sec"])
        tx_rate = self._format_usb_rate(rates["tx_bytes_per_sec"])
        packet_rate = f"{rates['packets_per_sec']:.1f} /s"
        event_rate = f"{rates['tdc_events_per_sec']:.1f} /s"
        raw_event_rate = f"{rates.get('tdc_raw_events_per_sec', 0.0):.1f}/s"
        self.rx_rate_label.setText(rx_rate)
        self.tx_rate_label.setText(tx_rate)
        self.packet_rate_label.setText(packet_rate)
        self.event_rate_label.setText(event_rate)
        self.global_rx_rate_label.setText(f"USB RX: {rx_rate}")
        self.global_tx_rate_label.setText(f"USB TX: {tx_rate}")
        self.global_packet_rate_label.setText(f"Packets: {packet_rate}")
        self.global_raw_event_rate_label.setText(f"Raw events: {raw_event_rate}")
        if self._last_tdc_test_label_core:
            self._set_tdc_test_status_text(self._last_tdc_test_label_core)

    @staticmethod
    def _format_tdc_raw_age(age_ms: float) -> str:
        if age_ms < 0:
            return "-"
        if age_ms >= 1000.0:
            return f"{age_ms / 1000.0:.2f} s"
        return f"{age_ms:.0f} ms"

    @staticmethod
    def _u32_delta(current: int, baseline: int | None) -> str:
        if baseline is None:
            return "-"
        delta = int(current) - int(baseline)
        if delta < 0:
            delta += 1 << 32
        return str(delta)

    def _format_tdc_transport_diag(self) -> str:
        stats = self._last_runtime_stats
        rates = self._last_runtime_rates
        status = self._last_status_packet
        raw_packets = int(getattr(stats, "tdc_raw_packet_count", 0)) if stats is not None else 0
        raw_events = int(getattr(stats, "tdc_raw_event_count", 0)) if stats is not None else 0
        raw_channels = getattr(stats, "last_tdc_raw_channels", (0, 0, 0, 0)) if stats is not None else (0, 0, 0, 0)
        age_ms = float(getattr(stats, "last_tdc_raw_packet_age_ms", -1.0)) if stats is not None else -1.0
        rx_running = bool(getattr(stats, "rx_reader_running", False)) if stats is not None else False
        rx_rate = self._format_usb_rate(float(rates.get("rx_bytes_per_sec", 0.0)))
        packet_rate = float(rates.get("packets_per_sec", 0.0))
        raw_packet_rate = float(rates.get("tdc_raw_packets_per_sec", 0.0))
        raw_event_rate = float(rates.get("tdc_raw_events_per_sec", 0.0))
        running = self.tdc_test_enable_check.isChecked()
        stalled = running and (
            not rx_running
            or age_ms > 500.0
            or (age_ms < 0.0 and packet_rate > 0.0)
            or (raw_packets > 0 and raw_event_rate < 1000.0 and age_ms > 250.0)
        )
        rx_health = "RX STALLED" if stalled else ("RX OK" if rx_running else "RX OFF")
        usb_drop_delta = self._u32_delta(
            int(status.usb_drop_count) if status is not None else 0,
            self._tdc_test_usb_drop_baseline,
        )
        payload_text = "-" if self._tdc_test_upload_payload_word is None else f"0x{self._tdc_test_upload_payload_word:08X}"
        return (
            f"{rx_health} | raw packets/events: {raw_packets}/{raw_events} "
            f"({raw_packet_rate:.1f} pkt/s, {raw_event_rate:.1f} ev/s) | "
            f"last raw age: {self._format_tdc_raw_age(age_ms)} | "
            f"raw CH: {raw_channels[0]}/{raw_channels[1]}/{raw_channels[2]}/{raw_channels[3]} | "
            f"USB RX: {rx_rate} | packets: {packet_rate:.1f}/s | "
            f"FPGA usb_drop delta: {usb_drop_delta} | upload: {payload_text}"
        )

    def _set_tdc_test_status_text(self, core_text: str) -> None:
        self.tdc_test_status_label.setText(f"{core_text} | {self._format_tdc_transport_diag()}")

    def on_status_packet(self, status: StatusPacket) -> None:
        self._last_status_packet = status
        readback_signature = (
            (int(status.gpx2_cfg_diag), tuple(status.gpx2_cfg_readback[:17]))
            if status.gpx2_cfg_readback
            else None
        )
        force_gpx2_refresh = bool(
            readback_signature
            and (self._gpx2_readback_pending or readback_signature != self._last_gpx2_render_signature)
        )
        now = time.monotonic()
        if (now - self._last_status_ui_update_monotonic) < 0.2 and not force_gpx2_refresh:
            return
        self._last_status_ui_update_monotonic = now

        decoded = self.controller.decode_status_flags(status.flags)
        temp_c = self._decode_temperature_c(status.temp_avg_raw)
        cps = int(status.counter_1s)
        readback_regs = status.gpx2_cfg_readback
        readback_hex = self._format_gpx2_config_bytes(readback_regs) if readback_regs else ""
        if force_gpx2_refresh:
            self._update_gpx2_readback_status(status)

        self.temp_current_label.setText(f"当前温度: {temp_c:.2f} °C")
        self.counter_label.setText(f"counter_1s: {cps}")
        self.drop_label.setText(f"tdc_drop: {status.tdc_drop_count} / usb_drop: {status.usb_drop_count}")

        self.temp_value_big_label.setText(f"{temp_c:.2f} °C")
        self.temp_detail_label.setText(f"raw = {status.temp_avg_raw}")
        self.cps_value_big_label.setText(f"{cps} cps")
        self.cps_detail_label.setText(f"uptime = {status.uptime_seconds}s")

        self._push_monitor_sample(temp_c, cps)

        summary = {
            "uptime_seconds": status.uptime_seconds,
            "flags_hex": f"0x{status.flags:04X}",
            "flash_busy": decoded.flash_busy,
            "flash_error": decoded.flash_error,
            "gpx2_cfg_done": decoded.gpx2_cfg_done,
            "gpx2_cfg_error": decoded.gpx2_cfg_error,
            "gpx2_event_overflow": decoded.gpx2_event_overflow,
            "usb_tx_backpressure": decoded.usb_tx_backpressure,
            "gate_clk_locked": decoded.gate_clk_locked,
            "gpx2_lclk_locked": decoded.gpx2_lclk_locked,
            "temp_avg_raw": status.temp_avg_raw,
            "temp_c": temp_c,
            "counter_1s": cps,
            "tdc_drop_count": status.tdc_drop_count,
            "usb_drop_count": status.usb_drop_count,
            "gpx2_raw_count_ch1": status.gpx2_raw_count_ch1,
            "gpx2_raw_count_ch2": status.gpx2_raw_count_ch2,
            "gpx2_raw_count_ch3": status.gpx2_raw_count_ch3,
            "gpx2_raw_count_ch4": status.gpx2_raw_count_ch4,
            "gate_trigger_count": status.gate_trigger_count,
            "gate_direct_pulse_count": status.gate_direct_pulse_count,
            "gate_divided_pulse_count": status.gate_divided_pulse_count,
            "gate_output_pulse_count": status.gate_output_pulse_count,
            "gpx2_cfg_profile": (status.gpx2_cfg_diag >> 19) & 0xF,
            "gpx2_cfg_sequence": (status.gpx2_cfg_diag >> 23) & 0xF,
            "gpx2_cfg_readback_mismatch": bool((status.gpx2_cfg_diag >> 18) & 0x1),
            "gpx2_cfg_mismatch_count": (status.gpx2_cfg_diag >> 13) & 0x1F,
            "gpx2_cfg_first_mismatch_idx": (status.gpx2_cfg_diag >> 8) & 0x1F,
            "gpx2_cfg_first_mismatch_rx": f"0x{status.gpx2_cfg_diag & 0xFF:02X}",
            "gpx2_cfg_readback_hex": readback_hex,
        }
        if readback_regs:
            summary["gpx2_cfg_readback_decoded"] = protocol.describe_gpx2_config_bytes(readback_regs)
        self.status_flags_view.setPlainText(json.dumps(summary, ensure_ascii=False, indent=2))

    def on_temperature_target_changed(self, temp_c: float, raw_code: int) -> None:
        self.temp_code_label.setText(f"raw: {raw_code}")
        self.append_log(f"Temperature target set: {temp_c:.2f} C -> raw {raw_code}")

    def on_analog_targets_changed(self, targets, codes) -> None:
        self.analog_code_label.setText(
            "codes: ch1={0} ch2={1} ch3={2} ch4={3}".format(
                codes.laser_sync_code,
                codes.pixel_sync_code,
                codes.avalanche_code,
                codes.bias_code,
            )
        )
        self.append_log(
            "Analog set: "
            f"laser={targets.laser_sync_threshold_mv}mV "
            f"pixel={targets.pixel_sync_threshold_mv}mV "
            f"avalanche={targets.avalanche_threshold_mv}mV "
            f"bias={targets.bias_voltage_v}V"
        )

    def on_histogram_snapshot(self, snapshot: HistogramSnapshot) -> None:
        if snapshot.histograms:
            key = f"{snapshot.current_row},{snapshot.current_col}"
            hist = snapshot.histograms.get(key)
            if hist is None:
                hist = next(iter(snapshot.histograms.values()))
            self.hist_curve.setData(np.arange(len(hist)), np.array(hist))
        else:
            self.hist_curve.setData([], [])

        if snapshot.image_projection:
            self.image_view.setImage(np.array(snapshot.image_projection, dtype=float).T, autoLevels=True)

    def _tdc_test_plot_series(
        self,
        values: np.ndarray,
        bin_width: int,
        bin_offset: int,
        start_bin: int,
        x_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        def _step_series(counts: np.ndarray, left_bins: np.ndarray, right_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            y_values = np.log10(np.maximum(counts, 1).astype(float, copy=False))
            left = (left_bins * bin_width + bin_offset) * 0.008 * x_scale
            right = (right_bins * bin_width + bin_offset) * 0.008 * x_scale
            x_step = np.empty(y_values.size * 2, dtype=float)
            y_step = np.empty(y_values.size * 2, dtype=float)
            x_step[0::2] = left
            x_step[1::2] = right
            y_step[0::2] = y_values
            y_step[1::2] = y_values
            return x_step, y_step

        if values.size <= self._tdc_test_plot_max_points:
            left_bins = np.arange(values.size, dtype=float) + float(start_bin)
            return _step_series(values, left_bins, left_bins + 1.0)

        factor = max(1, int(np.ceil(values.size / self._tdc_test_plot_max_points)))
        group_count = values.size // factor
        grouped = values[: group_count * factor].reshape(group_count, factor).mean(axis=1)
        left_bins = np.arange(group_count, dtype=float) * factor + float(start_bin)
        right_bins = left_bins + float(factor)
        if group_count * factor < values.size:
            grouped = np.append(grouped, values[group_count * factor :].mean())
            left_bins = np.append(left_bins, group_count * factor + float(start_bin))
            right_bins = np.append(right_bins, values.size + float(start_bin))
        return _step_series(grouped, left_bins, right_bins)

    @staticmethod
    def _tdc_linear_step_series(
        values: np.ndarray,
        bin_width: int,
        bin_offset: int,
        start_bin: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.asarray(values, dtype=float)
        left_bins = np.arange(counts.size, dtype=float) + float(start_bin)
        right_bins = left_bins + 1.0
        left = (left_bins * bin_width + bin_offset) * 0.008
        right = (right_bins * bin_width + bin_offset) * 0.008
        x_step = np.empty(counts.size * 2, dtype=float)
        y_step = np.empty(counts.size * 2, dtype=float)
        x_step[0::2] = left
        x_step[1::2] = right
        y_step[0::2] = counts
        y_step[1::2] = counts
        return x_step, y_step

    def _clear_tdc_peak_zoom(self) -> None:
        self.tdc_peak_curve.setData([], [])
        for marker in (self.tdc_peak_half_line, self.tdc_peak_left_line, self.tdc_peak_right_line):
            marker.setVisible(False)

    def _update_tdc_peak_zoom(
        self,
        values: np.ndarray,
        shape,
        bin_width: int,
        bin_offset: int,
    ) -> None:
        if values.size == 0 or shape.peak_bin < 0 or shape.peak_count <= 0:
            self._clear_tdc_peak_zoom()
            return
        bin_width_ns = bin_width * 0.008
        half_span_ns = max(5.0, shape.fwhm_ns * 8.0, bin_width_ns * 20.0)
        left_raw = max(0.0, (shape.peak_ns - half_span_ns) / 0.008)
        right_raw = min(float(values.size * bin_width + bin_offset), (shape.peak_ns + half_span_ns) / 0.008)
        start_bin = max(0, int(np.floor((left_raw - bin_offset) / bin_width)))
        stop_bin = min(values.size, int(np.ceil((right_raw - bin_offset) / bin_width)) + 1)
        if stop_bin <= start_bin:
            start_bin = max(0, shape.peak_bin - 16)
            stop_bin = min(values.size, shape.peak_bin + 17)
        zoom_values = values[start_bin:stop_bin]
        x_axis, y_axis = self._tdc_linear_step_series(zoom_values, bin_width, bin_offset, start_bin)
        self.tdc_peak_curve.setData(x_axis, y_axis)
        self.tdc_peak_half_line.setValue(float(shape.half_count))
        self.tdc_peak_left_line.setValue(float(shape.fwhm_left_ns))
        self.tdc_peak_right_line.setValue(float(shape.fwhm_right_ns))
        for marker in (self.tdc_peak_half_line, self.tdc_peak_left_line, self.tdc_peak_right_line):
            marker.setVisible(True)
        self.tdc_peak_plot.setXRange(
            float((start_bin * bin_width + bin_offset) * 0.008),
            float(((max(start_bin, stop_bin - 1) + 1) * bin_width + bin_offset) * 0.008),
            padding=0.02,
        )

    @staticmethod
    def _tdc_axis_scale(max_ns: float) -> tuple[float, str]:
        if max_ns >= 1000.0:
            return 0.001, "us"
        if 0.0 < max_ns < 1.0:
            return 1000.0, "ps"
        return 1.0, "ns"

    @staticmethod
    def _format_tdc_time(ns: float) -> str:
        if ns >= 1_000_000.0:
            return f"{ns * 0.000001:.3f} ms"
        if ns >= 1000.0:
            return f"{ns * 0.001:.3f} us"
        if 0.0 <= ns < 1.0:
            return f"{ns * 1000.0:.1f} ps"
        return f"{ns:.3f} ns"

    @staticmethod
    def _tdc_afterpulse_stats(
        values: np.ndarray,
        *,
        min_bin: int,
        peak_bin: int,
        bin_width: int,
        bin_offset: int,
        threshold: int,
        afterpulse_delay_ns: float,
        tail_fraction: float = 0.2,
    ) -> dict[str, float | int]:
        if values.size == 0 or peak_bin < 0 or peak_bin >= values.size:
            return {"pc": 0, "tc": 0, "n": 0, "dc": 0.0, "app": 0.0, "after_start_bin": min_bin}

        peak_count = int(values[peak_bin])
        active_threshold = int(threshold)
        if active_threshold <= 0:
            active_threshold = max(1, int(np.ceil(peak_count * 0.5)))

        left = peak_bin
        while left > min_bin and int(values[left - 1]) > active_threshold:
            left -= 1

        right = peak_bin
        while right < values.size - 1 and int(values[right + 1]) > active_threshold:
            right += 1

        prompt_region = values[left : right + 1]
        prompt_mask = prompt_region > active_threshold
        pc = int(prompt_region[prompt_mask].sum()) if np.any(prompt_mask) else peak_count

        peak_ns = (peak_bin * bin_width + bin_offset) * 0.008
        after_start_ns = peak_ns + max(0.0, float(afterpulse_delay_ns))
        after_start_bin = int(np.ceil(((after_start_ns / 0.008) - bin_offset) / max(1, bin_width)))
        after_start_bin = max(min_bin, right + 1, min(values.size, after_start_bin))
        after_region = values[after_start_bin:]
        n = int(after_region.size)
        tc = int(after_region.sum()) if n else 0

        analysis_region = values[min_bin:]
        tail_count = max(1, int(np.ceil(analysis_region.size * tail_fraction))) if analysis_region.size else 0
        dc = float(analysis_region[-tail_count:].mean()) if tail_count else 0.0
        app = max(0.0, ((float(tc) - dc * float(n)) / float(pc) * 100.0)) if pc > 0 else 0.0
        return {"pc": pc, "tc": tc, "n": n, "dc": dc, "app": app, "after_start_bin": after_start_bin}

    @staticmethod
    def _tdc_background_stats(values: np.ndarray, *, min_bin: int, peak_bin: int) -> dict[str, float]:
        if values.size == 0 or peak_bin < min_bin or peak_bin >= values.size:
            return {"mean": 0.0, "std": 0.0, "peak_to_bg": 0.0}
        region = values[min_bin:].astype(float, copy=False)
        if region.size == 0:
            return {"mean": 0.0, "std": 0.0, "peak_to_bg": 0.0}
        local_peak = peak_bin - min_bin
        mask = np.ones(region.size, dtype=bool)
        half_width = 2
        left = max(0, local_peak - half_width)
        right = min(region.size, local_peak + half_width + 1)
        mask[left:right] = False
        background = region[mask] if np.any(mask) else region
        bg_mean = float(background.mean()) if background.size else 0.0
        bg_std = float(background.std()) if background.size else 0.0
        peak_count = float(values[peak_bin])
        peak_to_bg = peak_count / bg_mean if bg_mean > 0.0 else (peak_count if peak_count > 0.0 else 0.0)
        return {"mean": bg_mean, "std": bg_std, "peak_to_bg": peak_to_bg}

    @staticmethod
    def _tdc_satellite_stats(
        values: np.ndarray,
        start_repeat: np.ndarray | None,
        stop_repeat: np.ndarray | None,
        shape,
        *,
        bin_width: int,
        bin_offset: int,
    ) -> str:
        if values.size == 0 or shape.peak_bin < 0 or shape.peak_count <= 0:
            return ""
        bin_width_ns = max(1, int(bin_width)) * 0.008
        threshold = max(
            3,
            int(np.ceil(float(shape.background_mean) + 6.0 * float(shape.background_std))),
            int(np.ceil(float(shape.peak_count) * 0.01)),
        )
        exclude_bins = max(3, int(round(2.0 / max(bin_width_ns, 1e-12))))
        min_sep_bins = max(1, int(round(5.0 / max(bin_width_ns, 1e-12))))
        search_lo = min(values.size - 1, int(shape.peak_bin) + exclude_bins)
        search_hi = min(
            values.size - 1,
            int(shape.peak_bin) + max(exclude_bins + 1, int(round(700.0 / max(bin_width_ns, 1e-12)))),
        )
        if not (0 < search_lo < search_hi):
            return " | satellites: none"
        values_i = values.astype(np.int64, copy=False)
        candidate_region = np.arange(search_lo, search_hi, dtype=np.int64)
        is_peak = (
            (values_i[candidate_region] >= values_i[candidate_region - 1])
            & (values_i[candidate_region] >= values_i[candidate_region + 1])
            & (values_i[candidate_region] >= threshold)
        )
        candidates = candidate_region[is_peak]
        if candidates.size == 0:
            return " | satellites: none"
        candidates = candidates[np.argsort(values_i[candidates])[::-1]]
        selected: list[int] = []
        for idx in candidates:
            if all(abs(int(idx) - existing) >= min_sep_bins for existing in selected):
                selected.append(int(idx))
            if len(selected) >= 4:
                break
        selected.sort()

        def repeat_hits(histogram: np.ndarray | None, satellite_bin: int) -> int:
            if histogram is None or histogram.size == 0:
                return 0
            offset_bin = max(0, int(satellite_bin) - int(shape.peak_bin))
            lo = max(0, offset_bin - 2)
            hi = min(histogram.size, offset_bin + 3)
            return int(np.asarray(histogram[lo:hi], dtype=np.uint64).sum())

        parts: list[str] = []
        total_start_hits = 0
        total_stop_hits = 0
        for idx in selected:
            offset_ns = (idx - int(shape.peak_bin)) * max(1, int(bin_width)) * 0.008
            dt_ns = (idx * max(1, int(bin_width)) + int(bin_offset)) * 0.008
            start_hits = repeat_hits(start_repeat, idx)
            stop_hits = repeat_hits(stop_repeat, idx)
            total_start_hits += start_hits
            total_stop_hits += stop_hits
            parts.append(
                f"{dt_ns:.1f}ns(+{offset_ns:.1f},"
                f"{int(values_i[idx])},sr={start_hits},pr={stop_hits})"
            )
        if total_stop_hits > max(5, total_start_hits * 3):
            source = "likely CH stop repeat"
        elif total_start_hits > max(5, total_stop_hits * 3):
            source = "likely CH start repeat"
        else:
            source = "source inconclusive"
        return f" | satellites: {source} " + "; ".join(parts)

    def on_tdc_test_snapshot(self, snapshot: HistogramSnapshot) -> None:
        hist = snapshot.histograms.get("tdc_test") if snapshot.histograms else None
        start_repeat_hist = snapshot.histograms.get("tdc_start_repeat") if snapshot.histograms else None
        stop_repeat_hist = snapshot.histograms.get("tdc_stop_repeat") if snapshot.histograms else None
        counts = snapshot.image_projection[0] if snapshot.image_projection else []
        accepted_count = int(counts[0]) if len(counts) > 0 else 0
        sync_count = int(counts[1]) if len(counts) > 1 else 0
        photon_count = int(counts[2]) if len(counts) > 2 else 0
        last_dt = int(counts[3]) if len(counts) > 3 else -1
        min_dt = int(counts[4]) if len(counts) > 4 else -1
        max_dt = int(counts[5]) if len(counts) > 5 else -1
        zero_dt = int(counts[6]) if len(counts) > 6 else 0
        neg_dt = int(counts[7]) if len(counts) > 7 else 0
        out_window = int(counts[8]) if len(counts) > 8 else 0
        ch_counts = [int(counts[idx]) if len(counts) > idx else 0 for idx in range(9, 13)]
        orphan_photon = int(counts[13]) if len(counts) > 13 else 0
        paired_photon = int(counts[14]) if len(counts) > 14 else 0
        timestamp_regression = int(counts[15]) if len(counts) > 15 else 0
        start_buffer_trim = int(counts[16]) if len(counts) > 16 else 0
        legacy_fold_risk = int(counts[17]) if len(counts) > 17 else 0
        photon_before_history = int(counts[18]) if len(counts) > 18 else 0
        photon_after_history = int(counts[19]) if len(counts) > 19 else 0
        sync_buffer_len = int(counts[20]) if len(counts) > 20 else 0
        sync_history_depth_raw = int(counts[21]) if len(counts) > 21 else 0
        sync_period_raw = int(counts[22]) if len(counts) > 22 else 0
        deadtime_filtered = int(counts[23]) if len(counts) > 23 else 0
        stale_batch_count = int(counts[24]) if len(counts) > 24 else 0
        raw_epoch = int(counts[25]) if len(counts) > 25 else 0
        reference_filtered = int(counts[26]) if len(counts) > 26 else 0
        hist_status_flags = int(counts[27]) if len(counts) > 27 else 0
        hist_expected_chunks = int(counts[28]) if len(counts) > 28 else 0
        hist_received_chunks = int(counts[29]) if len(counts) > 29 else 0
        hist_missing_chunks = int(counts[30]) if len(counts) > 30 else 0
        hist_partial_flag = int(counts[31]) if len(counts) > 31 else 0
        hist_last_chunk_index = int(counts[32]) if len(counts) > 32 else 0
        if hist is None:
            self.tdc_test_curve.setData([], [])
            self._clear_tdc_peak_zoom()
            self._last_tdc_test_label_core = "sync: 0 / photon: 0 / pairs: 0 / peak: -"
            self._set_tdc_test_status_text(self._last_tdc_test_label_core)
            return
        values = np.asarray(hist, dtype=np.uint32)
        bin_width = max(1, self.tdc_bin_width_spin.value())
        bin_offset = self.tdc_bin_offset_spin.value()
        min_bin = min(max(0, self.tdc_display_min_bin_spin.value()), max(0, values.size - 1))
        view_values = values[min_bin:]
        max_ns = ((values.size - 1) * bin_width + bin_offset) * 0.008 if values.size else 0.0
        x_scale, x_unit = self._tdc_axis_scale(max_ns)
        self.tdc_test_plot.getPlotItem().setLabel("bottom", f"dt from reference ({x_unit})")
        x_axis, plot_values = self._tdc_test_plot_series(view_values, bin_width, bin_offset, min_bin, x_scale)
        self.tdc_test_curve.setData(x_axis, plot_values)
        view_total = int(view_values.sum()) if view_values.size else 0

        def _fmt_dt(raw: int) -> str:
            return "-" if raw < 0 else self._format_tdc_time(raw * 0.008)

        bin_width_ns = bin_width * 0.008
        window_ns = len(values) * bin_width_ns
        view_start_ns = (min_bin * bin_width + bin_offset) * 0.008
        view_end_ns = ((values.size - 1) * bin_width + bin_offset) * 0.008 if values.size else view_start_ns
        pairs_per_photon = accepted_count / max(1, photon_count)
        no_pair_ratio = orphan_photon / max(1, photon_count)
        pairing_mode = str(self.tdc_pairing_combo.currentData() or "prev_start")
        pairing_label = (
            "commercial/nearest-start" if pairing_mode == "prev_start" else "correlation/all-start"
        )
        time_formula = "forward"
        diag = (
            f"dt last/min/max: {_fmt_dt(last_dt)} / {_fmt_dt(min_dt)} / {_fmt_dt(max_dt)} | "
            f"bin: {bin_width} ticks = {self._format_tdc_time(bin_width_ns)} | "
            f"window: {self._format_tdc_time(window_ns)} | "
            f"view: bin {min_bin}..{max(0, values.size - 1)} "
            f"({self._format_tdc_time(view_start_ns)}..{self._format_tdc_time(view_end_ns)}) | "
            f"pairs/photon: {pairs_per_photon:.2f} ({paired_photon} photons paired) | "
            f"zero/neg/out/no-pair: {zero_dt}/{neg_dt}/{out_window}/{orphan_photon} | "
            f"no-pair ratio: {no_pair_ratio:.3f} | "
            f"no-pair before/after: {photon_before_history}/{photon_after_history} | "
            f"diag reg/trim/fold: {timestamp_regression}/{start_buffer_trim}/{legacy_fold_risk} | "
            f"deadtime_filtered: {deadtime_filtered} | ref_cleanup_filtered: {reference_filtered} | "
            f"stale batches: {stale_batch_count} | epoch: {raw_epoch} | "
            f"sync buffer: {sync_buffer_len} / {self._format_tdc_time(sync_history_depth_raw * 0.008)} | "
            f"sync period: {self._format_tdc_time(sync_period_raw * 0.008) if sync_period_raw else '-'} | "
            f"pairing/formula: {pairing_label}/{time_formula} | "
            f"CH: {ch_counts[0]}/{ch_counts[1]}/{ch_counts[2]}/{ch_counts[3]} | "
            f"hist chunks: {hist_received_chunks}/{hist_expected_chunks} "
            f"missing {hist_missing_chunks} last {hist_last_chunk_index}"
            f"{' PARTIAL' if hist_partial_flag else ''} | "
            f"fpga hist status: 0x{hist_status_flags:08X} "
            f"({protocol.describe_tdc_hist_status_flags(hist_status_flags)})"
        )
        source_data = str(self.tdc_source_combo.currentData())
        source_tag = (
            "SYNTHETIC | "
            if source_data == "synthetic"
            else "GPX2 EXAMPLE SPI | "
            if source_data == "gpx2_spi"
            else "GPX2 FPGA HIST | "
            if source_data == "gpx2_fpga"
            else "GPX2 RAW EXT | "
        )
        if view_total > 0:
            peak_bin = min_bin + int(view_values.argmax())
            peak_count = int(values[peak_bin])
            peak_ns = (peak_bin * bin_width + bin_offset) * 0.008
            metrics = TcspcCommercialAnalyzer.analyze(
                values,
                TdcTestSettings(
                    enabled=True,
                    start_channel=0,
                    stop_channel=1,
                    pairing_mode=pairing_mode,
                    time_formula=time_formula,
                    bin_width_raw=bin_width,
                    bin_offset=bin_offset,
                    bin_count=int(values.size),
                    tdc_deadtime_ps=float(self.tdc_deadtime_ps_spin.value()),
                    reference_deadtime_ns=float(self.tdc_reference_deadtime_ns_spin.value()),
                    detector_deadtime_ns=float(self.tdc_afterpulse_delay_spin.value()),
                ),
                min_bin=min_bin,
                start_repeat_histogram=(
                    np.asarray(start_repeat_hist, dtype=np.uint32) if start_repeat_hist is not None else None
                ),
                stop_repeat_histogram=(
                    np.asarray(stop_repeat_hist, dtype=np.uint32) if stop_repeat_hist is not None else None
                ),
                expected_peak_separation_raw=sync_period_raw if sync_period_raw > 0 else None,
            )
            shape = metrics.shape
            self._update_tdc_peak_zoom(values, shape, bin_width, bin_offset)
            afterpulse = self._tdc_afterpulse_stats(
                values,
                min_bin=min_bin,
                peak_bin=peak_bin,
                bin_width=bin_width,
                bin_offset=bin_offset,
                threshold=self.tdc_prompt_threshold_spin.value(),
                afterpulse_delay_ns=self.tdc_afterpulse_delay_spin.value(),
            )
            satellite_items = "; ".join(
                f"{item.dt_ns:.1f}ns(+{item.offset_ns:.1f},"
                f"{item.count},sr={item.start_repeat_hits},pr={item.stop_repeat_hits})"
                for item in metrics.satellite_peaks[:4]
            )
            satellite_diag = (
                " | satellites: none"
                if not metrics.satellite_peaks
                else f" | satellites: {metrics.satellite_decision} {satellite_items}"
            )
            invalid_reasons: list[str] = []
            hist_status = protocol.decode_tdc_hist_status_flags(hist_status_flags)
            if timestamp_regression > 0:
                invalid_reasons.append("timestamp_regression")
            if stale_batch_count > 0 and accepted_count == 0:
                invalid_reasons.append("only_stale_batches")
            if source_data == "gpx2" and photon_count >= 1000 and no_pair_ratio > 0.05:
                invalid_reasons.append(f"pairing_unhealthy_no_pair={no_pair_ratio:.3f}")
            if metrics.deadtime_violation_ratio > 0.01:
                invalid_reasons.append(
                    f"physical_deadtime_violation={metrics.deadtime_violation_ratio:.3f}"
                )
            if source_data == "gpx2_fpga" and hist_status["ddr_backend"] and not hist_status["ddr_calibrated"]:
                invalid_reasons.append("ddr_not_calibrated")
            if hist_status["ddr_ring_overflow"]:
                invalid_reasons.append("ddr_ring_overflow")
            if hist_status["ddr_storage_stall"]:
                invalid_reasons.append("ddr_storage_stall")
            if hist_status["ddr_cdc_req_overflow"] or hist_status["ddr_cdc_rsp_overflow"]:
                invalid_reasons.append("ddr_cdc_overflow")
            validity = "INVALID HISTOGRAM" if invalid_reasons else "VALID"
            afterpulse_diag = (
                f" | afterpulse: {float(afterpulse['app']):.3f}% "
                f"(start bin {int(afterpulse['after_start_bin'])}, "
                f"PC {int(afterpulse['pc'])}, TC {int(afterpulse['tc'])}, "
                f"DC {float(afterpulse['dc']):.2f}/bin)"
                f" | bg {metrics.background_mean:.2f}+/-{metrics.background_std:.2f}/bin "
                f"CV {metrics.background_cv:.2f} P/B {metrics.peak_to_bg:.2f}"
                f" | deadtime violation {metrics.deadtime_violation_count} "
                f"({metrics.deadtime_violation_ratio * 100.0:.2f}%, "
                f"{self._format_tdc_time(metrics.deadtime_start_ns)}.."
                f"{self._format_tdc_time(metrics.deadtime_end_ns)})"
                f" | half {shape.half_count:.1f} "
                f"FWHM {self._format_tdc_time(shape.fwhm_ns)} "
                f"({self._format_tdc_time(shape.fwhm_left_ns)}.."
                f"{self._format_tdc_time(shape.fwhm_right_ns)}) "
                f"RMS {self._format_tdc_time(shape.rms_ns)} "
                f"sep {self._format_tdc_time(shape.peak_separation_ns) if shape.peak_separation_raw else '-'}"
                f"{satellite_diag}"
            )
            self._last_tdc_test_label_core = (
                f"{source_tag}sync: {sync_count} / photon: {photon_count} / pairs: {accepted_count} / "
                f"peak: bin {peak_bin}, {self._format_tdc_time(peak_ns)} ({peak_count}) | "
                f"{pairing_label} | {validity}"
                f"{(': ' + ','.join(invalid_reasons)) if invalid_reasons else ''} | "
                f"FWHM(linear full-res) {self._format_tdc_time(shape.fwhm_ns)} "
                f"half {shape.half_count:.1f} "
                f"sep {self._format_tdc_time(shape.peak_separation_ns) if shape.peak_separation_raw else '-'} | "
                f"ref cleanup {self._format_tdc_time(float(self.tdc_reference_deadtime_ns_spin.value()))} "
                f"filtered {reference_filtered} | "
                f"{diag}{afterpulse_diag}"
            )
        else:
            self._clear_tdc_peak_zoom()
            self._last_tdc_test_label_core = (
                f"{source_tag}sync: {sync_count} / photon: {photon_count} / pairs: {accepted_count} / "
                f"peak: - | {diag}"
            )
        self._set_tdc_test_status_text(self._last_tdc_test_label_core)

    def on_recording_state_changed(self, active: bool, text: str) -> None:
        self.acq_status_label.setText("采集中" if active else "未采集")
        self.record_value_big_label.setText("Recording" if active else "Idle")
        self.record_detail_label.setText(text or "未开始录制")
        if text:
            self.append_log(f"Recording -> {text}")

    def on_offline_snapshot(self, snapshot: HistogramSnapshot) -> None:
        self.offline_status_label.setText("已加载")
        if snapshot.histograms:
            hist = next(iter(snapshot.histograms.values()))
            self.offline_hist_curve.setData(np.arange(len(hist)), np.array(hist))
        else:
            self.offline_hist_curve.setData([], [])
        if snapshot.image_projection:
            self.offline_image_view.setImage(np.array(snapshot.image_projection, dtype=float).T, autoLevels=True)

    def on_set_temperature(self) -> None:
        def action() -> None:
            result = self.controller.apply_temperature_target(self.temp_target_spin.value(), 1500)
            self._show_command_result(result)

        self._guard_action(action)

    def on_apply_analog(self) -> None:
        def action() -> None:
            result = self.controller.apply_analog_outputs(
                self.laser_thr_spin.value(),
                self.pixel_thr_spin.value(),
                self.avalanche_thr_spin.value(),
                self.bias_spin.value(),
                1500,
            )
            self._show_command_result(result)

        self._guard_action(action)

    def on_save_analog_defaults_to_flash(self) -> None:
        def action() -> None:
            result = self.controller.save_analog_defaults_to_flash(
                self.temp_target_spin.value(),
                self.laser_thr_spin.value(),
                self.pixel_thr_spin.value(),
                self.avalanche_thr_spin.value(),
                self.bias_spin.value(),
                1500,
                30000,
            )
            self._show_command_result(result)

        self._guard_action(action)

    def on_apply_gpx2_custom(self) -> None:
        def action() -> None:
            regs = self._collect_gpx2_register_values()
            result = self.controller.configure_gpx2_custom(
                regs,
                timeout_ms=1000,
                profile=int(self.gpx2_profile_combo.currentData()),
                sequence_mode=11,
            )
            self._show_gpx2_command_result(result, "manual write-only 0x80 sequence")
            self.append_log(
                "GPX2 custom registers -> "
                + ", ".join(f"{name}={value}" for name, value in protocol.describe_gpx2_config_bytes(regs).items())
            )

        self._guard_action(action)

    def on_gpx2_registers_changed(self) -> None:
        def action() -> None:
            regs = self._collect_gpx2_register_values()
            self.gpx2_diag_label.setText(
                f"编辑值已同步 | REFCLK_DIV={protocol.gpx2_refclk_divisions(regs)}"
            )

        self._guard_action(action)

    def on_fill_gpx2_from_hex(self) -> None:
        self._guard_action(self._parse_gpx2_custom_config_bytes)

    def on_load_gpx2_defaults(self) -> None:
        self._set_gpx2_register_values(protocol.GPX2_DEFAULT_CONFIG_BYTES)
        self.gpx2_diag_label.setText("已恢复项目默认 GPX2 配置")

    def on_fill_gpx2_lvds_test(self) -> None:
        self._set_gpx2_register_values(protocol.gpx2_lvds_test_config_bytes())
        self.gpx2_diag_label.setText("已填充 LVDS test pattern 配置")

    def on_copy_gpx2_readback(self) -> None:
        if not self._last_gpx2_readback:
            self.append_log("GPX2 copy readback skipped: no config readback available yet.")
            self.gpx2_diag_label.setText("没有可覆盖的回读数据")
            return
        self._set_gpx2_register_values(self._last_gpx2_readback)
        self.gpx2_diag_label.setText("已用最近回读覆盖编辑区")

    def on_run_gpx2_profile_sequence(self) -> None:
        def action() -> None:
            result = self.controller.configure_gpx2(
                profile=int(self.gpx2_profile_combo.currentData()),
                sequence_mode=int(self.gpx2_sequence_combo.currentData()),
            )
            self._show_gpx2_command_result(result, "profile-based sequence")

        self._guard_action(action)

    def on_run_gpx2_custom_sequence(self) -> None:
        def action() -> None:
            regs = self._collect_gpx2_register_values()
            result = self.controller.configure_gpx2_custom(
                regs,
                timeout_ms=1000,
                profile=int(self.gpx2_profile_combo.currentData()),
                sequence_mode=int(self.gpx2_sequence_combo.currentData()),
            )
            self._show_gpx2_command_result(result, "custom register sequence")

        self._guard_action(action)

    def on_gpx2_power_opcode(self) -> None:
        def action() -> None:
            result = self.controller.reset_gpx2_power_opcode(1000)
            self._show_gpx2_command_result(result, "0x30 POWER/reset opcode")

        self._guard_action(action)

    def on_gpx2_init_opcode(self) -> None:
        def action() -> None:
            result = self.controller.configure_gpx2(timeout_ms=1000, profile=0, sequence_mode=4)
            self._show_gpx2_command_result(result, "0x18 INIT opcode")

        self._guard_action(action)

    def on_read_gpx2_config(self) -> None:
        def action() -> None:
            if not self.read_enable_check.isChecked():
                self._gpx2_readback_temp_rx = True
                self.append_log("GPX2 config readback: temporary FT601 RX reader enabled.")
                self._set_rx_reader_checked(True)
            else:
                self._gpx2_readback_temp_rx = False
            self._gpx2_readback_pending = True
            self._gpx2_readback_status_count_at_request = self._current_status_count()
            self._gpx2_readback_timer.start(5000)
            try:
                result = self.controller.read_gpx2_config(1000)
            except Exception:
                self._gpx2_readback_pending = False
                self._gpx2_readback_status_count_at_request = -1
                self._gpx2_readback_timer.stop()
                self._restore_gpx2_temp_rx()
                raise
            self._show_gpx2_command_result(result, "0x40 config readback request")

        self._guard_action(action)

    def on_gpx2_readback_timeout(self) -> None:
        if self._gpx2_readback_pending:
            self._gpx2_readback_pending = False
            self._gpx2_readback_status_count_at_request = -1
            self._restore_gpx2_temp_rx()
            if self._last_gpx2_readback:
                self.gpx2_result_label.setText("GPX2 readback timeout | showing cached snapshot")
                self.gpx2_diag_label.setText("0x40 回读等待超时，当前显示最近一次缓存结果")
                self.append_log(
                    "GPX2 config readback timeout: no fresh STATUS payload captured within 5000 ms; showing cached snapshot."
                )
            else:
                self.gpx2_result_label.setText("GPX2 readback timeout | no STATUS readback available")
                self.gpx2_diag_label.setText("等待 0x40 配置回读超时")
                self.append_log(
                    "GPX2 config readback timeout: no STATUS payload with GPX2 readback captured within 5000 ms."
                )

    def on_add_pixel_row(self) -> None:
        row = self.pixel_table.rowCount()
        self.pixel_table.insertRow(row)
        defaults = ["0", "0", "v1", ""]
        for col, value in enumerate(defaults):
            self.pixel_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))

    def _table_records(self) -> List[PixelParamRecord]:
        records: list[PixelParamRecord] = []
        for row in range(self.pixel_table.rowCount()):
            addr_item = self.pixel_table.item(row, 0)
            value_item = self.pixel_table.item(row, 1)
            version_item = self.pixel_table.item(row, 2)
            comment_item = self.pixel_table.item(row, 3)
            if addr_item is None or value_item is None:
                continue
            records.append(
                PixelParamRecord(
                    addr=int(addr_item.text(), 0),
                    value36=int(value_item.text(), 0),
                    version=version_item.text() if version_item else "v1",
                    comment=comment_item.text() if comment_item else "",
                )
            )
        return records

    def _replace_table_records(self, records: List[PixelParamRecord]) -> None:
        self.pixel_table.setRowCount(0)
        for record in records:
            row = self.pixel_table.rowCount()
            self.pixel_table.insertRow(row)
            self.pixel_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(record.addr)))
            self.pixel_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(record.value36)))
            self.pixel_table.setItem(row, 2, QtWidgets.QTableWidgetItem(record.version))
            self.pixel_table.setItem(row, 3, QtWidgets.QTableWidgetItem(record.comment))
        self.controller.set_pixel_records(records)

    def _import_pixel_records(self, fmt: str) -> None:
        filter_text = "JSON Files (*.json)" if fmt == "json" else "CSV Files (*.csv)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "导入像素阵列", "", filter_text)
        if not path:
            return
        records = self.controller.load_pixel_records(path, fmt)
        self._replace_table_records(records)
        self.append_log(f"Imported pixel records from {path}")

    def _export_pixel_records(self, fmt: str) -> None:
        filter_text = "JSON Files (*.json)" if fmt == "json" else "CSV Files (*.csv)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出像素阵列", "", filter_text)
        if not path:
            return
        records = self._table_records()
        self.controller.save_pixel_records(path, records, fmt)
        self.append_log(f"Exported pixel records to {path}")

    def on_write_selected_pixel_row(self) -> None:
        def action() -> None:
            current = self.pixel_table.currentRow()
            if current < 0:
                return
            records = self._table_records()
            if current >= len(records):
                return
            record = records[current]
            result = self.controller.write_pixel_param(record.addr, record.value36, 2000)
            self._show_command_result(result)

        self._guard_action(action)

    def on_write_all_pixels(self) -> None:
        def action() -> None:
            records = self._table_records()
            results = self.controller.write_all_pixel_records(records, 2000)
            success_count = sum(1 for item in results if item.success)
            self.append_log(f"Pixel bulk write: {success_count}/{len(results)} success")
            if results:
                self._show_command_result(results[-1])

        self._guard_action(action)

    @staticmethod
    def _tdc_upload_payload_word(enable: bool, source: str, start_ch: int, stop_ch: int) -> int:
        mask = (1 << (int(start_ch) - 1)) | (1 << (int(stop_ch) - 1))
        source_l = str(source).lower()
        return (
            (1 if enable else 0)
            | ((1 if source_l == "synthetic" else 0) << 1)
            | ((1 if source_l == "gpx2_spi" else 0) << 2)
            | ((1 if source_l == "gpx2" else 0) << 3)
            | ((mask & 0xF) << 4)
        )

    def on_apply_tdc_test(self) -> None:
        def action() -> None:
            start_ch = int(self.tdc_start_combo.currentData())
            stop_ch = int(self.tdc_stop_combo.currentData())
            if start_ch == stop_ch:
                raise ValueError("Reference and photon channels must be different.")
            source = str(self.tdc_source_combo.currentData())
            pairing_mode = str(self.tdc_pairing_combo.currentData() or "prev_start")
            enabled = True
            self.tdc_test_enable_check.setChecked(True)
            self._tdc_test_upload_payload_word = self._tdc_upload_payload_word(enabled, source, start_ch, stop_ch)
            self._tdc_test_usb_drop_baseline = (
                int(self._last_status_packet.usb_drop_count)
                if self._last_status_packet is not None
                else None
            )
            if self._tdc_test_timer.isActive():
                self._tdc_test_timer.stop()
            if source == "synthetic":
                self.append_log("TCSPC source is FPGA synthetic: external GPX2 inputs and thresholds are ignored.")
            snapshot = self.controller.configure_tdc_test(
                enabled=enabled,
                start_channel_ui=start_ch,
                stop_channel_ui=stop_ch,
                refclk_divisions=self.tdc_refdiv_spin.value(),
                bin_width_raw=self.tdc_bin_width_spin.value(),
                bin_offset=self.tdc_bin_offset_spin.value(),
                bin_count=self.tdc_bin_count_spin.value(),
                acquisition_time_s=self.tdc_acq_time_spin.value(),
                source=source,
                pairing_mode=pairing_mode,
                tdc_deadtime_ps=self.tdc_deadtime_ps_spin.value(),
                reference_deadtime_ns=self.tdc_reference_deadtime_ns_spin.value(),
            )
            self.on_tdc_test_snapshot(snapshot)
            result = self.controller.last_tdc_test_upload_result()
            if result is not None:
                self._show_command_result(result)
                if source == "gpx2_fpga":
                    self.append_log("TCSPC FPGA histogram configured; periodic CMD_TDC_HIST_READOUT enabled.")
                else:
                    self.append_log(
                        "TCSPC CMD_TDC_TEST_UPLOAD payload word: "
                        f"0x{self._tdc_test_upload_payload_word:08X}"
                    )
            duration_ms = int(self.tdc_acq_time_spin.value() * 1000)
            if duration_ms > 0:
                self._tdc_test_timer.start(duration_ms)
                self.append_log(f"TCSPC acquisition timer started: {self.tdc_acq_time_spin.value():.3f} s")
            else:
                self.append_log("TCSPC started in continuous mode (Acq time = 0).")

        self._guard_action(action)

    def on_stop_tdc_test(self) -> None:
        def action() -> None:
            if self._tdc_test_timer.isActive():
                self._tdc_test_timer.stop()
            result = self.controller.stop_tdc_test_upload(1000)
            self.tdc_test_enable_check.setChecked(False)
            self._tdc_test_upload_payload_word = 0
            self._show_command_result(result)
            if self._last_tdc_test_label_core:
                self._set_tdc_test_status_text(self._last_tdc_test_label_core)

        self._guard_action(action)

    def on_clear_tdc_test(self) -> None:
        def action() -> None:
            snapshot = self.controller.clear_tdc_test_histogram()
            self.on_tdc_test_snapshot(snapshot)

        self._guard_action(action)

    def on_start_recording(self) -> None:
        def action() -> None:
            self._update_config_from_controls()
            self.controller.save_current_config()
            session_dir = self.controller.start_recording(
                self.session_name_edit.text().strip() or "capture",
                self.session_notes_edit.text().strip(),
            )
            self.append_log(f"Recording started at {session_dir}")

        self._guard_action(action)

    def on_open_session(self) -> None:
        def action() -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开 Session", "", "Session JSON (*.json)")
            if not path:
                return
            snapshot = self.controller.replay_session(path)
            self.on_offline_snapshot(snapshot)
            self.offline_status_label.setText(Path(path).name)

        self._guard_action(action)

    def on_export_hist_csv(self) -> None:
        def action() -> None:
            snapshot = self.controller.current_snapshot()
            if not snapshot.histograms:
                return
            key = f"{snapshot.current_row},{snapshot.current_col}"
            hist = snapshot.histograms.get(key)
            if hist is None:
                hist = next(iter(snapshot.histograms.values()))
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出 Histogram CSV", "", "CSV Files (*.csv)")
            if not path:
                return
            np.savetxt(path, np.array(hist, dtype=np.uint32), fmt="%d", delimiter=",")
            self.append_log(f"Histogram exported to {path}")

        self._guard_action(action)
