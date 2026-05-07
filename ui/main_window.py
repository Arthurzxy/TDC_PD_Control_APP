from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from app.models import HistogramSnapshot, PixelParamRecord, StatusPacket


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
        self.temp_history: deque[float] = deque(maxlen=120)
        self.cps_history: deque[int] = deque(maxlen=120)
        self.monitor_index: deque[int] = deque(maxlen=120)

        pg.setConfigOptions(antialias=True, background="#0f172a", foreground="#e2e8f0")

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

        self.connection_tab = self._build_connection_tab()
        self.control_tab = self._build_control_tab()
        self.pixel_tab = self._build_pixel_tab()
        self.tdc_test_tab = self._build_tdc_test_tab()
        self.acquisition_tab = self._build_acquisition_tab()
        self.offline_tab = self._build_offline_tab()
        self.log_tab = self._build_log_tab()

        self.tabs.addTab(self.connection_tab, "设备连接")
        self.tabs.addTab(self.control_tab, "FPGA 控制")
        self.tabs.addTab(self.pixel_tab, "像素阵列")
        self.tabs.addTab(self.tdc_test_tab, "TDC Test")
        self.tabs.addTab(self.acquisition_tab, "实时采集")
        self.tabs.addTab(self.offline_tab, "离线分析")
        self.tabs.addTab(self.log_tab, "日志调试")

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
        self.read_enable_check.setChecked(True)

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
        self.rx_rate_label = self._make_stat_pill("RX", "0 B/s")
        self.tx_rate_label = self._make_stat_pill("TX", "0 B/s")
        self.packet_rate_label = self._make_stat_pill("PACKETS", "0 /s")
        self.event_rate_label = self._make_stat_pill("EVENTS", "0 /s")
        stats_grid.addWidget(self.rx_rate_label.parentWidget(), 0, 0)
        stats_grid.addWidget(self.tx_rate_label.parentWidget(), 0, 1)
        stats_grid.addWidget(self.packet_rate_label.parentWidget(), 1, 0)
        stats_grid.addWidget(self.event_rate_label.parentWidget(), 1, 1)
        layout.addLayout(stats_grid)
        layout.addStretch(1)
        return frame

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
        layout.addWidget(analog_group)

        fpga_group = QtWidgets.QGroupBox("核心控制")
        fpga_grid = QtWidgets.QGridLayout(fpga_group)
        fpga_grid.setHorizontalSpacing(12)
        fpga_grid.setVerticalSpacing(12)

        self.gpx2_init_btn = QtWidgets.QPushButton("初始化 GPX2")
        self.flash_save_btn = QtWidgets.QPushButton("FLASH_SAVE")
        self.flash_load_btn = QtWidgets.QPushButton("FLASH_LOAD")

        self.gate_holdoff_spin = QtWidgets.QSpinBox()
        self.gate_holdoff_spin.setRange(0, (1 << 24) - 1)
        self.gate_holdoff_btn = QtWidgets.QPushButton("设置 Holdoff")

        self.gate_div_spin = QtWidgets.QSpinBox()
        self.gate_div_spin.setRange(1, 4095)
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
        for spin in (self.sig2_delay_fine, self.sig3_delay_fine, self.sig2_width_fine, self.sig3_width_fine):
            spin.setRange(0, 31)
        for spin in (self.sig2_width_coarse, self.sig3_width_coarse):
            spin.setRange(0, 7)

        self.sig2_apply_btn = QtWidgets.QPushButton("设置 SIG2")
        self.sig3_apply_btn = QtWidgets.QPushButton("设置 SIG3")
        self.control_result_label = QtWidgets.QLabel("最近命令: -")
        self.control_result_label.setObjectName("InfoText")

        fpga_grid.addWidget(self.gpx2_init_btn, 0, 0)
        fpga_grid.addWidget(self.flash_save_btn, 0, 1)
        fpga_grid.addWidget(self.flash_load_btn, 0, 2)

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

        fpga_grid.addWidget(QtWidgets.QLabel("SIG2 delay_c/f width_c/f"), 5, 0)
        fpga_grid.addWidget(self.sig2_delay_coarse, 5, 1)
        fpga_grid.addWidget(self.sig2_delay_fine, 5, 2)
        fpga_grid.addWidget(self.sig2_width_coarse, 5, 3)
        fpga_grid.addWidget(self.sig2_width_fine, 5, 4)
        fpga_grid.addWidget(self.sig2_apply_btn, 5, 5)

        fpga_grid.addWidget(QtWidgets.QLabel("SIG3 delay_c/f width_c/f"), 6, 0)
        fpga_grid.addWidget(self.sig3_delay_coarse, 6, 1)
        fpga_grid.addWidget(self.sig3_delay_fine, 6, 2)
        fpga_grid.addWidget(self.sig3_width_coarse, 6, 3)
        fpga_grid.addWidget(self.sig3_width_fine, 6, 4)
        fpga_grid.addWidget(self.sig3_apply_btn, 6, 5)
        fpga_grid.addWidget(self.control_result_label, 7, 0, 1, 6)
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

        self.tdc_test_enable_check = QtWidgets.QCheckBox("Enable")
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

        self.tdc_test_apply_btn = QtWidgets.QPushButton("Apply")
        self.tdc_test_clear_btn = QtWidgets.QPushButton("Clear")
        self.tdc_test_clear_btn.setObjectName("Secondary")
        self.tdc_test_status_label = QtWidgets.QLabel("pairs: 0 / peak: -")
        self.tdc_test_status_label.setObjectName("InfoText")

        controls_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        controls_layout.addWidget(self.tdc_test_enable_check, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Start"), 0, 2)
        controls_layout.addWidget(self.tdc_start_combo, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Stop"), 0, 4)
        controls_layout.addWidget(self.tdc_stop_combo, 0, 5)
        controls_layout.addWidget(QtWidgets.QLabel("REF divisions"), 1, 0)
        controls_layout.addWidget(self.tdc_refdiv_spin, 1, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Bin width"), 1, 2)
        controls_layout.addWidget(self.tdc_bin_width_spin, 1, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Offset"), 1, 4)
        controls_layout.addWidget(self.tdc_bin_offset_spin, 1, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Bins"), 2, 0)
        controls_layout.addWidget(self.tdc_bin_count_spin, 2, 1)
        controls_layout.addWidget(self.tdc_test_apply_btn, 2, 3)
        controls_layout.addWidget(self.tdc_test_clear_btn, 2, 4)
        controls_layout.addWidget(self.tdc_test_status_label, 2, 5)
        layout.addWidget(controls_card)

        self.tdc_test_plot = pg.PlotWidget(title="TDC Start/Stop Histogram")
        self._style_plot_widget(self.tdc_test_plot, "dt bin")
        self.tdc_test_plot.getPlotItem().setLabel("left", "Counts")
        self.tdc_test_curve = self.tdc_test_plot.plot(pen=pg.mkPen("#22c55e", width=2))
        layout.addWidget(self._wrap_card("TDC Start/Stop Histogram", self.tdc_test_plot), 1)
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
        self.gpx2_init_btn.clicked.connect(lambda: self._show_command_result(c.configure_gpx2()))
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
        self.tdc_start_combo.setCurrentIndex(max(0, min(3, tdc.start_channel)))
        self.tdc_stop_combo.setCurrentIndex(max(0, min(3, tdc.stop_channel)))
        self.tdc_refdiv_spin.setValue(tdc.refclk_divisions)
        self.tdc_bin_width_spin.setValue(tdc.bin_width_raw)
        self.tdc_bin_offset_spin.setValue(tdc.bin_offset)
        self.tdc_bin_count_spin.setValue(tdc.bin_count)

    def _guard_action(self, func) -> None:
        try:
            func()
        except Exception as exc:
            self.append_log(f"Error: {exc}")
            QtWidgets.QMessageBox.warning(self, "错误", str(exc))

    def _parse_int(self, text: str) -> int:
        return int(text.strip(), 0)

    def _current_device_index(self) -> int:
        data = self.device_combo.currentData()
        return 0 if data is None else int(data)

    def _show_command_result(self, result) -> None:
        status_text = "SENT" if result.success else "FAILED"
        detail = result.message or status_text
        self.control_result_label.setText(
            f"最近命令: cmd=0x{result.cmd_id:02X} status={status_text} detail={detail}"
        )
        self.append_log(self.control_result_label.text())

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
        self.append_log(f"Device state: {text}")

    def on_stats_updated(self, stats, rates) -> None:
        self.rx_rate_label.setText(f"{rates['rx_bytes_per_sec']:.1f} B/s")
        self.tx_rate_label.setText(f"{rates['tx_bytes_per_sec']:.1f} B/s")
        self.packet_rate_label.setText(f"{rates['packets_per_sec']:.1f} /s")
        self.event_rate_label.setText(f"{rates['tdc_events_per_sec']:.1f} /s")

    def on_status_packet(self, status: StatusPacket) -> None:
        decoded = self.controller.decode_status_flags(status.flags)
        temp_c = self._decode_temperature_c(status.temp_avg_raw)
        cps = int(status.counter_1s)

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
        }
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

    def on_tdc_test_snapshot(self, snapshot: HistogramSnapshot) -> None:
        hist = snapshot.histograms.get("tdc_test") if snapshot.histograms else None
        if hist is None:
            self.tdc_test_curve.setData([], [])
            self.tdc_test_status_label.setText("pairs: 0 / peak: -")
            return
        values = np.array(hist, dtype=np.uint32)
        self.tdc_test_curve.setData(np.arange(len(values)), values)
        total = int(values.sum())
        if total > 0:
            peak_bin = int(values.argmax())
            peak_count = int(values[peak_bin])
            self.tdc_test_status_label.setText(f"pairs: {total} / peak: bin {peak_bin} ({peak_count})")
        else:
            self.tdc_test_status_label.setText("pairs: 0 / peak: -")

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

    def on_apply_tdc_test(self) -> None:
        def action() -> None:
            start_ch = int(self.tdc_start_combo.currentData())
            stop_ch = int(self.tdc_stop_combo.currentData())
            if start_ch == stop_ch:
                raise ValueError("Start and stop channels must be different.")
            snapshot = self.controller.configure_tdc_test(
                enabled=self.tdc_test_enable_check.isChecked(),
                start_channel_ui=start_ch,
                stop_channel_ui=stop_ch,
                refclk_divisions=self.tdc_refdiv_spin.value(),
                bin_width_raw=self.tdc_bin_width_spin.value(),
                bin_offset=self.tdc_bin_offset_spin.value(),
                bin_count=self.tdc_bin_count_spin.value(),
            )
            self.on_tdc_test_snapshot(snapshot)

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
