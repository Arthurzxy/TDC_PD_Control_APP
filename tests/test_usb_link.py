from __future__ import annotations

import os
import unittest
from unittest import mock

from PyQt5 import QtCore

from app.protocol import build_test_tdc_raw_packet
from app.usb_link import DeviceService


class DeviceServiceTests(unittest.TestCase):
    def test_set_read_enabled_restarts_stopped_rx_worker(self) -> None:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        self.assertIsNotNone(app)

        class FakeSignal:
            def connect(self, _slot) -> None:
                pass

        class StoppedWorker:
            def __init__(self) -> None:
                self.wait_timeout = None

            def isRunning(self) -> bool:
                return False

            def wait(self, timeout_ms: int) -> None:
                self.wait_timeout = timeout_ms

        class FakeRxWorker:
            def __init__(self, *_args, **_kwargs) -> None:
                self.parsed_received = FakeSignal()
                self.io_error = FakeSignal()
                self.started = False

            def set_capture_raw_bytes(self, _enabled: bool) -> None:
                pass

            def set_include_packet_objects(self, _enabled: bool) -> None:
                pass

            def start(self) -> None:
                self.started = True

            def isRunning(self) -> bool:
                return self.started

        service = DeviceService()
        stopped = StoppedWorker()
        service.device = object()
        service.rx_worker = stopped

        with mock.patch("app.usb_link.RxWorker", FakeRxWorker):
            service.set_read_enabled(True)

        self.assertEqual(stopped.wait_timeout, 1000)
        self.assertIsInstance(service.rx_worker, FakeRxWorker)
        self.assertTrue(service.rx_worker.started)
        self.assertTrue(service.read_enabled)
        self.assertTrue(service.is_rx_reader_running())

    def test_tdc_raw_batch_does_not_emit_legacy_event_signal_by_default(self) -> None:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        self.assertIsNotNone(app)

        service = DeviceService()
        batches = []
        legacy_event_lists = []
        service.tdc_raw_batch_received.connect(batches.append)
        service.tdc_events_received.connect(legacy_event_lists.append)

        service._handle_raw_bytes(build_test_tdc_raw_packet())

        self.assertEqual([len(batch) for batch in batches], [2])
        self.assertEqual(legacy_event_lists, [])
        self.assertEqual(service.runtime_stats.tdc_event_count, 2)
        self.assertEqual(service.runtime_stats.tdc_raw_packet_count, 1)
        self.assertEqual(service.runtime_stats.tdc_raw_event_count, 2)
        self.assertEqual(service.runtime_stats.last_tdc_raw_channels, (1, 1, 0, 0))
        self.assertGreaterEqual(service.runtime_stats.last_tdc_raw_packet_age_ms, 0.0)

    def test_tdc_raw_batch_can_emit_legacy_event_signal_when_enabled(self) -> None:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        self.assertIsNotNone(app)

        service = DeviceService()
        service.set_emit_legacy_raw_events(True)
        legacy_event_lists = []
        service.tdc_events_received.connect(legacy_event_lists.append)

        service._handle_raw_bytes(build_test_tdc_raw_packet())

        self.assertEqual([len(events) for events in legacy_event_lists], [2])
        self.assertEqual([event.channel for event in legacy_event_lists[0]], [0, 1])

    def test_tdc_raw_fast_path_skips_high_rate_debug_signals_by_default(self) -> None:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        self.assertIsNotNone(app)

        service = DeviceService()
        raw_blocks = []
        packets = []
        batches = []
        service.raw_bytes_received.connect(raw_blocks.append)
        service.packet_received.connect(packets.append)
        service.tdc_raw_batch_received.connect(batches.append)

        for _ in range(8):
            service._handle_raw_bytes(build_test_tdc_raw_packet())

        self.assertEqual(raw_blocks, [])
        self.assertEqual(packets, [])
        self.assertEqual([len(batch) for batch in batches], [2] * 8)
        self.assertEqual(service.runtime_stats.tdc_raw_packet_count, 8)

    def test_open_device_keeps_configured_read_block_size(self) -> None:
        app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
        self.assertIsNotNone(app)

        class FakeTransport:
            read_pipe = 0x82
            write_pipe = 0x02

            def open_device(self, index: int) -> None:
                self.index = index

            def close_device(self) -> None:
                pass

            def set_suspend_timeout(self, value: int) -> None:
                self.suspend_timeout = value

            def get_suspend_timeout(self) -> int:
                return getattr(self, "suspend_timeout", 0)

            def set_pipe_timeout(self, pipe_id: int, timeout_ms: int) -> None:
                self.timeout = (pipe_id, timeout_ms)

            def get_pipe_timeout(self, pipe_id: int) -> int:
                return getattr(self, "timeout", (pipe_id, 0))[1]

            def flush_pipe(self, pipe_id: int) -> None:
                self.flushed = pipe_id

        service = DeviceService()
        service._transport_candidates = lambda: [("fake", FakeTransport)]
        service._log_device_usb_diagnostics = lambda: None
        try:
            service.open_device(read_block_size=256 * 1024, read_enabled=False)
            self.assertEqual(service.read_block_size, 256 * 1024)
            self.assertFalse(service.read_enabled)
        finally:
            service.close_device()

    def test_transport_candidates_prefer_official_pyd3xx(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FT601_BACKEND", None)
            candidates = DeviceService._transport_candidates()

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "pyd3xx")

    def test_transport_candidates_can_force_dll_backend(self) -> None:
        with mock.patch.dict(os.environ, {"FT601_BACKEND": "d3xx_dll"}):
            candidates = DeviceService._transport_candidates()

        if os.name == "nt":
            self.assertEqual([name for name, _ in candidates], ["d3xx_dll"])
        else:
            self.assertEqual(candidates[0][0], "pyd3xx")


if __name__ == "__main__":
    unittest.main()
