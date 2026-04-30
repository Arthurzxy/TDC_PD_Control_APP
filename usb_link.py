from __future__ import annotations

"""Runtime USB chain: FT_Create -> FT_ReadPipe / FT_WritePipe -> PacketParser -> STATUS/TDC fan-out."""

import ctypes
import os
import queue
import threading
import time
from collections import defaultdict, deque
from ctypes import byref, c_int, c_uint, c_ulong, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore

import app.protocol as protocol
from app.models import CommandHistoryItem, CommandResult, RuntimeStats, StatusFlagsDecoded, StatusPacket
from app.protocol import CommandEncoder, PacketParser
from app.utils.logging_utils import RateMeter


FT_OK = 0
FT_OPEN_BY_INDEX = 0x00000010
DEFAULT_READ_PIPE = 0x82
DEFAULT_WRITE_PIPE = 0x02
FT_SERIAL_NUMBER_SIZE = 16
FT_DESCRIPTION_SIZE = 32

FT_STATUS_NAMES = {
    0: "FT_OK",
    1: "FT_INVALID_HANDLE",
    2: "FT_DEVICE_NOT_FOUND",
    3: "FT_DEVICE_NOT_OPENED",
    4: "FT_IO_ERROR",
    5: "FT_INSUFFICIENT_RESOURCES",
    6: "FT_INVALID_PARAMETER",
    7: "FT_INVALID_BAUD_RATE",
    8: "FT_DEVICE_NOT_OPENED_FOR_ERASE",
    9: "FT_DEVICE_NOT_OPENED_FOR_WRITE",
    10: "FT_FAILED_TO_WRITE_DEVICE",
    11: "FT_EEPROM_READ_FAILED",
    12: "FT_EEPROM_WRITE_FAILED",
    13: "FT_EEPROM_ERASE_FAILED",
    14: "FT_EEPROM_NOT_PRESENT",
    15: "FT_EEPROM_NOT_PROGRAMMED",
    16: "FT_INVALID_ARGS",
    17: "FT_NOT_SUPPORTED",
    18: "FT_NO_MORE_ITEMS",
    19: "FT_TIMEOUT",
    20: "FT_OPERATION_ABORTED",
    21: "FT_RESERVED_PIPE",
    22: "FT_INVALID_CONTROL_REQUEST_DIRECTION",
    23: "FT_INVALID_CONTROL_REQUEST_TYPE",
    24: "FT_IO_PENDING",
    25: "FT_IO_INCOMPLETE",
    26: "FT_HANDLE_EOF",
    27: "FT_BUSY",
    28: "FT_NO_SYSTEM_RESOURCES",
    29: "FT_DEVICE_LIST_NOT_READY",
    30: "FT_DEVICE_NOT_CONNECTED",
    31: "FT_INCORRECT_DEVICE_PATH",
    32: "FT_OTHER_ERROR",
}

PIPE_TYPE_NAMES = {
    0: "CONTROL",
    1: "ISOCHRONOUS",
    2: "BULK",
    3: "INTERRUPT",
}

FIFO_CLOCK_NAMES = {
    0: "100MHz",
    1: "66MHz",
}

FIFO_MODE_NAMES = {
    0: "245",
    1: "600",
}

CHANNEL_CONFIG_NAMES = {
    0: "4CH",
    1: "2CH",
    2: "1CH",
    3: "1CH_OUT",
    4: "1CH_IN",
}


class D3XXError(RuntimeError):
    pass


@dataclass
class DeviceInfo:
    index: int
    description: str = ""
    serial: str = ""


class FTInterfaceDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_ubyte),
        ("bDescriptorType", ctypes.c_ubyte),
        ("bInterfaceNumber", ctypes.c_ubyte),
        ("bAlternateSetting", ctypes.c_ubyte),
        ("bNumEndpoints", ctypes.c_ubyte),
        ("bInterfaceClass", ctypes.c_ubyte),
        ("bInterfaceSubClass", ctypes.c_ubyte),
        ("bInterfaceProtocol", ctypes.c_ubyte),
        ("iInterface", ctypes.c_ubyte),
    ]


class FTConfigurationDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_ubyte),
        ("bDescriptorType", ctypes.c_ubyte),
        ("wTotalLength", ctypes.c_ushort),
        ("bNumInterfaces", ctypes.c_ubyte),
        ("bConfigurationValue", ctypes.c_ubyte),
        ("iConfiguration", ctypes.c_ubyte),
        ("bmAttributes", ctypes.c_ubyte),
        ("MaxPower", ctypes.c_ubyte),
    ]


class FTPipeInformation(ctypes.Structure):
    _fields_ = [
        ("PipeType", c_ulong),
        ("PipeId", ctypes.c_ubyte),
        ("MaximumPacketSize", ctypes.c_ushort),
        ("Interval", ctypes.c_ubyte),
    ]


class FT60XConfiguration(ctypes.Structure):
    _fields_ = [
        ("VendorID", ctypes.c_ushort),
        ("ProductID", ctypes.c_ushort),
        ("StringDescriptors", ctypes.c_ubyte * 128),
        ("bInterval", ctypes.c_ubyte),
        ("PowerAttributes", ctypes.c_ubyte),
        ("PowerConsumption", ctypes.c_ushort),
        ("Reserved2", ctypes.c_ubyte),
        ("FIFOClock", ctypes.c_ubyte),
        ("FIFOMode", ctypes.c_ubyte),
        ("ChannelConfig", ctypes.c_ubyte),
        ("OptionalFeatureSupport", ctypes.c_ushort),
        ("BatteryChargingGPIOConfig", ctypes.c_ubyte),
        ("FlashEEPROMDetection", ctypes.c_ubyte),
        ("MSIO_Control", c_ulong),
        ("GPIO_Control", c_ulong),
    ]


class D3XXDevice:
    """Small ctypes wrapper around the FTDI D3XX API."""

    def __init__(self, dll_name: str = "FTD3XXWU.dll") -> None:
        self._dll_name = dll_name
        self._dll_path = self._resolve_dll_path(dll_name)
        self._dll = None
        self._dll_directory_handle = None
        self._handle = c_void_p()
        self._io_lock = threading.Lock()
        self.read_pipe = DEFAULT_READ_PIPE
        self.write_pipe = DEFAULT_WRITE_PIPE

    @staticmethod
    def _resolve_dll_path(dll_name: str) -> str:
        candidate = Path(dll_name)
        if candidate.is_absolute():
            return str(candidate)

        # 优先使用 app 目录内随工程分发的 DLL，避免系统 PATH 中残留旧版
        # FTD3XX DLL 导致运行时枚举或收发行为不一致。
        local_candidate = Path(__file__).resolve().with_name("FTD3XXWU.dll")
        if local_candidate.exists():
            return str(local_candidate)

        return dll_name

    def load(self) -> None:
        if self._dll is None:
            try:
                dll_path = Path(self._dll_path)
                if dll_path.is_absolute() and hasattr(os, "add_dll_directory"):
                    self._dll_directory_handle = os.add_dll_directory(str(dll_path.parent))
                self._dll = ctypes.WinDLL(self._dll_path)
            except OSError as exc:
                raise D3XXError(f"Failed to load FTD3XX DLL from '{self._dll_path}': {exc}") from exc
            self._bind_functions()

    def _bind_functions(self) -> None:
        self._dll.FT_CreateDeviceInfoList.argtypes = [ctypes.POINTER(c_ulong)]
        self._dll.FT_CreateDeviceInfoList.restype = c_ulong
        self._dll.FT_GetDeviceInfoDetail.argtypes = [
            c_ulong,
            ctypes.POINTER(c_ulong),
            ctypes.POINTER(c_ulong),
            ctypes.POINTER(c_ulong),
            ctypes.POINTER(c_ulong),
            c_void_p,
            c_void_p,
            ctypes.POINTER(c_void_p),
        ]
        self._dll.FT_GetDeviceInfoDetail.restype = c_ulong
        self._dll.FT_Create.argtypes = [c_void_p, c_ulong, ctypes.POINTER(c_void_p)]
        self._dll.FT_Create.restype = c_ulong
        self._dll.FT_GetInterfaceDescriptor.argtypes = [c_void_p, ctypes.c_ubyte, ctypes.POINTER(FTInterfaceDescriptor)]
        self._dll.FT_GetInterfaceDescriptor.restype = c_ulong
        self._dll.FT_GetPipeInformation.argtypes = [
            c_void_p,
            ctypes.c_ubyte,
            ctypes.c_ubyte,
            ctypes.POINTER(FTPipeInformation),
        ]
        self._dll.FT_GetPipeInformation.restype = c_ulong
        self._dll.FT_GetConfigurationDescriptor.argtypes = [
            c_void_p,
            ctypes.POINTER(FTConfigurationDescriptor),
        ]
        self._dll.FT_GetConfigurationDescriptor.restype = c_ulong
        self._dll.FT_GetChipConfiguration.argtypes = [c_void_p, ctypes.POINTER(FT60XConfiguration)]
        self._dll.FT_GetChipConfiguration.restype = c_ulong
        self._dll.FT_Close.argtypes = [c_void_p]
        self._dll.FT_Close.restype = c_ulong
        self._dll.FT_FlushPipe.argtypes = [c_void_p, ctypes.c_ubyte]
        self._dll.FT_FlushPipe.restype = c_ulong
        self._dll.FT_ReadPipe.argtypes = [c_void_p, c_uint, c_void_p, c_ulong, ctypes.POINTER(c_ulong), c_void_p]
        self._dll.FT_ReadPipe.restype = c_ulong
        self._dll.FT_WritePipe.argtypes = [c_void_p, c_uint, c_void_p, c_ulong, ctypes.POINTER(c_ulong), c_void_p]
        self._dll.FT_WritePipe.restype = c_ulong
        self._dll.FT_SetPipeTimeout.argtypes = [c_void_p, ctypes.c_ubyte, c_ulong]
        self._dll.FT_SetPipeTimeout.restype = c_ulong
        self._dll.FT_GetPipeTimeout.argtypes = [c_void_p, ctypes.c_ubyte, ctypes.POINTER(c_ulong)]
        self._dll.FT_GetPipeTimeout.restype = c_ulong
        if hasattr(self._dll, "FT_SetSuspendTimeout"):
            self._dll.FT_SetSuspendTimeout.argtypes = [c_void_p, c_ulong]
            self._dll.FT_SetSuspendTimeout.restype = c_ulong
        if hasattr(self._dll, "FT_GetSuspendTimeout"):
            self._dll.FT_GetSuspendTimeout.argtypes = [c_void_p, ctypes.POINTER(c_ulong)]
            self._dll.FT_GetSuspendTimeout.restype = c_ulong

    @staticmethod
    def _status_text(status: int) -> str:
        return FT_STATUS_NAMES.get(int(status), f"FT_STATUS_{int(status)}")

    @staticmethod
    def is_timeout_exception(exc: Exception) -> bool:
        return "FT_TIMEOUT" in str(exc)

    @staticmethod
    def is_read_idle_exception(exc: Exception) -> bool:
        text = str(exc)
        return "FT_TIMEOUT" in text or "FT_OTHER_ERROR" in text

    @staticmethod
    def _decode_c_string(raw: bytes) -> str:
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    def open_device(self, index: int = 0) -> None:
        self.load()
        status = self._dll.FT_Create(c_void_p(int(index)), FT_OPEN_BY_INDEX, byref(self._handle))
        if status != FT_OK:
            raise D3XXError(f"FT_Create failed with status {status} ({self._status_text(status)})")

    def enumerate_devices(self, max_devices: int = 8) -> List[DeviceInfo]:
        self.load()
        count = c_ulong()
        status = self._dll.FT_CreateDeviceInfoList(byref(count))
        if status != FT_OK:
            raise D3XXError(
                f"FT_CreateDeviceInfoList failed with status {status} ({self._status_text(status)})"
            )

        devices: List[DeviceInfo] = []
        for index in range(min(int(count.value), max_devices)):
            flags = c_ulong()
            dev_type = c_ulong()
            dev_id = c_ulong()
            loc_id = c_ulong()
            handle = c_void_p()
            serial = ctypes.create_string_buffer(FT_SERIAL_NUMBER_SIZE)
            description = ctypes.create_string_buffer(FT_DESCRIPTION_SIZE)
            status = self._dll.FT_GetDeviceInfoDetail(
                c_ulong(index),
                byref(flags),
                byref(dev_type),
                byref(dev_id),
                byref(loc_id),
                serial,
                description,
                byref(handle),
            )
            if status == FT_OK:
                devices.append(
                    DeviceInfo(
                        index=index,
                        description=self._decode_c_string(description.raw) or f"FT60x Device {index}",
                        serial=self._decode_c_string(serial.raw),
                    )
                )
        return devices

    def close_device(self) -> None:
        if self._dll is None or not self._handle:
            return
        status = self._dll.FT_Close(self._handle)
        if status != FT_OK:
            raise D3XXError(f"FT_Close failed with status {status} ({self._status_text(status)})")
        self._handle = c_void_p()

    def get_interface_descriptor(self, interface_index: int = 0) -> dict:
        if not self._handle:
            raise D3XXError("Device is not open")
        descriptor = FTInterfaceDescriptor()
        status = self._dll.FT_GetInterfaceDescriptor(self._handle, ctypes.c_ubyte(interface_index), byref(descriptor))
        if status != FT_OK:
            raise D3XXError(
                f"FT_GetInterfaceDescriptor failed with status {status} ({self._status_text(status)})"
            )
        return {
            "interface_index": int(interface_index),
            "interface_number": int(descriptor.bInterfaceNumber),
            "alternate_setting": int(descriptor.bAlternateSetting),
            "num_endpoints": int(descriptor.bNumEndpoints),
            "interface_class": int(descriptor.bInterfaceClass),
            "interface_subclass": int(descriptor.bInterfaceSubClass),
            "interface_protocol": int(descriptor.bInterfaceProtocol),
        }

    def get_pipe_information(self, interface_index: int, pipe_index: int) -> dict:
        if not self._handle:
            raise D3XXError("Device is not open")
        pipe_info = FTPipeInformation()
        status = self._dll.FT_GetPipeInformation(
            self._handle,
            ctypes.c_ubyte(interface_index),
            ctypes.c_ubyte(pipe_index),
            byref(pipe_info),
        )
        if status != FT_OK:
            raise D3XXError(
                f"FT_GetPipeInformation failed with status {status} ({self._status_text(status)})"
            )
        return {
            "interface_index": int(interface_index),
            "pipe_index": int(pipe_index),
            "pipe_type": int(pipe_info.PipeType),
            "pipe_id": int(pipe_info.PipeId),
            "max_packet_size": int(pipe_info.MaximumPacketSize),
            "interval": int(pipe_info.Interval),
        }

    def list_pipe_information(self, interface_index: int = 0) -> list[dict]:
        descriptor = self.get_interface_descriptor(interface_index)
        pipes: list[dict] = []
        for pipe_index in range(descriptor["num_endpoints"]):
            try:
                pipes.append(self.get_pipe_information(interface_index, pipe_index))
            except D3XXError:
                break
        return pipes

    def get_configuration_descriptor(self) -> dict:
        if not self._handle:
            raise D3XXError("Device is not open")
        descriptor = FTConfigurationDescriptor()
        status = self._dll.FT_GetConfigurationDescriptor(self._handle, byref(descriptor))
        if status != FT_OK:
            raise D3XXError(
                f"FT_GetConfigurationDescriptor failed with status {status} ({self._status_text(status)})"
            )
        return {
            "total_length": int(descriptor.wTotalLength),
            "num_interfaces": int(descriptor.bNumInterfaces),
            "configuration_value": int(descriptor.bConfigurationValue),
            "attributes": int(descriptor.bmAttributes),
            "max_power": int(descriptor.MaxPower),
        }

    def get_chip_configuration(self) -> dict:
        if not self._handle:
            raise D3XXError("Device is not open")
        config = FT60XConfiguration()
        status = self._dll.FT_GetChipConfiguration(self._handle, byref(config))
        if status != FT_OK:
            raise D3XXError(
                f"FT_GetChipConfiguration failed with status {status} ({self._status_text(status)})"
            )
        return {
            "vendor_id": int(config.VendorID),
            "product_id": int(config.ProductID),
            "fifo_clock": int(config.FIFOClock),
            "fifo_mode": int(config.FIFOMode),
            "channel_config": int(config.ChannelConfig),
            "optional_feature_support": int(config.OptionalFeatureSupport),
            "battery_charging_gpio_config": int(config.BatteryChargingGPIOConfig),
            "flash_eeprom_detection": int(config.FlashEEPROMDetection),
            "msio_control": int(config.MSIO_Control),
            "gpio_control": int(config.GPIO_Control),
        }

    def set_pipe_timeout(self, pipe_id: int, timeout_ms: int) -> None:
        if not self._handle:
            raise D3XXError("Device is not open")
        status = self._dll.FT_SetPipeTimeout(self._handle, ctypes.c_ubyte(pipe_id), c_ulong(timeout_ms))
        if status != FT_OK:
            raise D3XXError(f"FT_SetPipeTimeout failed with status {status} ({self._status_text(status)})")

    def get_pipe_timeout(self, pipe_id: int) -> int:
        if not self._handle:
            raise D3XXError("Device is not open")
        timeout_ms = c_ulong()
        status = self._dll.FT_GetPipeTimeout(self._handle, ctypes.c_ubyte(pipe_id), byref(timeout_ms))
        if status != FT_OK:
            raise D3XXError(f"FT_GetPipeTimeout failed with status {status} ({self._status_text(status)})")
        return int(timeout_ms.value)

    def set_suspend_timeout(self, timeout_ms: int) -> None:
        if not self._handle:
            raise D3XXError("Device is not open")
        if not hasattr(self._dll, "FT_SetSuspendTimeout"):
            raise D3XXError("FT_SetSuspendTimeout is not available in this D3XX DLL")
        status = self._dll.FT_SetSuspendTimeout(self._handle, c_ulong(int(timeout_ms)))
        if status != FT_OK:
            raise D3XXError(f"FT_SetSuspendTimeout failed with status {status} ({self._status_text(status)})")

    def get_suspend_timeout(self) -> int:
        if not self._handle:
            raise D3XXError("Device is not open")
        if not hasattr(self._dll, "FT_GetSuspendTimeout"):
            raise D3XXError("FT_GetSuspendTimeout is not available in this D3XX DLL")
        timeout_ms = c_ulong()
        status = self._dll.FT_GetSuspendTimeout(self._handle, byref(timeout_ms))
        if status != FT_OK:
            raise D3XXError(f"FT_GetSuspendTimeout failed with status {status} ({self._status_text(status)})")
        return int(timeout_ms.value)

    def flush_pipe(self, pipe_id: int) -> None:
        if not self._handle:
            raise D3XXError("Device is not open")
        status = self._dll.FT_FlushPipe(self._handle, ctypes.c_ubyte(pipe_id))
        if status != FT_OK:
            raise D3XXError(f"FT_FlushPipe failed with status {status} ({self._status_text(status)})")

    def read_block(self, size: int, pipe_id: Optional[int] = None) -> bytes:
        if not self._handle:
            raise D3XXError("Device is not open")
        pipe = self.read_pipe if pipe_id is None else pipe_id
        buffer = (ctypes.c_ubyte * size)()
        transferred = c_ulong()
        with self._io_lock:
            status = self._dll.FT_ReadPipe(self._handle, pipe, buffer, size, byref(transferred), None)
        if status != FT_OK:
            raise D3XXError(f"FT_ReadPipe failed with status {status} ({self._status_text(status)})")
        return bytes(buffer[: transferred.value])

    def write_block(self, bytes_data: bytes, pipe_id: Optional[int] = None) -> int:
        if not self._handle:
            raise D3XXError("Device is not open")
        pipe = self.write_pipe if pipe_id is None else pipe_id
        raw = (ctypes.c_ubyte * len(bytes_data)).from_buffer_copy(bytes_data)
        transferred = c_ulong()
        with self._io_lock:
            status = self._dll.FT_WritePipe(self._handle, pipe, raw, len(bytes_data), byref(transferred), None)
        if status != FT_OK:
            raise D3XXError(f"FT_WritePipe failed with status {status} ({self._status_text(status)})")
        if int(transferred.value) != len(bytes_data):
            raise D3XXError(
                f"FT_WritePipe short write: requested {len(bytes_data)} byte(s), "
                f"transferred {int(transferred.value)} byte(s)"
            )
        return int(transferred.value)


class PyD3XXDevice:
    """FT601 transport using the PyD3XX demo flow exactly: open, get pipes, read/write pipes."""

    def __init__(self) -> None:
        self._pyd3xx = None
        self._device = None
        self._io_lock = threading.Lock()
        self._pipes_by_id: dict[int, object] = {}
        self._pipe_meta_by_id: dict[int, dict] = {}
        self._overlaps_by_id: dict[int, object] = {}
        self._pipe_timeout_ms: dict[int, int] = {}
        self.read_pipe = DEFAULT_READ_PIPE
        self.write_pipe = DEFAULT_WRITE_PIPE

    def load(self) -> None:
        if self._pyd3xx is not None:
            return
        try:
            import PyD3XX  # type: ignore[import-not-found]
        except ImportError as exc:
            raise D3XXError(
                "PyD3XX is not installed. Install it with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        self._pyd3xx = PyD3XX
        if hasattr(PyD3XX, "SetPrintLevel") and hasattr(PyD3XX, "PRINT_NONE"):
            PyD3XX.SetPrintLevel(PyD3XX.PRINT_NONE)

    def _ok(self) -> int:
        self.load()
        return int(getattr(self._pyd3xx, "FT_OK", FT_OK))

    def _status_value(self, name: str, fallback: int) -> int:
        self.load()
        return int(getattr(self._pyd3xx, name, fallback))

    def _platform(self) -> str:
        self.load()
        return str(getattr(self._pyd3xx, "Platform", "")).lower()

    def _status_text(self, status: int) -> str:
        self.load()
        status = int(status)
        status_names = getattr(self._pyd3xx, "FT_STATUS_STR", None)
        if isinstance(status_names, dict):
            return str(status_names.get(status, FT_STATUS_NAMES.get(status, f"FT_STATUS_{status}")))
        if isinstance(status_names, (list, tuple)) and 0 <= status < len(status_names):
            return str(status_names[status])
        return FT_STATUS_NAMES.get(status, f"FT_STATUS_{status}")

    @staticmethod
    def is_timeout_exception(exc: Exception) -> bool:
        return "FT_TIMEOUT" in str(exc)

    @staticmethod
    def is_read_idle_exception(exc: Exception) -> bool:
        text = str(exc)
        return "FT_TIMEOUT" in text or "FT_OTHER_ERROR" in text or "FT_IO_INCOMPLETE" in text

    @staticmethod
    def _field(obj: object, *names: str, default=None):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if isinstance(value, bytes):
                    return value.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
                return value
        return default

    def open_device(self, index: int = 0) -> None:
        self.load()
        status, count = self._pyd3xx.FT_CreateDeviceInfoList()
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_CreateDeviceInfoList failed with status {status} ({self._status_text(status)})"
            )
        if int(count) <= int(index):
            raise D3XXError(f"FT601 device index {index} not found; detected {int(count)} device(s)")

        status, device = self._pyd3xx.FT_GetDeviceInfoDetail(int(index))
        if int(status) != self._ok():
            raise D3XXError(f"FT_GetDeviceInfoDetail failed with status {status} ({self._status_text(status)})")

        status = self._pyd3xx.FT_Create(int(index), self._pyd3xx.FT_OPEN_BY_INDEX, device)
        if int(status) != self._ok():
            raise D3XXError(f"FT_Create failed with status {status} ({self._status_text(status)})")

        self._device = device
        self._refresh_pipe_cache()

    def enumerate_devices(self, max_devices: int = 8) -> List[DeviceInfo]:
        self.load()
        status, count = self._pyd3xx.FT_CreateDeviceInfoList()
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_CreateDeviceInfoList failed with status {status} ({self._status_text(status)})"
            )

        devices: List[DeviceInfo] = []
        for index in range(min(int(count), max_devices)):
            status, device = self._pyd3xx.FT_GetDeviceInfoDetail(index)
            if int(status) == self._ok():
                description = self._field(device, "Description", "description", default="")
                serial = self._field(device, "SerialNumber", "Serial", "serial", default="")
                devices.append(
                    DeviceInfo(
                        index=index,
                        description=str(description or f"FT60x Device {index}"),
                        serial=str(serial or ""),
                    )
                )
        return devices

    def close_device(self) -> None:
        if self._pyd3xx is None or self._device is None:
            return
        self._release_overlaps()
        status = self._pyd3xx.FT_Close(self._device)
        if int(status) != self._ok():
            raise D3XXError(f"FT_Close failed with status {status} ({self._status_text(status)})")
        self._device = None
        self._pipes_by_id.clear()
        self._pipe_meta_by_id.clear()
        self._overlaps_by_id.clear()
        self._pipe_timeout_ms.clear()

    def get_interface_descriptor(self, interface_index: int = 0) -> dict:
        if self._device is None:
            raise D3XXError("Device is not open")
        status, descriptor = self._pyd3xx.FT_GetInterfaceDescriptor(self._device, int(interface_index))
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_GetInterfaceDescriptor failed with status {status} ({self._status_text(status)})"
            )
        return {
            "interface_index": int(interface_index),
            "interface_number": int(self._field(descriptor, "bInterfaceNumber", default=0)),
            "alternate_setting": int(self._field(descriptor, "bAlternateSetting", default=0)),
            "num_endpoints": int(self._field(descriptor, "bNumEndpoints", default=0)),
            "interface_class": int(self._field(descriptor, "bInterfaceClass", default=0)),
            "interface_subclass": int(self._field(descriptor, "bInterfaceSubClass", default=0)),
            "interface_protocol": int(self._field(descriptor, "bInterfaceProtocol", default=0)),
        }

    def _pipe_to_dict(self, interface_index: int, pipe_index: int, pipe: object) -> dict:
        pipe_id = int(self._field(pipe, "PipeId", "PipeID", "pipe_id", default=pipe_index))
        return {
            "interface_index": int(interface_index),
            "pipe_index": int(pipe_index),
            "pipe_type": int(self._field(pipe, "PipeType", "pipe_type", default=0)),
            "pipe_id": pipe_id,
            "max_packet_size": int(self._field(pipe, "MaximumPacketSize", "MaxPacketSize", default=0)),
            "interval": int(self._field(pipe, "Interval", "interval", default=0)),
        }

    def get_pipe_information(self, interface_index: int, pipe_index: int) -> dict:
        if self._device is None:
            raise D3XXError("Device is not open")
        status, pipe = self._pyd3xx.FT_GetPipeInformation(self._device, int(interface_index), int(pipe_index))
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_GetPipeInformation failed with status {status} ({self._status_text(status)})"
            )
        pipe_info = self._pipe_to_dict(interface_index, pipe_index, pipe)
        self._pipes_by_id[pipe_info["pipe_id"]] = pipe
        self._pipe_meta_by_id[pipe_info["pipe_id"]] = pipe_info
        if pipe_info["pipe_id"] not in self._overlaps_by_id:
            status, overlap = self._pyd3xx.FT_InitializeOverlapped(self._device)
            if int(status) != self._ok():
                raise D3XXError(
                    f"FT_InitializeOverlapped failed for pipe 0x{pipe_info['pipe_id']:02X} "
                    f"with status {status} ({self._status_text(status)})"
                )
            self._overlaps_by_id[pipe_info["pipe_id"]] = overlap
        return pipe_info

    def list_pipe_information(self, interface_index: int = 0) -> list[dict]:
        descriptor = self.get_interface_descriptor(interface_index)
        pipes: list[dict] = []
        for pipe_index in range(descriptor["num_endpoints"]):
            try:
                pipes.append(self.get_pipe_information(interface_index, pipe_index))
            except D3XXError:
                break
        return pipes

    def get_configuration_descriptor(self) -> dict:
        if self._device is None:
            raise D3XXError("Device is not open")
        status, descriptor = self._pyd3xx.FT_GetConfigurationDescriptor(self._device)
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_GetConfigurationDescriptor failed with status {status} ({self._status_text(status)})"
            )
        return {
            "total_length": int(self._field(descriptor, "wTotalLength", default=0)),
            "num_interfaces": int(self._field(descriptor, "bNumInterfaces", default=0)),
            "configuration_value": int(self._field(descriptor, "bConfigurationValue", default=0)),
            "attributes": int(self._field(descriptor, "bmAttributes", default=0)),
            "max_power": int(self._field(descriptor, "MaxPower", default=0)),
        }

    def get_chip_configuration(self) -> dict:
        if self._device is None:
            raise D3XXError("Device is not open")
        status, config = self._pyd3xx.FT_GetChipConfiguration(self._device)
        if int(status) != self._ok():
            raise D3XXError(
                f"FT_GetChipConfiguration failed with status {status} ({self._status_text(status)})"
            )
        return {
            "vendor_id": int(self._field(config, "VendorID", default=0)),
            "product_id": int(self._field(config, "ProductID", default=0)),
            "fifo_clock": int(self._field(config, "FIFOClock", default=0)),
            "fifo_mode": int(self._field(config, "FIFOMode", default=0)),
            "channel_config": int(self._field(config, "ChannelConfig", default=0)),
            "optional_feature_support": int(self._field(config, "OptionalFeatureSupport", default=0)),
            "battery_charging_gpio_config": int(self._field(config, "BatteryChargingGPIOConfig", default=0)),
            "flash_eeprom_detection": int(self._field(config, "FlashEEPROMDetection", default=0)),
            "msio_control": int(self._field(config, "MSIO_Control", default=0)),
            "gpio_control": int(self._field(config, "GPIO_Control", default=0)),
        }

    def set_pipe_timeout(self, pipe_id: int, timeout_ms: int) -> None:
        pipe = self._pipe_for_id(pipe_id)
        status = self._pyd3xx.FT_SetPipeTimeout(self._device, pipe, int(timeout_ms))
        if int(status) != self._ok():
            raise D3XXError(f"FT_SetPipeTimeout failed with status {status} ({self._status_text(status)})")
        self._pipe_timeout_ms[int(pipe_id)] = int(timeout_ms)

    def get_pipe_timeout(self, pipe_id: int) -> int:
        pipe = self._pipe_for_id(pipe_id)
        status, timeout_ms = self._pyd3xx.FT_GetPipeTimeout(self._device, pipe)
        if int(status) != self._ok():
            raise D3XXError(f"FT_GetPipeTimeout failed with status {status} ({self._status_text(status)})")
        return int(timeout_ms)

    def set_suspend_timeout(self, timeout_ms: int) -> None:
        if self._device is None:
            raise D3XXError("Device is not open")
        timeout_ms = int(timeout_ms)

        method = getattr(self._device, "setSuspendTimeout", None)
        if callable(method):
            result = method(timeout_ms)
            if result is not None and int(result) != self._ok():
                raise D3XXError(
                    f"setSuspendTimeout failed with status {result} ({self._status_text(int(result))})"
                )
            return

        if hasattr(self._pyd3xx, "FT_SetSuspendTimeout"):
            status = self._pyd3xx.FT_SetSuspendTimeout(self._device, timeout_ms)
            if int(status) != self._ok():
                raise D3XXError(
                    f"FT_SetSuspendTimeout failed with status {status} ({self._status_text(status)})"
                )
            return

        raise D3XXError("FT_SetSuspendTimeout/setSuspendTimeout is not available in this PyD3XX module")

    def get_suspend_timeout(self) -> int:
        if self._device is None:
            raise D3XXError("Device is not open")

        method = getattr(self._device, "getSuspendTimeout", None)
        if callable(method):
            return int(method())

        if hasattr(self._pyd3xx, "FT_GetSuspendTimeout"):
            status, timeout_ms = self._pyd3xx.FT_GetSuspendTimeout(self._device)
            if int(status) != self._ok():
                raise D3XXError(
                    f"FT_GetSuspendTimeout failed with status {status} ({self._status_text(status)})"
                )
            return int(timeout_ms)

        raise D3XXError("FT_GetSuspendTimeout/getSuspendTimeout is not available in this PyD3XX module")

    def flush_pipe(self, pipe_id: int) -> None:
        pipe = self._pipe_for_id(pipe_id)
        status = self._pyd3xx.FT_FlushPipe(self._device, pipe)
        if int(status) != self._ok():
            raise D3XXError(f"FT_FlushPipe failed with status {status} ({self._status_text(status)})")

    def abort_pipe(self, pipe_id: int) -> None:
        pipe = self._pipe_for_id(pipe_id)
        status = self._pyd3xx.FT_AbortPipe(self._device, pipe)
        if int(status) != self._ok():
            raise D3XXError(f"FT_AbortPipe failed with status {status} ({self._status_text(status)})")

    def _release_overlaps(self) -> None:
        if self._pyd3xx is None or self._device is None:
            self._overlaps_by_id.clear()
            return
        for pipe_id, overlap in list(self._overlaps_by_id.items()):
            status = self._pyd3xx.FT_ReleaseOverlapped(self._device, overlap)
            if int(status) != self._ok():
                raise D3XXError(
                    f"FT_ReleaseOverlapped failed for pipe 0x{pipe_id:02X} "
                    f"with status {status} ({self._status_text(status)})"
                )
        self._overlaps_by_id.clear()

    def _refresh_pipe_cache(self) -> None:
        self._pipes_by_id.clear()
        self._pipe_meta_by_id.clear()
        self._pipe_timeout_ms.clear()
        try:
            interface_count = max(1, min(4, int(self.get_configuration_descriptor()["num_interfaces"])))
        except D3XXError:
            interface_count = 2
        for interface_index in range(interface_count):
            try:
                descriptor = self.get_interface_descriptor(interface_index)
            except D3XXError:
                continue
            for pipe_index in range(int(descriptor["num_endpoints"])):
                try:
                    self.get_pipe_information(interface_index, pipe_index)
                except D3XXError:
                    continue

    def _pipe_for_id(self, pipe_id: int) -> object:
        if self._device is None:
            raise D3XXError("Device is not open")
        pipe_id = int(pipe_id)
        if pipe_id not in self._pipes_by_id:
            self._refresh_pipe_cache()
        if pipe_id not in self._pipes_by_id:
            raise D3XXError(f"FT601 pipe 0x{pipe_id:02X} was not reported by PyD3XX")
        return self._pipes_by_id[pipe_id]

    def _overlap_for_id(self, pipe_id: int) -> object:
        self._pipe_for_id(pipe_id)
        return self._overlaps_by_id[int(pipe_id)]

    def _fifo_index_for_pipe(self, pipe_id: int) -> int:
        meta = self._pipe_meta_by_id.get(int(pipe_id))
        if meta is None:
            self._pipe_for_id(pipe_id)
            meta = self._pipe_meta_by_id[int(pipe_id)]
        return int(meta["pipe_index"]) // 2

    def _wait_for_overlap(self, pipe_id: int, timeout_ms: Optional[int] = None) -> int:
        overlap = self._overlap_for_id(pipe_id)
        status = self._status_value("FT_IO_INCOMPLETE", 25)
        io_incomplete = self._status_value("FT_IO_INCOMPLETE", 25)
        io_pending = self._status_value("FT_IO_PENDING", 24)
        if timeout_ms is None:
            timeout_ms = self._pipe_timeout_ms.get(int(pipe_id), 1000)
        deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0

        while status in (io_incomplete, io_pending):
            status, transferred = self._pyd3xx.FT_GetOverlappedResult(self._device, overlap, False)
            if int(status) == self._ok():
                return int(transferred)
            if int(status) in (io_incomplete, io_pending):
                if time.monotonic() >= deadline:
                    try:
                        self.abort_pipe(pipe_id)
                    finally:
                        raise D3XXError(
                            f"FT_TIMEOUT: overlapped transfer on pipe 0x{pipe_id:02X} "
                            f"did not complete within {timeout_ms} ms"
                        )
                time.sleep(0.001)
                continue
            raise D3XXError(
                f"FT_GetOverlappedResult failed for pipe 0x{pipe_id:02X} "
                f"with status {status} ({self._status_text(status)})"
            )
        return 0

    def read_block(self, size: int, pipe_id: Optional[int] = None) -> bytes:
        pipe_id = self.read_pipe if pipe_id is None else int(pipe_id)
        pipe = self._pipe_for_id(pipe_id)
        overlap = self._overlap_for_id(pipe_id)
        with self._io_lock:
            if self._platform() in {"linux", "darwin"}:
                status, read_buffer, _ = self._pyd3xx.FT_ReadPipeAsync(
                    self._device,
                    self._fifo_index_for_pipe(pipe_id),
                    int(size),
                    overlap,
                )
            else:
                status, read_buffer, _ = self._pyd3xx.FT_ReadPipe(
                    self._device,
                    pipe,
                    int(size),
                    overlap,
                )
            if int(status) not in (self._ok(), self._status_value("FT_IO_PENDING", 24)):
                raise D3XXError(f"FT_ReadPipe failed with status {status} ({self._status_text(status)})")
            transferred = self._wait_for_overlap(pipe_id)
        if int(transferred) <= 0:
            return b""
        return bytes(read_buffer.Value()[: int(transferred)])

    def write_block(self, bytes_data: bytes, pipe_id: Optional[int] = None) -> int:
        pipe_id = self.write_pipe if pipe_id is None else int(pipe_id)
        pipe = self._pipe_for_id(pipe_id)
        overlap = self._overlap_for_id(pipe_id)
        buffer = self._pyd3xx.FT_Buffer.from_bytes(bytes(bytes_data))
        with self._io_lock:
            if self._platform() in {"linux", "darwin"}:
                status, _ = self._pyd3xx.FT_WritePipeAsync(
                    self._device,
                    self._fifo_index_for_pipe(pipe_id),
                    buffer,
                    len(bytes_data),
                    overlap,
                )
            else:
                status, _ = self._pyd3xx.FT_WritePipe(
                    self._device,
                    pipe,
                    buffer,
                    len(bytes_data),
                    overlap,
                )
            if int(status) not in (self._ok(), self._status_value("FT_IO_PENDING", 24)):
                raise D3XXError(f"FT_WritePipe failed with status {status} ({self._status_text(status)})")
            transferred = self._wait_for_overlap(pipe_id)
        if int(transferred) != len(bytes_data):
            raise D3XXError(
                f"FT_WritePipe short write: requested {len(bytes_data)} byte(s), "
                f"transferred {int(transferred)} byte(s)"
            )
        return int(transferred)


class RxWorker(QtCore.QThread):
    bytes_received = QtCore.pyqtSignal(bytes)
    io_error = QtCore.pyqtSignal(str)

    def __init__(self, device, read_block_size: int = 256 * 1024, poll_interval_s: float = 0.001, parent=None) -> None:
        super().__init__(parent)
        self.device = device
        self.read_block_size = read_block_size
        self.poll_interval_s = poll_interval_s
        self._running = threading.Event()
        self._running.set()

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        while self._running.is_set():
            try:
                block = self.device.read_block(self.read_block_size)
                if block:
                    self.bytes_received.emit(block)
                else:
                    time.sleep(self.poll_interval_s)
            except Exception as exc:  # pragma: no cover - hardware path
                if hasattr(self.device, "is_read_idle_exception") and self.device.is_read_idle_exception(exc):
                    time.sleep(self.poll_interval_s)
                    continue
                self.io_error.emit(str(exc))
                time.sleep(self.poll_interval_s)


class TxWorker(QtCore.QThread):
    bytes_written = QtCore.pyqtSignal(int)
    io_error = QtCore.pyqtSignal(str)

    def __init__(self, device, parent=None) -> None:
        super().__init__(parent)
        self.device = device
        self.queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._running = threading.Event()
        self._running.set()

    def enqueue(self, payload: bytes) -> None:
        self.queue.put(payload)

    def stop(self) -> None:
        self._running.clear()
        self.queue.put(None)

    def run(self) -> None:
        while self._running.is_set():
            payload = self.queue.get()
            if payload is None:
                continue
            try:
                written = self.device.write_block(payload)
                self.bytes_written.emit(written)
            except Exception as exc:  # pragma: no cover - hardware path
                if hasattr(self.device, "is_timeout_exception") and self.device.is_timeout_exception(exc):
                    self.io_error.emit(
                        "FT_WritePipe timeout: host-to-FPGA downlink was not accepted. "
                        "Check FT601 mode, write_pipe, and whether the FPGA FIFO interface is alive."
                    )
                else:
                    self.io_error.emit(str(exc))


class DeviceService(QtCore.QObject):
    device_state_changed = QtCore.pyqtSignal(bool, str)
    log_message = QtCore.pyqtSignal(str)
    packet_received = QtCore.pyqtSignal(object)
    status_received = QtCore.pyqtSignal(object)
    tdc_events_received = QtCore.pyqtSignal(object)
    raw_bytes_received = QtCore.pyqtSignal(bytes)
    stats_updated = QtCore.pyqtSignal(object, object)
    read_state_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.device: Optional[PyD3XXDevice] = None
        self.rx_worker: Optional[RxWorker] = None
        self.tx_worker: Optional[TxWorker] = None
        self.parser = PacketParser()
        self.encoder = CommandEncoder()
        self.runtime_stats = RuntimeStats()
        self.rx_rate_meter = RateMeter()
        self.tx_rate_meter = RateMeter()
        self.packet_rate_meter = RateMeter()
        self.event_rate_meter = RateMeter()
        self.command_history: deque[CommandHistoryItem] = deque(maxlen=200)
        self.latest_status: Optional[StatusPacket] = None
        self._status_event = threading.Event()
        self.read_pipe = None
        self.write_pipe = None
        self.read_block_size = 256 * 1024
        self.read_enabled = True

    def enumerate_devices(self) -> list[DeviceInfo]:
        device = PyD3XXDevice()
        try:
            return device.enumerate_devices()
        except Exception as exc:
            self.log_message.emit(f"Device enumerate failed: {exc}")
            return []

    def _log_device_usb_diagnostics(self) -> None:
        if self.device is None:
            return

        try:
            chip_cfg = self.device.get_chip_configuration()
            self.log_message.emit(
                "FT601 chip config: "
                f"vid=0x{chip_cfg['vendor_id']:04X} pid=0x{chip_cfg['product_id']:04X} "
                f"fifo_clock={chip_cfg['fifo_clock']}({FIFO_CLOCK_NAMES.get(chip_cfg['fifo_clock'], 'UNKNOWN')}) "
                f"fifo_mode={chip_cfg['fifo_mode']}({FIFO_MODE_NAMES.get(chip_cfg['fifo_mode'], 'UNKNOWN')}) "
                f"channel_config={chip_cfg['channel_config']}({CHANNEL_CONFIG_NAMES.get(chip_cfg['channel_config'], 'UNKNOWN')}) "
                f"flash_eeprom_detection={chip_cfg['flash_eeprom_detection']}"
            )
        except Exception as exc:
            self.log_message.emit(f"FT601 chip config read failed: {exc}")

        try:
            config_descriptor = self.device.get_configuration_descriptor()
            self.log_message.emit(
                "FT601 config descriptor: "
                f"interfaces={config_descriptor['num_interfaces']} "
                f"value={config_descriptor['configuration_value']} "
                f"attributes=0x{config_descriptor['attributes']:02X} "
                f"max_power={config_descriptor['max_power'] * 2}mA"
            )
        except Exception as exc:
            self.log_message.emit(f"FT601 configuration descriptor read failed: {exc}")
            config_descriptor = {"num_interfaces": 2}

        try:
            interface_count = max(1, int(config_descriptor.get("num_interfaces", 1)))
            present_pipe_ids: set[int] = set()
            pipe_interfaces: dict[int, list[int]] = defaultdict(list)
            total_pipes = 0

            for interface_index in range(interface_count):
                descriptor = self.device.get_interface_descriptor(interface_index)
                self.log_message.emit(
                    f"FT601 interface[{interface_index}]: "
                    f"interface_number={descriptor['interface_number']} "
                    f"alt={descriptor['alternate_setting']} "
                    f"endpoints={descriptor['num_endpoints']} "
                    f"class=0x{descriptor['interface_class']:02X}"
                )
                pipes = self.device.list_pipe_information(interface_index)
                if not pipes:
                    self.log_message.emit(f"FT601 pipe list: no endpoints reported on interface[{interface_index}]")
                    continue

                for pipe in pipes:
                    total_pipes += 1
                    present_pipe_ids.add(pipe["pipe_id"])
                    pipe_interfaces[pipe["pipe_id"]].append(interface_index)
                    self.log_message.emit(
                        "FT601 pipe: "
                        f"if={interface_index} index={pipe['pipe_index']} id=0x{pipe['pipe_id']:02X} "
                        f"type={pipe['pipe_type']}({PIPE_TYPE_NAMES.get(pipe['pipe_type'], 'UNKNOWN')}) "
                        f"max_packet={pipe['max_packet_size']} interval={pipe['interval']}"
                    )
        except Exception as exc:
            self.log_message.emit(f"FT601 pipe diagnostics failed: {exc}")
            return

        if total_pipes == 0:
            self.log_message.emit("FT601 pipe list: no endpoints reported across all interfaces")

        read_ifaces = pipe_interfaces.get(self.read_pipe, [])
        write_ifaces = pipe_interfaces.get(self.write_pipe, [])
        self.log_message.emit(
            "FT601 pipe selection: "
            f"read_pipe=0x{self.read_pipe:02X} present={self.read_pipe in present_pipe_ids} on_if={read_ifaces or '[]'}, "
            f"write_pipe=0x{self.write_pipe:02X} present={self.write_pipe in present_pipe_ids} on_if={write_ifaces or '[]'}"
        )

    def open_device(
        self,
        device_index: int = 0,
        read_pipe: Optional[int] = None,
        write_pipe: Optional[int] = None,
        read_block_size: int = 256 * 1024,
        read_enabled: bool = True,
    ) -> None:
        if self.device is not None:
            return
        self.device = PyD3XXDevice()
        try:
            self.device.open_device(device_index)
            if read_pipe is not None:
                self.device.read_pipe = int(read_pipe)
            if write_pipe is not None:
                self.device.write_pipe = int(write_pipe)
        except Exception as exc:
            self.device = None
            self.device_state_changed.emit(False, str(exc))
            raise

        self.read_pipe = self.device.read_pipe
        self.write_pipe = self.device.write_pipe
        # Keep control-link bring-up reads short. Large blocking reads make it
        # harder to interleave command writes and can turn an empty IN pipe into
        # noisy FT_OTHER_ERROR reports on some D3XX versions.
        self.read_block_size = min(int(read_block_size), 16 * 1024)
        self.read_enabled = bool(read_enabled)
        self._status_event.clear()

        try:
            self.device.set_suspend_timeout(0)
            actual_suspend_timeout = self.device.get_suspend_timeout()
            self.log_message.emit(
                f"FT601 suspend timeout disabled: {actual_suspend_timeout} ms"
            )
        except Exception as exc:
            self.log_message.emit(f"FT601 suspend timeout setup skipped: {exc}")

        for pipe_id, timeout_ms, label in (
            (self.read_pipe, 100, "read"),
            (self.write_pipe, 1000, "write"),
        ):
            try:
                self.device.set_pipe_timeout(pipe_id, timeout_ms)
                actual_timeout = self.device.get_pipe_timeout(pipe_id)
                self.log_message.emit(
                    f"FT601 {label} pipe 0x{pipe_id:02X} timeout set to {actual_timeout} ms"
                )
            except Exception as exc:
                self.log_message.emit(f"FT601 {label} pipe timeout setup failed: {exc}")

            try:
                self.device.flush_pipe(pipe_id)
                self.log_message.emit(f"FT601 {label} pipe 0x{pipe_id:02X} flushed")
            except Exception as exc:
                self.log_message.emit(f"FT601 {label} pipe flush skipped: {exc}")

        self.tx_worker = TxWorker(self.device)
        self.tx_worker.bytes_written.connect(self._handle_tx_bytes)
        self.tx_worker.io_error.connect(self._handle_io_error)
        self.tx_worker.start()
        if self.read_enabled:
            self._start_rx_worker()
        self.device_state_changed.emit(True, "Connected")
        self.log_message.emit(
            f"FT601 device opened. read_pipe=0x{self.read_pipe:02X} "
            f"write_pipe=0x{self.write_pipe:02X} read_block_size={self.read_block_size} "
            f"read_enabled={int(self.read_enabled)}"
        )
        self._log_device_usb_diagnostics()

    def _start_rx_worker(self) -> None:
        if self.device is None or self.rx_worker is not None:
            return
        self.rx_worker = RxWorker(self.device, read_block_size=self.read_block_size)
        self.rx_worker.bytes_received.connect(self._handle_raw_bytes)
        self.rx_worker.io_error.connect(self._handle_io_error)
        self.rx_worker.start()
        self.read_enabled = True
        self.read_state_changed.emit(True)
        self.log_message.emit("FT601 RX reader enabled.")

    def _stop_rx_worker(self) -> None:
        if self.rx_worker is not None:
            self.rx_worker.stop()
            self.rx_worker.wait(1000)
            self.rx_worker = None
        self.read_enabled = False
        self._status_event.clear()
        self.read_state_changed.emit(False)
        self.log_message.emit("FT601 RX reader disabled.")

    def set_read_enabled(self, enabled: bool) -> None:
        if self.device is None:
            self.read_enabled = bool(enabled)
            self.read_state_changed.emit(self.read_enabled)
            return
        if enabled:
            self._start_rx_worker()
        else:
            self._stop_rx_worker()

    def close_device(self) -> None:
        if self.rx_worker is not None:
            self.rx_worker.stop()
            self.rx_worker.wait(1000)
            self.rx_worker = None
        if self.tx_worker is not None:
            self.tx_worker.stop()
            self.tx_worker.wait(1000)
            self.tx_worker = None
        if self.device is not None:
            try:
                self.device.close_device()
            finally:
                self.device = None
        self._status_event.clear()
        self.device_state_changed.emit(False, "Disconnected")
        self.log_message.emit("FT601 device closed.")

    def send_bytes(self, payload: bytes) -> None:
        if self.tx_worker is None:
            raise D3XXError("Device not open")
        self.tx_worker.enqueue(payload)

    @staticmethod
    def _frame_words(frame: bytes) -> list[int]:
        if len(frame) % 4 != 0:
            return []
        return [int.from_bytes(frame[index:index + 4], byteorder="little", signed=False) for index in range(0, len(frame), 4)]

    @staticmethod
    def _format_words(words: list[int], max_words: int | None = None) -> str:
        if not words:
            return "(none)"
        shown = words if max_words is None else words[:max_words]
        text = ", ".join(f"0x{word:08X}" for word in shown)
        if max_words is not None and len(words) > max_words:
            text += f", ... (+{len(words) - max_words} word(s))"
        return text

    @staticmethod
    def _format_bytes(data: bytes, max_bytes: int | None = None) -> str:
        if not data:
            return "(empty)"
        shown = data if max_bytes is None else data[:max_bytes]
        text = " ".join(f"{byte:02X}" for byte in shown)
        if max_bytes is not None and len(data) > max_bytes:
            text += f" ... (+{len(data) - max_bytes} byte(s))"
        return text

    @staticmethod
    def _packet_type_text(pkt_type: int) -> str:
        if pkt_type == protocol.PKT_TDC_RAW:
            return "TDC_RAW"
        if pkt_type == protocol.PKT_STATUS:
            return "STATUS"
        return f"0x{pkt_type:02X}"

    def _format_parsed_packet(self, packet) -> str:
        header = packet.header
        payload = self._format_words(packet.payload_words, max_words=64)
        message = (
            f"RX PARSED PACKET type={self._packet_type_text(header.pkt_type)} "
            f"sync=0x{header.sync:02X} ver={header.version} hdr_words={header.hdr_words} "
            f"seq={header.seq} payload_words={header.payload_words} "
            f"item_count={header.item_count} flags=0x{header.flags:04X} "
            f"timestamp_us={header.timestamp_us} payload=[{payload}]"
        )
        if packet.status is not None:
            message += (
                f" | STATUS flags=0x{packet.status.flags:04X} uptime_s={packet.status.uptime_seconds} "
                f"temp_avg_raw={packet.status.temp_avg_raw} counter_1s={packet.status.counter_1s} "
                f"tdc_drop={packet.status.tdc_drop_count} usb_drop={packet.status.usb_drop_count}"
            )
        elif packet.tdc_events is not None:
            preview = packet.tdc_events[:4]
            event_text = "; ".join(
                f"#{idx}:type={event.rec_type},ch={event.channel},class={event.event_class},"
                f"refid={event.refid},tstop={event.tstop}"
                for idx, event in enumerate(preview)
            )
            if len(packet.tdc_events) > len(preview):
                event_text += f"; ... (+{len(packet.tdc_events) - len(preview)} event(s))"
            message += f" | TDC events={len(packet.tdc_events)} [{event_text or 'none'}]"
        return message

    def send_command_sync(
        self,
        cmd_id: int,
        frame: bytes,
        timeout_ms: int = 1000,
        debug_details: str | None = None,
    ) -> CommandResult:
        cmd_name = protocol.CMD_NAMES.get(cmd_id, f"0x{cmd_id:02X}")
        frame_words = self._frame_words(frame)
        header_word = frame_words[0] if frame_words else None
        payload_words = frame_words[1:] if len(frame_words) > 1 else []
        setting_message = (
            f"TX SETTING {cmd_name} cmd=0x{cmd_id:02X} "
            f"pipe=0x{(self.write_pipe or DEFAULT_WRITE_PIPE):02X} timeout={timeout_ms}ms"
        )
        if debug_details:
            setting_message += f" | {debug_details}"
        self.log_message.emit(setting_message)

        packet_message = f"TX USB PACKET {cmd_name} length={len(frame)} byte(s)"
        if header_word is not None:
            packet_message += f" | header=0x{header_word:08X}"
        packet_message += (
            f" | payload_words=[{self._format_words(payload_words)}] "
            f"| bytes={self._format_bytes(frame)}"
        )
        self.log_message.emit(packet_message)

        self.send_bytes(frame)
        result = CommandResult(True, cmd_id, "SENT")
        self._append_history(result, 0)
        self.log_message.emit(f"TX {cmd_name} -> SENT")
        return result

    def _handle_raw_bytes(self, data: bytes) -> None:
        self.runtime_stats.rx_bytes += len(data)
        self.log_message.emit(
            f"RX USB BLOCK length={len(data)} byte(s) pipe=0x{(self.read_pipe or DEFAULT_READ_PIPE):02X} "
            f"bytes={self._format_bytes(data, max_bytes=512)}"
        )
        self.raw_bytes_received.emit(data)
        for packet in self.parser.feed(data):
            self.runtime_stats.packet_count += 1
            if packet.raw_bytes:
                self.log_message.emit(
                    f"RX USB PACKET length={len(packet.raw_bytes)} byte(s) "
                    f"bytes={self._format_bytes(packet.raw_bytes, max_bytes=512)}"
                )
            self.log_message.emit(self._format_parsed_packet(packet))
            self.packet_received.emit(packet)
            if packet.status is not None:
                self.runtime_stats.status_count += 1
                self.latest_status = packet.status
                self._status_event.set()
                self.status_received.emit(packet.status)
            if packet.tdc_events is not None:
                self.runtime_stats.tdc_event_count += len(packet.tdc_events)
                self.tdc_events_received.emit(packet.tdc_events)
        self._emit_stats()

    def _handle_tx_bytes(self, count: int) -> None:
        self.runtime_stats.tx_bytes += count
        self.log_message.emit(
            f"FT_WritePipe completed: wrote {count} byte(s) to pipe "
            f"0x{(self.write_pipe or DEFAULT_WRITE_PIPE):02X}"
        )
        self._emit_stats()

    def _handle_io_error(self, message: str) -> None:
        self.log_message.emit(f"I/O error: {message}")

    def wait_for_status_available(self, timeout_ms: int = 1500) -> bool:
        return self._status_event.wait(timeout_ms / 1000.0)

    def _append_history(self, result: CommandResult, timestamp_us: int) -> None:
        self.command_history.appendleft(
            CommandHistoryItem(
                cmd_id=result.cmd_id,
                name=protocol.CMD_NAMES.get(result.cmd_id, f"0x{result.cmd_id:02X}"),
                timestamp_us=timestamp_us,
                success=result.success,
                detail=result.message,
            )
        )

    def _emit_stats(self) -> None:
        rates = {
            "rx_bytes_per_sec": self.rx_rate_meter.update(self.runtime_stats.rx_bytes),
            "tx_bytes_per_sec": self.tx_rate_meter.update(self.runtime_stats.tx_bytes),
            "packets_per_sec": self.packet_rate_meter.update(self.runtime_stats.packet_count),
            "tdc_events_per_sec": self.event_rate_meter.update(self.runtime_stats.tdc_event_count),
        }
        self.stats_updated.emit(self.runtime_stats, rates)

    @staticmethod
    def decode_status_flags(flags: int) -> StatusFlagsDecoded:
        return StatusFlagsDecoded(
            flash_busy=bool((flags >> 7) & 0x1),
            gpx2_lclk_locked=bool((flags >> 6) & 0x1),
            gate_clk_locked=bool((flags >> 5) & 0x1),
            flash_error=bool((flags >> 4) & 0x1),
            usb_tx_backpressure=bool((flags >> 3) & 0x1),
            gpx2_event_overflow=bool((flags >> 2) & 0x1),
            gpx2_cfg_error=bool((flags >> 1) & 0x1),
            gpx2_cfg_done=bool(flags & 0x1),
        )
