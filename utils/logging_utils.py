from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


class RateMeter:
    def __init__(self) -> None:
        self.last_time = monotonic()
        self.last_value = 0

    def update(self, total_value: int) -> float:
        now = monotonic()
        elapsed = max(1e-6, now - self.last_time)
        rate = (total_value - self.last_value) / elapsed
        self.last_time = now
        self.last_value = total_value
        return rate


def hexdump(data: bytes, width: int = 16) -> str:
    lines = []
    for index in range(0, len(data), width):
        chunk = data[index : index + width]
        hex_text = " ".join(f"{byte:02X}" for byte in chunk)
        lines.append(f"{index:08X}  {hex_text}")
    return "\n".join(lines)

