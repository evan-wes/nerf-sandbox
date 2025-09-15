"""
GPU thermal monitoring & throttling utilities.

- Polls temperature (via pynvml if available; otherwise best-effort).
- Can gradually increase micro-batching or sleep to reduce heat.
- Writes to TensorBoard if provided.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    _PN = True
except Exception:
    _PN = False


@dataclass
class _ThermalState:
    last_check_step: int = 0
    last_check_time: float = 0.0
    last_temp: Optional[int] = None


class GpuThermalManager:
    def __init__(self, enable_throttle=False, temp_threshold=85, check_every=20,
                 cooldown_seconds=45, max_micro=16, throttle_sleep=5.0):
        self.enable = bool(enable_throttle)
        self.temp_threshold = int(temp_threshold)
        self.check_every = int(check_every)
        self.cooldown_seconds = int(cooldown_seconds)
        self.max_micro = int(max_micro)
        self.throttle_sleep = float(throttle_sleep)
        self._st = _ThermalState()
        if _PN:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass

    def _gpu_temp(self) -> Optional[int]:
        if not _PN:
            return None
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            return int(t)
        except Exception:
            return None

    def poll(self) -> dict:
        t = self._gpu_temp()
        self._st.last_temp = t
        self._st.last_check_time = time.time()
        return {"temp": t, "ts": self._st.last_check_time}

    def log_to_tb(self, tb_writer, step: int) -> None:
        if tb_writer is None: return
        if self._st.last_temp is not None:
            try:
                tb_writer.add_scalar("sys/gpu_temp_c", self._st.last_temp, step)
            except Exception:
                pass

    def guard(self, step: int, trainer_like) -> None:
        """Periodically check temperature and throttle/cool if needed."""
        if not self.enable: return
        if step - self._st.last_check_step < self.check_every:
            return

        self._st.last_check_step = step
        info = self.poll()
        t = info.get("temp", None)
        if t is None:
            return  # can't read temp; do nothing

        if t >= self.temp_threshold:
            # Cooldown path: increase micro-chunks if possible, otherwise sleep
            if hasattr(trainer_like, "micro_chunks"):
                mc = int(getattr(trainer_like, "micro_chunks", 0))
                if mc < self.max_micro:
                    setattr(trainer_like, "micro_chunks", mc + 1)
                    print(f"[THERMAL] GPU temp {t}C >= {self.temp_threshold}C → increasing micro_chunks to {mc+1}")
                    return
            print(f"[THERMAL] GPU temp {t}C >= {self.temp_threshold}C → sleeping {self.throttle_sleep}s")
            time.sleep(self.throttle_sleep)
