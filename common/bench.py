"""Единые замеры производительности для всех модулей.

Чтобы метрики M1..M5 были сопоставимы, время, память и устройство
измеряются одним и тем же кодом.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psutil

REPORT_DIR = Path(__file__).resolve().parent.parent / "report" / "metrics"


# --------------------------------------------------------------------------- #
# устройство
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> str:
    """auto -> mps, если доступна, иначе cpu."""
    if prefer != "auto":
        return prefer
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def gpu_memory_mb() -> float:
    """Текущая занятая память GPU. На CPU-устройствах вернёт 0."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**20
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 2**20
    except (ImportError, AttributeError):
        pass
    return 0.0


def rss_mb() -> float:
    """Резидентная память процесса."""
    return psutil.Process().memory_info().rss / 2**20


# --------------------------------------------------------------------------- #
# замер
# --------------------------------------------------------------------------- #
@dataclass
class Bench:
    """Накопитель замеров одного эксперимента.

    with Bench("m1_rubert") as b:
        for text in texts:
            with b.step():
                model(text)
    b.save()
    """

    name: str
    # фактическое устройство, на котором считает модель. Без явного указания
    # берётся доступное в системе, что может не совпадать: M1 сознательно
    # работает на CPU, хотя MPS доступна.
    device: str | None = None
    extra: dict = field(default_factory=dict)
    timings_ms: list[float] = field(default_factory=list)
    _rss_start: float = 0.0
    _rss_peak: float = 0.0
    _t_start: float = 0.0

    def __enter__(self) -> "Bench":
        self._rss_start = rss_mb()
        self._rss_peak = self._rss_start
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.total_s = time.perf_counter() - self._t_start

    @contextmanager
    def step(self):
        """Один замеряемый вызов модели."""
        t0 = time.perf_counter()
        yield
        self.timings_ms.append((time.perf_counter() - t0) * 1000)
        self._rss_peak = max(self._rss_peak, rss_mb())

    # ---- агрегаты -------------------------------------------------------- #
    def _pct(self, p: float) -> float:
        if not self.timings_ms:
            return 0.0
        data = sorted(self.timings_ms)
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]

    def summary(self) -> dict:
        n = len(self.timings_ms)
        total = getattr(self, "total_s", time.perf_counter() - self._t_start)
        return {
            "name": self.name,
            "device": self.device or get_device(),
            "n_calls": n,
            "total_s": round(total, 3),
            "latency_ms": {
                "mean": round(statistics.fmean(self.timings_ms), 2) if n else 0,
                "p50": round(self._pct(50), 2),
                "p95": round(self._pct(95), 2),
                "max": round(max(self.timings_ms), 2) if n else 0,
            },
            "throughput_per_s": round(n / total, 2) if total > 0 else 0,
            "memory_mb": {
                "rss_start": round(self._rss_start, 1),
                "rss_peak": round(self._rss_peak, 1),
                "rss_delta": round(self._rss_peak - self._rss_start, 1),
                "gpu_allocated": round(gpu_memory_mb(), 1),
            },
            "env": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            **self.extra,
        }

    def save(self, filename: str | None = None) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / (filename or f"{self.name}.json")
        path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def report(self) -> None:
        s = self.summary()
        lat, mem = s["latency_ms"], s["memory_mb"]
        print(
            f"\n[{s['name']}] device={s['device']} calls={s['n_calls']}\n"
            f"  латентность: p50 {lat['p50']} мс | p95 {lat['p95']} мс | "
            f"среднее {lat['mean']} мс\n"
            f"  пропускная способность: {s['throughput_per_s']}/с\n"
            f"  память: RSS пик {mem['rss_peak']} МБ (+{mem['rss_delta']}) | "
            f"GPU {mem['gpu_allocated']} МБ"
        )
