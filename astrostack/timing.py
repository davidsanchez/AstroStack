"""Chronométrage des étapes de calibration (affiché dans le journal en fin d'empilement).

Les lectures RAW se font dans des threads : les durées sont donc CUMULÉES sur toutes les
images (elles peuvent dépasser le temps réel total). Le nombre d'appels est indiqué pour
repérer les images relues (cache plein, passes par bandes).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext


class Chrono:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tot: dict[str, float] = {}
        self._cnt: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        with self._lock:
            self._tot[name] = self._tot.get(name, 0.0) + dt
            self._cnt[name] = self._cnt.get(name, 0) + 1

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def summary(self) -> str:
        with self._lock:
            parts = [
                f"{k} {v:.2f} s ({self._cnt[k]}× , {v / self._cnt[k]:.2f} s/appel)"
                for k, v in self._tot.items()
            ]
        return " ; ".join(parts) if parts else "aucune mesure"


class Lap:
    """Découpe une portion de code en étapes successives (chaque mark = temps depuis le mark précédent)."""

    def __init__(self, chrono: Chrono | None) -> None:
        self._chrono = chrono
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if self._chrono is None:
            return
        now = time.perf_counter()
        self._chrono.add(name, now - self._t)
        self._t = now


def measure(chrono: Chrono | None, name: str):
    return chrono.measure(name) if chrono is not None else nullcontext()
