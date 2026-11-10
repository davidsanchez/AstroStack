"""Thread de calcul : lance le pipeline hors de l'interface (annulable, avec progression)."""
from PySide6.QtCore import QThread, Signal

from .pipeline import run_pipeline


class StackWorker(QThread):
    progressed = Signal(int, int, int, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        """Demande l'annulation coopérative (vérifiée à chaque étape du pipeline)."""
        self._cancel = True

    def run(self) -> None:
        from .pipeline import CancelledError

        try:
            result = run_pipeline(
                progress_cb=lambda s, i, n, m: self.progressed.emit(s, i, n, m),
                should_cancel=lambda: self._cancel,
                **self._cfg,
            )
            self.done.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__} : {exc}")
