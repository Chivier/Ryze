"""Progress tracking utilities."""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and report progress for long-running operations."""

    def __init__(self, total_steps: int = 100, callback: Optional[Callable[[float, str], None]] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self._callback = callback

    def update(self, steps: int = 1, message: str = "") -> None:
        self.current_step = min(self.current_step + steps, self.total_steps)
        progress = self.current_step / self.total_steps
        if self._callback:
            self._callback(progress, message)
        logger.debug("Progress: %.1f%% - %s", progress * 100, message)

    def set_callback(self, callback: Callable[[float, str], None]) -> None:
        self._callback = callback

    @property
    def progress(self) -> float:
        return self.current_step / self.total_steps if self.total_steps > 0 else 0.0
