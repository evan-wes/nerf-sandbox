"""
Signal handling utilities for graceful control of long-running training.

- SIGINT: set ctrl.sigint=True, Trainer exits after saving a checkpoint.
- SIGUSR1: set ctrl.sigusr1=True, Trainer pauses to render previews & path.
- SIGUSR2: set ctrl.cancel_render=True, cancel in-progress path rendering.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass


@dataclass
class SignalController:
    sigint: bool = False
    sigusr1: bool = False
    cancel_render: bool = False


def install_signal_handlers(ctrl: SignalController) -> None:
    def on_sigint(signum, frame):
        ctrl.sigint = True
        print("[SIGNAL] SIGINT received — will exit gracefully after checkpoint.")
    def on_usr1(signum, frame):
        ctrl.sigusr1 = True
        print("[SIGNAL] SIGUSR1 received — will render previews/path at next safe point.")
    def on_usr2(signum, frame):
        ctrl.cancel_render = True
        print("[SIGNAL] SIGUSR2 received — cancel current render task if running.")

    signal.signal(signal.SIGINT, on_sigint)
    try:
        signal.signal(signal.SIGUSR1, on_usr1)
        signal.signal(signal.SIGUSR2, on_usr2)
    except AttributeError:
        # Windows does not have SIGUSR1/2
        pass
