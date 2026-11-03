"""Process logging for the driver monitor.

The console follows ``log_level``. The file under ``data/logs`` always keeps DEBUG,
including the file and line the message came from, so a field failure can be read
after the process has moved on. The dashboard process writes ``dms-ui.log`` beside
it; two processes must not share one rotating file.

Rule code should log a state change once, through :class:`StateLog`, instead of
once per frame.
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_HOOKS = False
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname).1s %(threadName)s %(name)s %(filename)s:%(lineno)d %(message)s"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5


def log_file(cfg, ui: bool = False) -> Path:
    """Resolve the log path after :func:`dms.config.load_config` has set the data dir."""
    from .config import data_path

    raw = str(cfg.get("log_file") or "logs/dms.log")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = data_path(*path.parts)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    if ui:
        path = path.with_name(f"{path.stem}-ui{path.suffix or '.log'}")
    return path


def configure_logging(level: str = "INFO", path: Path | None = None) -> Path | None:
    """Attach console and rotating-file handlers to the ``dms`` logger tree."""
    level_name = str(level or "INFO").upper()
    console_level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("dms")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rolling = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8")
        rolling.setLevel(logging.DEBUG)
        rolling.setFormatter(formatter)
        logger.addHandler(rolling)

    _install_hooks()
    logger.info("log file %s (console %s, file DEBUG)", path or "disabled", level_name)
    return path


def _install_hooks() -> None:
    global _HOOKS
    if _HOOKS:
        return
    _HOOKS = True

    def _sys(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("dms").critical("unhandled exception", exc_info=(exc_type, exc, tb))

    def _thread(args):
        if args.exc_type is SystemExit:
            return
        name = getattr(args.thread, "name", "?")
        logging.getLogger("dms").critical(
            "thread %s crashed", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _sys
    threading.excepthook = _thread


class StateLog:
    """Log a rule only when its value changes, so a 30 FPS loop stays readable."""

    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger
        self._prev: dict = {}

    def change(self, key: str, value, message: str, *args, level: int = logging.DEBUG) -> None:
        if self._prev.get(key) == value:
            return
        self._prev[key] = value
        # stacklevel points at the caller, not at this helper
        self.log.log(level, message, *args, stacklevel=2)

    def rise(self, key: str, active: bool, message: str, *args) -> None:
        """Log when a condition becomes true, and one line when it clears."""
        previous = bool(self._prev.get(key, False))
        current = bool(active)
        self._prev[key] = current
        if current and not previous:
            self.log.info(message, *args, stacklevel=2)
        elif previous and not current:
            self.log.info("%s cleared", key, stacklevel=2)
