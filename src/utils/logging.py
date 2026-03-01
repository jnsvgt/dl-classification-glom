import logging
import sys


_ROOT_LOGGER_NAME = "glomerular_ml"
_root = logging.getLogger(_ROOT_LOGGER_NAME)


def setup_logging(level="INFO", log_file=None):
    """Configure logging for the application."""
    _root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not _root.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        _root.addHandler(console)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            _root.addHandler(fh)


def get_logger(name):
    """Get a child logger under the app namespace."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
