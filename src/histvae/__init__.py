from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("histvae")
except PackageNotFoundError:  # pragma: no cover - fallback for non-installed usage
    __version__ = "0+unknown"

from .core import HistVAE
