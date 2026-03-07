import os
import sys
import tempfile
from pathlib import Path

# Give matplotlib a writable, process-local config/cache dir so test collection
# does not hang on the global font cache lock in shared environments.
_MPLDIR = Path(tempfile.mkdtemp(prefix="mplconfig_"))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
