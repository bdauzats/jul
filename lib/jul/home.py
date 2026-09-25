"""Where jul keeps its files: presets, contexts, calibration data. `JUL_HOME` moves it (default ~/.jul),
for example into a read-only deployment package where the home directory is not writable."""

import os
from pathlib import Path

JUL_HOME = Path(os.environ.get("JUL_HOME") or Path.home() / ".jul")
