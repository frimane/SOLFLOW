# Public core package. The implementation is split into small source files

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_PARTS = (
    "base.py", "math.py", "representations.py", "daydata.py",
    "windows.py", "priors.py", "flow.py", "metrics.py",
    "baselines.py", "pipeline.py", "selftest.py", "cli.py",
)

for _part in _PARTS:
    _path = _PACKAGE_DIR / _part
    exec(compile(_path.read_text(encoding="utf-8"), str(_path), "exec"), globals(), globals())

del Path, _PACKAGE_DIR, _PARTS, _part, _path
