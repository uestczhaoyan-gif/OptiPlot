"""Find a working Python environment without installing or modifying packages."""

from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
candidates = [
    os.environ.get("OPTIPLOT_PYTHON"),
    str(root / ".venv" / "Scripts" / "python.exe"),
    sys.executable,
]
conda = Path.home() / ".conda" / "environments.txt"
if conda.exists():
    candidates.extend(
        str(Path(p.strip()) / "python.exe")
        for p in conda.read_text(encoding="utf-8").splitlines()
        if p.strip()
    )
candidates.extend(
    [str(Path.home() / "anaconda3" / "python.exe"), str(Path.home() / "miniconda3" / "python.exe")]
)
seen = set()
env = os.environ.copy()
env.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
for candidate in candidates:
    if not candidate or candidate in seen or not Path(candidate).is_file():
        continue
    seen.add(candidate)
    try:
        probe = subprocess.run(
            [candidate, "-c", "import numpy,pandas,matplotlib,scipy,openpyxl,PIL,tkinter"],
            capture_output=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        continue
    if probe.returncode == 0:
        print("OptiPlot Python:", candidate)
        if "--check" in sys.argv:
            raise SystemExit(0)
        raise SystemExit(
            subprocess.call([candidate, str(root / "app.py"), *sys.argv[1:]], cwd=root, env=env)
        )
print("No Python environment with the plotting dependencies was found.")
print("Run: python -m pip install -r requirements.txt")
print("Then: python app.py")
raise SystemExit(1)
