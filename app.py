"""Hugging Face Spaces and root entrypoint for BIST Financial Analyst Assistant."""

import sys
from pathlib import Path
import runpy

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    app_path = project_root / "src" / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
