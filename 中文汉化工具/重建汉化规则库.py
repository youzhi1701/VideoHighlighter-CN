#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "rebuild_localization_catalog.py"), run_name="__main__")
