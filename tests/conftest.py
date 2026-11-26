"""Test the local source; skip the historical interactive plotting script."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
collect_ignore = ["test_reddemcee.py"]
