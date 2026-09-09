"""
SIH26055 Smart Scan demo.

    streamlit run ew_scan_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.ui import render

render()
