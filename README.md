# SIH26055 — Smart Scan Strategy for Electronic Warfare

DRDO software problem: closed-loop electronic-support scan when there is **no prior reliable intel** on emitters. The receiver is narrowband relative to the mission spectrum, so the scheduler must choose **which dwell band to listen to, and when**.

## Data

`config_*.h5` files are synthetic **pulse descriptor word (PDW)** scenarios (ToA, Frequency, PulseWidth, AoA, Amplitude) plus a radar-emitter library in metadata. The environment maps pulses onto the receiver’s 500 MHz dwells and 0.1 s time slots. Ground truth is used **only** for evaluation.

Prefer `config_1.h5`, `config_10.h5`, `config_103.h5`, or `config_106.h5` for demos. `config_101.h5` and `config_104.h5` are stare-mode files with 1M+ pulses.

## Run

```bash
# from project root, using the existing venv
.venv/bin/python3 main.py --list
.venv/bin/python3 main.py --scenario config_1.h5
.venv/bin/python3 -m streamlit run ew_scan_app.py
```

The dashboard compares **sequential sweep** (open-loop), **UCB**, and **SmartScan V5.3** on Pd, Pfa, intercept rate, and intercept time.

## Presentation

Open `presentation/SIH26055_idea_presentation.html` in a browser (arrow keys to change slides). Print to PDF for the SIH portal. Copy diagrams from `presentation/assets/` into the official SIH PPT template if your SPOC requires that file. Fill team names on slide 1.

Speaker notes for the pitch: `presentation/SPEAKER_SCRIPT.md` (about 6 minutes + demo).

## Layout

- `data/pdw_loader.py` — HDF5 → occupancy grid
- `environment/` — RF truth and power
- `receiver/` — narrowband observe + adaptive detect
- `scheduler/smartscan_v53.py` — closed-loop policy
- `dashboard/` — comparison engine and Streamlit UI
- `metrics/evaluation.py` — SIH figures of merit
