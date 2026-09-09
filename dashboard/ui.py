"""Streamlit UI for SIH26055 Smart Scan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.engine import compare_schedulers, metrics_row
from data.pdw_loader import list_pdw_scenarios, load_pdw_environment
from environment.noise import NoiseModel
from environment.rf_environment import RFEnvironment


def _heatmap(matrix, title, cmap, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    im = ax.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Dwell band")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=True)
def _cached_load(path: str, slot_duration_s: float):
    environment, info = load_pdw_environment(
        path,
        slot_duration_s=slot_duration_s,
    )
    info = dict(info)
    return {
        "truth": environment.truth.copy(),
        "power": environment.power.copy(),
        "num_bands": environment.config.num_bands,
        "num_time_slots": environment.config.num_time_slots,
        "band_start_mhz": environment.config.band_start_mhz,
        "band_width_mhz": environment.config.band_width_mhz,
        "noise_floor": float(environment.noise_model.noise_floor_dbm),
        "band_centres": np.asarray(environment.band_centres_mhz),
        "events": environment.transmission_events,
        "emitter_band_map": environment.emitter_band_map,
        "info": info,
    }


def _rebuild_env(payload):
    from environment.rf_environment import RFEnvironmentConfig

    config = RFEnvironmentConfig(
        num_bands=payload["num_bands"],
        num_time_slots=payload["num_time_slots"],
        band_start_mhz=payload["band_start_mhz"],
        band_width_mhz=payload["band_width_mhz"],
    )
    environment = RFEnvironment(
        config,
        [],
        NoiseModel(noise_floor_dbm=payload["noise_floor"]),
    )
    environment.load_occupancy(
        truth=payload["truth"],
        power=payload["power"],
        transmission_events=payload["events"],
        emitter_band_map=payload["emitter_band_map"],
        band_centres_mhz=payload["band_centres"],
        scenario_name=payload["info"]["name"],
    )
    return environment, payload["info"]


def render():
    st.set_page_config(
        page_title="SIH26055 Smart Scan",
        layout="wide",
    )
    st.title("Smart Scan Strategy for Electronic Warfare")
    st.caption(
        "SIH26055 · DRDO · Closed-loop ES receiver scheduler with no prior emitter intel. "
        "Occupancy is built from PDW HDF5 scenarios (ToA, Frequency, Amplitude)."
    )

    scenarios = list_pdw_scenarios(ROOT)
    if not scenarios:
        st.error("No config_*.h5 files found in the project root.")
        return

    labels = {
        item.name: (
            f"{item.name}  ·  {item.scan_mode}  ·  "
            f"{item.num_pulses:,} PDWs  ·  {item.num_transmitters} emitters"
        )
        for item in scenarios
    }
    default = "config_1.h5" if "config_1.h5" in labels else scenarios[0].name

    with st.sidebar:
        st.header("Scenario")
        selected_name = st.selectbox(
            "PDW file",
            options=list(labels.keys()),
            format_func=lambda name: labels[name],
            index=list(labels.keys()).index(default),
        )
        slot_duration = st.select_slider(
            "Time-slot duration (s)",
            options=[0.05, 0.1, 0.2, 0.5],
            value=0.1,
        )
        scans_per_slot = st.slider("Scans per time slot", 1, 8, 4)
        st.caption(
            "Large stare-mode files (config_101 / 104) have 1M+ pulses and take longer to load."
        )
        run_clicked = st.button("Run comparison", type="primary")

    selected = next(item for item in scenarios if item.name == selected_name)
    payload = _cached_load(str(selected.path), float(slot_duration))
    environment, info = _rebuild_env(payload)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pulses mapped", f"{info['mapped_pulses']:,} / {info['num_pulses']:,}")
    c2.metric("Bands (receiver dwells)", environment.config.num_bands)
    c3.metric("Time slots", environment.config.num_time_slots)
    c4.metric("Spectrum occupancy", f"{info['occupancy'] * 100:.2f}%")

    st.write(
        f"**{info['description']}**  \n"
        f"Scan mode `{info['scan_mode']}` · "
        f"IBW {info['bandwidth_mhz']:.0f} MHz · "
        f"RF {info['freq_range_mhz'][0]:.0f}–{info['freq_range_mhz'][1]:.0f} MHz · "
        f"{len(info['transmitters'])} library emitters"
    )

    tabs = st.tabs(["Scheduler comparison", "Ground-truth spectrum", "Emitter library"])

    with tabs[1]:
        st.pyplot(
            _heatmap(environment.truth, "Occupied cells (PDW ground truth)", "magma", 0, 1),
            use_container_width=True,
        )
        st.caption("Rows are receiver dwell bands. Cells with pulses are occupied.")

    with tabs[2]:
        rows = []
        for tx in info["transmitters"]:
            freq_list = list(tx["freqs_mhz"])
            freqs = ", ".join(f"{f:.0f}" for f in freq_list[:6])
            if len(freq_list) > 6:
                freqs += ", …"
            rows.append(
                {
                    "ID": tx["id"],
                    "Emitter": tx["name"],
                    "Freq mode": tx["freq_mode"],
                    "RF (MHz)": freqs,
                    "Scan": tx["scan_type"],
                    "RPM": tx["scan_rate_rpm"],
                    "PRI": tx["pri_mode"],
                    "Power (W)": tx["power_w"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[0]:
        st.markdown(
            "The scheduler never sees emitter IDs. It only gets noisy power from the "
            "bands it chooses. **Sequential** is the open-loop sweep. **UCB** is a "
            "bandit baseline. **SmartScan V5.3** is the current closed-loop policy."
        )
        if not run_clicked and "compare_runs" not in st.session_state:
            st.info("Press **Run comparison** in the sidebar.")
            return

        if run_clicked:
            with st.spinner("Scanning the PDW-derived spectrum…"):
                st.session_state["compare_runs"] = compare_schedulers(
                    environment,
                    names=["sequential", "ucb", "smartscan"],
                    scans_per_slot=scans_per_slot,
                )

        runs = st.session_state.get("compare_runs", [])
        table = pd.DataFrame([metrics_row(run) for run in runs])
        display = table.copy()
        for col in ["Pd", "Pfa", "Precision", "Accuracy", "Intercept rate"]:
            display[col] = display[col].map(lambda x: f"{100 * x:.1f}%")
        display["Avg intercept time (slots)"] = display["Avg intercept time (slots)"].map(
            lambda x: f"{x:.2f}"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.bar_chart(table.set_index("Scheduler")[["Pd", "Intercept rate"]])

        cols = st.columns(len(runs))
        for col, run in zip(cols, runs):
            with col:
                st.pyplot(
                    _heatmap(
                        run["scan_mask"],
                        f"{run['name']} dwells",
                        "Blues",
                        0,
                        1,
                    ),
                    use_container_width=True,
                )
