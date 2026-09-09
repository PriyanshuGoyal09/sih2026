"""Load JC-Wise-style PDW HDF5 scenarios into the SmartScan RF grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from environment.emitter import Emitter
from environment.noise import NoiseModel
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PDWScenarioInfo:
    path: Path
    name: str
    num_pulses: int
    collection_time_s: float
    scan_mode: str
    num_transmitters: int
    description: str


def list_pdw_scenarios(root: Path | None = None) -> list[PDWScenarioInfo]:
    root = root or PROJECT_ROOT
    scenarios = []
    for path in sorted(root.glob("config_*.h5")):
        with h5py.File(path, "r") as handle:
            meta = dict(handle["metadata"].attrs)
            receiver = dict(handle["metadata/receiver"].attrs)
            scenarios.append(
                PDWScenarioInfo(
                    path=path,
                    name=path.name,
                    num_pulses=int(meta.get("num_pulses", handle["data"].shape[0])),
                    collection_time_s=float(meta.get("collection_time_s", 30.0)),
                    scan_mode=str(receiver.get("scan_mode", "unknown")),
                    num_transmitters=len(handle["metadata/transmitters"].keys()),
                    description=str(meta.get("description", "")),
                )
            )
    return scenarios


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_transmitters(group: h5py.Group) -> list[dict[str, Any]]:
    transmitters = []
    keys = sorted(
        group.keys(),
        key=lambda name: int(name.rsplit("_", 1)[-1])
        if name.rsplit("_", 1)[-1].isdigit()
        else name,
    )
    for key in keys:
        node = group[key]
        index = int(key.rsplit("_", 1)[-1]) if key.rsplit("_", 1)[-1].isdigit() else key
        freq_cfg = node["frequency_config"]
        freqs = (
            np.asarray(freq_cfg["freqs_mhz"][:], dtype=float)
            if "freqs_mhz" in freq_cfg
            else np.array([], dtype=float)
        )
        scan = node["scan_config"].attrs if "scan_config" in node else {}
        power = node["power_config"].attrs if "power_config" in node else {}
        pri = node["pri_config"].attrs if "pri_config" in node else {}
        transmitters.append(
            {
                "id": index,
                "key": key,
                "name": str(_decode(node.attrs.get("function", key))),
                "freq_mode": str(_decode(freq_cfg.attrs.get("freq_mode", "unknown"))),
                "freqs_mhz": freqs.tolist(),
                "center_mhz": float(np.mean(freqs)) if freqs.size else 0.0,
                "scan_type": str(_decode(scan.get("scan_type", "unknown"))),
                "scan_rate_rpm": float(scan.get("scan_rate_rpm", 0.0)),
                "beam_width_deg": float(scan.get("beam_width_deg", 0.0)),
                "power_w": float(power.get("power_w", 0.0)),
                "gain_db": float(power.get("gain", 0.0)),
                "pri_mode": str(_decode(pri.get("pri_mode", "unknown"))),
            }
        )
    return transmitters


def _behavior_from_transmitter(tx: dict[str, Any]) -> tuple[str, float, int]:
    mode = tx["freq_mode"].lower()
    if "hop" in mode or "random" in mode:
        n_freqs = max(len(tx["freqs_mhz"]), 1)
        return "agile", 1.0, max(1, min(n_freqs, 6))
    rpm = tx["scan_rate_rpm"]
    beam = tx["beam_width_deg"]
    if rpm > 0 and beam > 0:
        duty = float(np.clip(beam / 360.0, 0.02, 0.5))
        return "periodic", duty, 0
    return "fixed", 1.0, 0


def _dwell_centres(handle: h5py.File, bandwidth_mhz: float) -> np.ndarray:
    receiver = handle["metadata/receiver"]
    dwell = np.asarray(receiver["dwell_centres_mhz"][:], dtype=float)
    if dwell.size > 0:
        return dwell

    freq_range = np.asarray(receiver["freq_range_mhz"][:], dtype=float)
    fmin, fmax = float(freq_range[0]), float(freq_range[1])
    start = fmin - bandwidth_mhz / 2.0
    centres = np.arange(start, fmax, bandwidth_mhz, dtype=float)
    if centres.size == 0:
        centres = np.array([0.5 * (fmin + fmax)], dtype=float)
    return centres


def load_pdw_environment(
    path: str | Path,
    slot_duration_s: float = 0.1,
    num_time_slots: int | None = None,
    seed: int = 42,
) -> tuple[RFEnvironment, dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with h5py.File(path, "r") as handle:
        pulses = np.asarray(handle["data"][:], dtype=np.float64)
        labels = np.asarray(handle["labels"][:], dtype=int).ravel()
        meta = dict(handle["metadata"].attrs)
        receiver_attrs = dict(handle["metadata/receiver"].attrs)
        transmitters = _read_transmitters(handle["metadata/transmitters"])
        bandwidth_mhz = float(receiver_attrs.get("bandwith_mhz", 500.0))
        sensitivity = float(receiver_attrs.get("sensitivity_dbm", -110.0))
        collection_s = float(meta.get("collection_time_s", 30.0))
        dwell = _dwell_centres(handle, bandwidth_mhz)
        freq_range = np.asarray(
            handle["metadata/receiver/freq_range_mhz"][:], dtype=float
        )

    toa_us, frequency_mhz, _pulse_width, _aoa, amplitude_dbm = pulses.T
    toa_s = toa_us * 1e-6

    if num_time_slots is None:
        span = max(float(toa_s.max()) if toa_s.size else collection_s, collection_s)
        num_time_slots = max(1, int(np.ceil(span / slot_duration_s)))

    num_bands = int(dwell.size)
    half_bw = bandwidth_mhz / 2.0

    time_index = np.clip(
        np.floor(toa_s / slot_duration_s).astype(int),
        0,
        num_time_slots - 1,
    )
    distances = np.abs(frequency_mhz[:, None] - dwell[None, :])
    band_index = np.argmin(distances, axis=1)
    in_band = distances[np.arange(frequency_mhz.size), band_index] <= half_bw

    noise = NoiseModel(noise_floor_dbm=sensitivity, noise_std_db=2.0)
    truth = np.zeros((num_time_slots, num_bands), dtype=int)
    power = np.full(
        (num_time_slots, num_bands),
        noise.noise_floor_dbm,
        dtype=float,
    )

    valid = np.nonzero(in_band)[0]
    if valid.size:
        t_valid = time_index[valid]
        b_valid = band_index[valid]
        a_valid = amplitude_dbm[valid]
        truth[t_valid, b_valid] = 1
        np.maximum.at(power, (t_valid, b_valid), a_valid)

    emitter_band_map: dict[int, dict[str, int]] = {
        t: {} for t in range(num_time_slots)
    }
    if valid.size:
        label_valid = labels[valid]
        # Keep the strongest pulse for each (slot, emitter).
        order = np.argsort(a_valid)
        for idx in order:
            emitter_id = f"E{int(label_valid[idx]):02d}"
            emitter_band_map[int(t_valid[idx])][emitter_id] = int(b_valid[idx])

    events = _build_events(emitter_band_map, num_time_slots)

    emitters = []
    for tx in transmitters:
        behavior, duty, agility = _behavior_from_transmitter(tx)
        start_band = int(np.argmin(np.abs(dwell - tx["center_mhz"]))) if dwell.size else 0
        # Convert approximate ERP to a received-power-like dBm placeholder.
        power_dbm = min(-20.0, 10.0 * np.log10(max(tx["power_w"], 1e-3)) - 70.0)
        emitters.append(
            Emitter(
                emitter_id=f"E{int(tx['id']):02d}",
                name=tx["name"],
                center_frequency=tx["center_mhz"],
                bandwidth=bandwidth_mhz,
                power_dbm=float(power_dbm),
                behavior=behavior,
                duty_cycle=duty,
                agility=agility,
                start_band=start_band,
            )
        )

    config = RFEnvironmentConfig(
        num_bands=num_bands,
        num_time_slots=num_time_slots,
        band_start_mhz=float(dwell[0] - half_bw) if dwell.size else float(freq_range[0]),
        band_width_mhz=bandwidth_mhz,
        seed=seed,
    )
    environment = RFEnvironment(config, emitters, noise)
    environment.load_occupancy(
        truth=truth,
        power=power,
        transmission_events=events,
        emitter_band_map=emitter_band_map,
        band_centres_mhz=dwell,
        scenario_name=path.name,
    )

    info = {
        "path": str(path),
        "name": path.name,
        "num_pulses": int(pulses.shape[0]),
        "mapped_pulses": int(valid.size),
        "collection_time_s": collection_s,
        "slot_duration_s": slot_duration_s,
        "scan_mode": str(receiver_attrs.get("scan_mode", "unknown")),
        "sensitivity_dbm": sensitivity,
        "bandwidth_mhz": bandwidth_mhz,
        "freq_range_mhz": freq_range.tolist(),
        "dwell_centres_mhz": dwell.tolist(),
        "transmitters": transmitters,
        "description": str(meta.get("description", "")),
        "occupancy": float(np.mean(truth)),
    }
    return environment, info


def _build_events(
    emitter_band_map: dict[int, dict[str, int]],
    num_time_slots: int,
) -> list[dict[str, Any]]:
    open_events: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for t in range(num_time_slots):
        active = emitter_band_map.get(t, {})
        for emitter_id, band in active.items():
            if emitter_id not in open_events:
                open_events[emitter_id] = {
                    "emitter_id": emitter_id,
                    "start_time": t,
                    "end_time": t,
                    "start_band": band,
                    "bands": [band],
                }
            else:
                event = open_events[emitter_id]
                event["end_time"] = t
                event["bands"].append(band)

        for emitter_id in list(open_events):
            if emitter_id not in active:
                events.append(open_events.pop(emitter_id))

    events.extend(open_events.values())
    return events
