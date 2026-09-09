from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from .emitter import Emitter
from .noise import NoiseModel


@dataclass
class RFEnvironmentConfig:
    num_bands: int = 20
    num_time_slots: int = 100
    band_start_mhz: float = 100.0
    band_width_mhz: float = 5.0
    seed: int = 42


class RFEnvironment:
    """
    Simulated wideband RF environment.

    Maintains ground truth for evaluation while keeping that
    information separate from the receiver/scheduler.
    """

    def __init__(
        self,
        config: RFEnvironmentConfig,
        emitters: List[Emitter],
        noise_model: NoiseModel | None = None
    ):

        self.config = config
        self.emitters = emitters

        self.noise_model = noise_model or NoiseModel()

        self.rng = np.random.default_rng(config.seed)

        self.current_time = 0

        self.truth = np.zeros(
            (
                config.num_time_slots,
                config.num_bands
            ),
            dtype=int
        )

        self.power = np.full(
            (
                config.num_time_slots,
                config.num_bands
            ),
            self.noise_model.noise_floor_dbm,
            dtype=float
        )

        # Evaluation-only mapping.
        self.emitter_band_map = {}

        # Complete ground-truth transmission events.
        self.transmission_events = []

        self._generated = False
        self._frozen = False
        self.band_centres_mhz = None
        self.scenario_name = None

    def load_occupancy(
        self,
        truth: np.ndarray,
        power: np.ndarray,
        transmission_events: list,
        emitter_band_map: dict,
        band_centres_mhz=None,
        scenario_name: str | None = None,
    ) -> None:
        """Install precomputed occupancy from PDW recordings. reset() will not regenerate."""
        if truth.shape != power.shape:
            raise ValueError("truth and power must have the same shape")

        self.truth = np.asarray(truth, dtype=int)
        self.power = np.asarray(power, dtype=float)
        self.config.num_time_slots, self.config.num_bands = self.truth.shape
        self.transmission_events = list(transmission_events)
        self.emitter_band_map = dict(emitter_band_map)
        self.band_centres_mhz = band_centres_mhz
        self.scenario_name = scenario_name
        self._frozen = True
        self._generated = True
        self.current_time = 0

    def reset(self) -> None:

        self.current_time = 0

        if self._frozen:
            return

        self.truth.fill(0)

        self.power.fill(
            self.noise_model.noise_floor_dbm
        )

        self.emitter_band_map.clear()
        self.transmission_events.clear()

        for emitter in self.emitters:
            emitter.reset()

        self._generate_environment()

        self._generated = True

    def _generate_environment(self) -> None:

        # Currently active event for each emitter.
        open_events = {}

        for t in range(self.config.num_time_slots):

            self.emitter_band_map[t] = {}

            noise = self.noise_model.generate_noise(
                self.config.num_bands,
                self.rng
            )

            self.power[t] = noise

            active_emitters_this_slot = set()

            for emitter in self.emitters:

                active = emitter.is_active(
                    t,
                    self.rng
                )

                if not active:

                    # Close an existing event.
                    if emitter.emitter_id in open_events:

                        event = open_events.pop(
                            emitter.emitter_id
                        )

                        event["end_time"] = t - 1

                        self.transmission_events.append(event)

                    continue

                band = emitter.get_band(
                    t,
                    self.config.num_bands,
                    self.rng
                )

                if band < 0 or band >= self.config.num_bands:
                    continue

                active_emitters_this_slot.add(
                    emitter.emitter_id
                )

                self.truth[t, band] = 1

                self.power[t, band] = max(
                    self.power[t, band],
                    emitter.power_dbm
                )

                self.emitter_band_map[t][
                    emitter.emitter_id
                ] = band

                # Start a new transmission event.
                if emitter.emitter_id not in open_events:

                    open_events[emitter.emitter_id] = {
                        "emitter_id": emitter.emitter_id,
                        "start_time": t,
                        "end_time": t,
                        "start_band": band,
                        "bands": [band]
                    }

                else:

                    event = open_events[
                        emitter.emitter_id
                    ]

                    event["end_time"] = t
                    event["bands"].append(band)

            # Close events for emitters that were active previously
            # but did not produce a valid transmission this slot.
            for emitter_id in list(open_events.keys()):

                if emitter_id not in active_emitters_this_slot:

                    event = open_events.pop(
                        emitter_id
                    )

                    event["end_time"] = t - 1

                    self.transmission_events.append(
                        event
                    )

        # Close events still active at the end.
        for emitter_id in list(open_events.keys()):

            event = open_events.pop(
                emitter_id
            )

            event["end_time"] = (
                self.config.num_time_slots - 1
            )

            self.transmission_events.append(event)

    def step(self) -> Dict[str, Any]:

        if not self._generated:
            self.reset()

        if self.current_time >= self.config.num_time_slots:
            raise RuntimeError(
                "Simulation has finished. Call reset()."
            )

        observation = {
            "time_slot": self.current_time,
            "truth": self.truth[
                self.current_time
            ].copy(),

            "power": self.power[
                self.current_time
            ].copy(),

            # This is environment-side information.
            # The receiver/scheduler must NOT use it.
            "active_emitters": list(
                self.emitter_band_map[
                    self.current_time
                ].keys()
            )
        }

        self.current_time += 1

        return observation

    def get_band_frequency(self, band: int) -> float:

        if self.band_centres_mhz is not None:
            return float(self.band_centres_mhz[band])

        return (
            self.config.band_start_mhz
            + band * self.config.band_width_mhz
        )

    def get_summary(self) -> Dict[str, Any]:

        total_transmissions = int(
            np.sum(self.truth)
        )

        occupied_slots = int(
            np.sum(
                np.any(
                    self.truth == 1,
                    axis=1
                )
            )
        )

        occupancy = (
            total_transmissions
            / self.truth.size
        )

        return {
            "num_bands": self.config.num_bands,
            "num_time_slots": self.config.num_time_slots,
            "total_transmissions": total_transmissions,
            "occupied_time_slots": occupied_slots,
            "spectrum_occupancy": occupancy,
            "transmission_events": len(
                self.transmission_events
            )
        }
