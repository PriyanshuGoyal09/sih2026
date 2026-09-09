from dataclasses import dataclass
import numpy as np


@dataclass
class ReceiverConfig:
    noise_floor_dbm: float = -100.0
    noise_std_db: float = 2.0
    receiver_bandwidth_mhz: float = 5.0


class NarrowbandReceiver:
    """
    Simulated narrowband electronic-support receiver.

    The receiver observes only the selected frequency band and time slot.
    It does not expose emitter identity or ground truth to the scheduler.
    """

    def __init__(self, environment, config=None, seed=123):
        self.environment = environment
        self.config = config or ReceiverConfig()
        self.rng = np.random.default_rng(seed)

    def observe(self, time_slot: int, band: int) -> dict:
        if not (0 <= time_slot < self.environment.config.num_time_slots):
            raise ValueError("Invalid time slot.")

        if not (0 <= band < self.environment.config.num_bands):
            raise ValueError("Invalid frequency band.")

        # Physical received power from the simulated RF environment.
        signal_power = self.environment.power[time_slot, band]

        # Receiver noise.
        noise = self.rng.normal(
            0.0,
            self.config.noise_std_db
        )

        measured_power = signal_power + noise

        # Approximate noise-relative SNR.
        snr_db = measured_power - self.config.noise_floor_dbm

        return {
            "time_slot": time_slot,
            "band": band,
            "measured_power_dbm": float(measured_power),
            "snr_db": float(snr_db),
        }
