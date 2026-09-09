import numpy as np

from environment.emitter import Emitter
from environment.rf_environment import (
    RFEnvironment,
    RFEnvironmentConfig
)
from environment.noise import NoiseModel


def create_environment():

    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        band_start_mhz=100.0,
        band_width_mhz=5.0,
        seed=42
    )

    emitters = [

        # Always transmitting.
        Emitter(
            emitter_id="E01",
            name="Fixed emitter",
            center_frequency=120.0,
            bandwidth=5.0,
            power_dbm=-55.0,
            behavior="fixed",
            start_band=4
        ),

        # Intermittent emitter.
        Emitter(
            emitter_id="E02",
            name="Intermittent emitter",
            center_frequency=150.0,
            bandwidth=5.0,
            power_dbm=-60.0,
            behavior="intermittent",
            duty_cycle=0.35,
            start_band=10
        ),

        # Another intermittent source.
        Emitter(
            emitter_id="E03",
            name="Low duty emitter",
            center_frequency=175.0,
            bandwidth=5.0,
            power_dbm=-65.0,
            behavior="intermittent",
            duty_cycle=0.20,
            start_band=15
        ),

        # Frequency-agile emitter.
        Emitter(
            emitter_id="E04",
            name="Agile emitter",
            center_frequency=160.0,
            bandwidth=5.0,
            power_dbm=-50.0,
            behavior="agile",
            agility=3,
            start_band=12
        ),

        # Burst emitter.
        Emitter(
            emitter_id="E05",
            name="Bursty emitter",
            center_frequency=135.0,
            bandwidth=5.0,
            power_dbm=-58.0,
            behavior="bursty",
            duty_cycle=0.15,
            start_band=7
        )
    ]

    noise = NoiseModel(
        noise_floor_dbm=-100.0,
        noise_std_db=2.0,
        false_alarm_probability=0.02
    )

    return RFEnvironment(
        config=config,
        emitters=emitters,
        noise_model=noise
    )


def main():

    env = create_environment()

    env.reset()

    print("\n=== SMARTSCAN RF ENVIRONMENT ===\n")

    print("Environment summary:")
    print(env.get_summary())

    print("\nFirst 20 time slots:\n")

    for t in range(20):

        observation = env.step()

        occupied = np.where(
            observation["truth"] == 1
        )[0]

        print(
            f"t={observation['time_slot']:02d} | "
            f"occupied bands={occupied.tolist()} | "
            f"emitters={observation['active_emitters']}"
        )


if __name__ == "__main__":
    main()
