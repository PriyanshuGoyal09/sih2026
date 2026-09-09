from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.detector import SignalDetector, DetectorConfig


def main():
    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        seed=42
    )

    emitters = [
        Emitter(
            emitter_id="E01",
            name="Fixed",
            center_frequency=122.5,
            bandwidth=5,
            power_dbm=-55,
            behavior="fixed",
            start_band=4
        ),
        Emitter(
            emitter_id="E02",
            name="Intermittent",
            center_frequency=152.5,
            bandwidth=5,
            power_dbm=-60,
            behavior="intermittent",
            duty_cycle=0.35,
            start_band=10
        ),
        Emitter(
            emitter_id="E04",
            name="Agile",
            center_frequency=162.5,
            bandwidth=5,
            power_dbm=-50,
            behavior="agile",
            agility=3,
            start_band=12
        ),
    ]

    environment = RFEnvironment(config, emitters)

    # Generate the RF scenario before the receiver starts observing it.
    environment.reset()

    receiver = NarrowbandReceiver(
        environment,
        ReceiverConfig(
            noise_floor_dbm=-100,
            noise_std_db=2
        ),
        seed=123
    )

    detector = SignalDetector(
        DetectorConfig(
            threshold_dbm=-85,
            false_alarm_probability=0.02
        ),
        seed=456
    )

    print("\n=== SMARTSCAN NARROWBAND RECEIVER ===\n")

    # Test a mixture of occupied and apparently empty bands.
    tests = [
        (0, 4),    # likely fixed emitter
        (0, 0),    # empty
        (1, 12),   # agile emitter position at t=1
        (5, 10),   # possible intermittent emitter
        (10, 3),   # likely empty
        (15, 4),   # fixed emitter
    ]

    for time_slot, band in tests:
        observation = receiver.observe(time_slot, band)
        result = detector.detect(observation)

        print(
            f"t={time_slot:02d} | "
            f"band={band:02d} | "
            f"power={result['measured_power_dbm']:7.2f} dBm | "
            f"SNR={result['snr_db']:6.2f} dB | "
            f"detected={str(result['detected']):5s} | "
            f"false_alarm={result['false_alarm']}"
        )


if __name__ == "__main__":
    main()
