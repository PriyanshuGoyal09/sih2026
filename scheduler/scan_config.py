from dataclasses import dataclass


@dataclass
class ScanConfig:
    scans_per_time_slot: int = 4
    dwell_time_ms: float = 1.0
