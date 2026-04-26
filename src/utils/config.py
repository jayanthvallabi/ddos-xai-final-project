from dataclasses import dataclass

@dataclass
class TrainConfig:
    test_size: float = 0.3
    random_state: int = 42
    rf_estimators: int = 300
    if_estimators: int = 300
    if_contamination: float = 0.1
