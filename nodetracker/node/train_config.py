from dataclasses import dataclass, field

@dataclass
class LightningTrainConfig:
    learning_rate: float = field(default=1e-3)
    sched_lr_gamma: float = field(default=1.0)
    sched_lr_step: int = field(default=1)
