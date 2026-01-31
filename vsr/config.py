from dataclasses import dataclass

@dataclass
class InferConfig:
    scale: int = 2
    window: int = 5
    precision: str = "amp"  # "fp32" or "amp"
    channels_last: bool = True
    compile: bool = False
    device: str = "cuda"
