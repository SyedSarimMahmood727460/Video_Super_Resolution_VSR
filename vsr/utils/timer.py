from dataclasses import dataclass

@dataclass
class Stat:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    fps: float

def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(k, len(xs) - 1))]

class LatencyTimer:
    def __init__(self):
        self._times_ms = []

    @property
    def times_ms(self):
        return self._times_ms

    def add_ms(self, t_ms: float):
        self._times_ms.append(float(t_ms))

    def summary(self) -> Stat:
        xs = self._times_ms
        mean = sum(xs) / len(xs) if xs else 0.0
        p50 = _pct(xs, 50)
        p90 = _pct(xs, 90)
        p99 = _pct(xs, 99)
        fps = 1000.0 / mean if mean > 0 else 0.0
        return Stat(mean_ms=mean, p50_ms=p50, p90_ms=p90, p99_ms=p99, fps=fps)
