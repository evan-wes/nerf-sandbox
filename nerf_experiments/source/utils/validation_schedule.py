from __future__ import annotations
from typing import List, Optional
import math

def _unique_increasing(xs: List[int]) -> List[int]:
    out = []
    last = -1
    for s in sorted(set(xs)):
        if s > last:
            out.append(s)
            last = s
    return out

def build_validation_steps(
    max_steps: int,
    *,
    base_every: Optional[int] = None,
    events: Optional[int] = None,
    strategy: str = "power",      # "power" | "log" | "linear"
    power: float = 2.0,           # >1 → denser early (for "power")
    first_step: Optional[int] = None,
    last_step: Optional[int] = None,
) -> List[int]:
    """
    Return a strictly increasing list of training steps at which to run validation.

    - If `events` is None: infer it from `base_every` (≈ max_steps // base_every),
      else default to ~60.
    - `strategy="power"` with power>1 concentrates events early (recommended).
    - `strategy="log"` densifies early via logspace.
    - `strategy="linear"` is the baseline equally-spaced cadence.

    We clamp to [first_step, last_step] and ensure uniqueness/monotonicity.
    """
    if max_steps <= 0:
        return []

    if events is None:
        if base_every is not None and base_every > 0:
            events = max(1, max_steps // base_every)
        else:
            events = 60  # sensible default

    first = int(first_step if first_step is not None else max(1, (max_steps // (events * 2))))
    last  = int(last_step if last_step is not None else max_steps)
    last  = max(first, min(last, max_steps))

    E = int(max(1, events))
    steps: List[int] = []

    if strategy.lower() == "power":
        # s_k = first + (last-first) * (k/E)^power, k=1..E
        for k in range(1, E + 1):
            frac = (k / float(E)) ** float(power)
            s = int(round(first + (last - first) * frac))
            steps.append(s)

    elif strategy.lower() == "log":
        # logspace between first..last (in base-e)
        lo = math.log(max(1.0, float(first)))
        hi = math.log(float(last))
        for k in range(1, E + 1):
            frac = k / float(E)
            s = int(round(math.exp(lo + (hi - lo) * frac)))
            steps.append(s)

    else:  # "linear"
        inc = (last - first) / float(E)
        for k in range(1, E + 1):
            s = int(round(first + inc * k))
            steps.append(s)

    steps = [min(max(1, s), max_steps) for s in steps]
    steps = _unique_increasing(steps)
    if steps and steps[-1] != last:
        steps[-1] = last
    elif not steps:
        steps = [last]
    return steps
