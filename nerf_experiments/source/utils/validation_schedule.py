"""Generates a schedule for iterations to render validation frames on."""

from __future__ import annotations
import numpy as np

def build_validation_steps(
    max_steps: int,
    *,
    base_every: int | None = None,
    num_val_steps: int | None = None,
    schedule: str = "power",
    power: float = 2.0,
) -> list[int]:
    """
    Returns a strictly increasing list of validation steps that ALWAYS ends at max_steps.

    Modes:
      - base_every: fixed-interval validations; ensures max_steps is included.
      - num_val_steps + schedule='power': dense-early schedule with exponent 'power'.
        (power=1 → uniform; power>1 → early-heavy)

    Notes:
      - If num_val_steps > max_steps, we clamp to max_steps (one per step).
      - All steps are in [1, max_steps], strictly increasing, last == max_steps.
    """
    S = int(max_steps)

    # --- Fixed interval mode ---
    if base_every is not None and base_every > 0:
        steps = list(range(int(base_every), S + 1, int(base_every)))
        if not steps or steps[-1] != S:
            steps.append(S)
        return steps

    # --- Power-biased mode (requires num_val_steps) ---
    if num_val_steps is None or num_val_steps <= 0:
        # sensible default: ~100 validations, uniform
        num_val_steps = min(100, S)

    E = int(num_val_steps)
    if E > S:
        E = S  # at least 1 step gap between validations

    if schedule.lower() != "power":
        # Fallback: uniform
        gap = S / E
        steps = [max(1, int(round((i + 1) * gap))) for i in range(E)]
        steps[-1] = S
        # ensure strictly increasing
        for k in range(1, E):
            steps[k] = max(steps[k], steps[k - 1] + 1)
        steps[-1] = S
        return steps

    # ---- Power schedule via integer gap allocation ----
    # Continuous target CDF: t(j) = (j/E)^power, j=0..E
    # Target fractional gaps (sum to 1): w_j = t(j) - t(j-1)

    j = np.arange(0, E + 1, dtype=np.float64)  # 0..E
    t = (j / E) ** float(power)
    w = np.diff(t)  # shape (E,), non-negative, sum ~ 1.0

    # Convert to integer gaps that sum EXACTLY to S with each gap >= 1
    base_gaps = np.ones(E, dtype=np.int64)             # ensures strictly increasing steps
    remaining = S - E                                  # leftover to distribute (could be 0)
    if remaining > 0:
        raw = w * remaining                            # fractional allocation
        floor_part = np.floor(raw).astype(np.int64)
        rem = int(remaining - int(floor_part.sum()))
        frac = raw - floor_part
        # Give the 'rem' extra units to the largest fractional parts
        if rem > 0:
            idx = np.argsort(frac)[-rem:]
            floor_part[idx] += 1
        gaps = base_gaps + floor_part
    else:
        # E == S → exactly one step between validations
        gaps = base_gaps

    # Cumulative sum → validation steps (strictly increasing, last == S)
    steps = np.cumsum(gaps)
    steps[-1] = S
    return steps.tolist()
