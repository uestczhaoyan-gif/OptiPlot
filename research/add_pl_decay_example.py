"""Add a photoluminescence decay so the semi-log diagnostic has a real case.

A lifetime trace falls by several orders of magnitude, which is exactly the
situation a linear vertical axis handles worst: the last two decades of the decay
are squeezed onto a few pixels and read as a flat tail at zero. The semi-log view
turns the exponential into a straight line, and the constant term underneath it -
the detector's own background - is what makes the line bend at the end, so the
example carries one on purpose. Values come from a fixed formula plus seeded
noise, not from any measurement.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_pl_decay.csv"

rng = np.random.default_rng(20260925)
delay = np.round(np.arange(0.0, 80.0 + 0.5, 0.5), 2)
SIGNAL, LIFETIME_NS, BACKGROUND = 1.0e6, 9.0, 5.0

clean = SIGNAL * np.exp(-delay / LIFETIME_NS) + BACKGROUND
value = clean * (1.0 + rng.normal(0.0, 0.02, delay.size))

frame = pd.DataFrame({"delay_ns": delay, "photoluminescence_au": np.round(value, 4)})
frame.to_csv(out, index=False)
live = frame.photoluminescence_au
print(
    f"wrote {out.name}: {len(frame)} rows, 0-{delay.max():g} ns, "
    f"signal {live.min():.3g}-{live.max():.3g} AU = "
    f"{np.log10(live.max() / live.min()):.2f} decades"
)
