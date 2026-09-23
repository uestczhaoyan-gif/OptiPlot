"""Add a multi-turn rotation sample so polar_and_cartesian is reachable.

A polarimeter that keeps rotating past 360° produces exactly this shape: the
second turn lands on the same rays as the first, and only the unwrapped view
shows that the drift between them is real rather than a second lobe. Values
come from a fixed formula, not from any instrument or paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_multi_turn.csv"

angle = np.arange(0.0, 720.0 + 1e-9, 3.0)
# a four-lobe pattern that decays slightly over the second turn
turn = angle / 360.0
signal = (1.0 - 0.18 * turn) * (0.35 + 0.65 * np.cos(np.deg2rad(2 * angle)) ** 2)
frame = pd.DataFrame(
    {
        "rotator_angle_deg": angle,
        "detected_power_W": np.round(signal, 6),
    }
)
frame.to_csv(out, index=False)
print(f"wrote {out.name}: {len(frame)} rows, span {angle.min():g}-{angle.max():g} deg")
