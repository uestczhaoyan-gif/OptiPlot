"""Add a device x metric table so the heat-table view has a real case.

Comparison tables in this field arrive wide: one row per device, one column per
metric, no sweep axis and no replicates. That shape defeats the usual readings -
a column cannot be plotted "against" another when both are device properties, and
a bar chart of forty unrelated units is meaningless - while it is exactly what a
per-metric colour table answers: which devices are strong on which metric, and
which device is the compromise.

The four metrics do not point the same way (a low dark current is good, a high
bandwidth is good), which is the point: the table has to say that colour is
rank-within-a-metric and nothing more. One cell is left unmeasured on purpose.

Values come from a fixed formula plus seeded noise, not from any measurement.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_device_matrix.csv"

rng = np.random.default_rng(20260926)
n = 14
gain = rng.uniform(0.6, 1.5, n)          # a shared fabrication factor per device
speed = rng.uniform(0.7, 1.4, n)         # independent of gain, which is the tension

frame = pd.DataFrame(
    {
        "device": [f"D{i:02d}" for i in range(1, n + 1)],
        "responsivity_A_W": np.round(0.62 * gain + rng.normal(0, 0.02, n), 4),
        "dark_current_nA": np.round(0.9 / gain + rng.normal(0, 0.05, n), 4),
        "bandwidth_GHz": np.round(28.0 * speed + rng.normal(0, 1.2, n), 3),
        "noise_nep_w_hz12": np.round(2.4e-12 / (gain * speed) + rng.normal(0, 6e-14, n), 15),
    }
)
# one device's noise floor was never characterised; the cell stays empty
frame.loc[5, "noise_nep_w_hz12"] = np.nan

frame.to_csv(out, index=False)
print(
    f"wrote {out.name}: {len(frame)} devices x {len(frame.columns) - 1} metrics, "
    f"one missing cell at row {5}"
)
