"""Add a variable-angle spectroscopy sample so peak_evolution is reachable.

Angle-resolved reflectance maps are a routine optics measurement and no bundled
example covered them. Every value is generated from a fixed seed; this is not
any paper's data, matching the convention of the other examples.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_angle_resolved.csv"

rng = np.random.default_rng(20260921)
theta = np.arange(0.0, 71.0, 5.0)
wavelength = np.round(np.linspace(1300.0, 1800.0, 41), 1)

rows = []
for t in theta:
    # a resonance that blue-shifts with angle and broadens slightly
    centre = 1620.0 - 1.6 * t
    width = 55.0 + 0.45 * t
    depth = 0.62 - 0.004 * t
    curve = 1.0 - depth * np.exp(-((wavelength - centre) ** 2) / (2 * width**2))
    curve = curve + rng.normal(0.0, 0.004, curve.size)
    for w, r in zip(wavelength, curve):
        rows.append({"theta_deg": t, "wavelength_nm": w, "reflectance": round(float(r), 5)})

frame = pd.DataFrame(rows)
# a handful of genuinely missing readings, so the grid is incomplete and the
# recommender has to notice rather than assume a full map
missing = rng.choice(len(frame), size=14, replace=False)
frame.loc[missing, "reflectance"] = np.nan
frame.to_csv(out, index=False)
print(f"wrote {out.name}: {len(frame)} rows, {frame.reflectance.isna().sum()} missing")
print(f"grid complete for peak extraction: {frame.dropna().groupby('theta_deg').size().unique()}")
