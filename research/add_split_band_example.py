"""Add a two-range spectrum sample so the broken-axis figure is reachable.

Measuring one sample on a VIS setup and again on an NIR setup leaves a real hole
between the ranges; no instrument invents points there. That gap is the only
condition under which this project breaks an axis, so the example has to carry
one. Values come from a fixed formula plus seeded noise, not from any measurement.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_split_band.csv"

rng = np.random.default_rng(20260923)
visible = np.round(np.arange(400.0, 700.0 + 1, 5.0), 1)
near_ir = np.round(np.arange(1200.0, 1500.0 + 1, 5.0), 1)


def response(wavelength):
    # one narrow line in the visible band and a broad band in the near infrared
    line = np.exp(-((wavelength - 532.0) ** 2) / (2 * 6.0**2))
    broad = 0.55 * np.exp(-((wavelength - 1350.0) ** 2) / (2 * 90.0**2))
    return 0.04 + line + broad


rows = []
for label, band in (("detector_si", visible), ("detector_ingaas", near_ir)):
    value = response(band) + rng.normal(0.0, 0.006, band.size)
    rows.append(pd.DataFrame({"wavelength_nm": band, "response_A": np.round(value, 6)}))

frame = pd.concat(rows, ignore_index=True)
frame.to_csv(out, index=False)
gaps = np.diff(np.sort(frame.wavelength_nm.unique()))
print(
    f"wrote {out.name}: {len(frame)} rows, "
    f"{frame.wavelength_nm.min():g}-{frame.wavelength_nm.max():g} nm, "
    f"widest gap {gaps.max():g} nm vs median {np.median(gaps):g} nm"
)
