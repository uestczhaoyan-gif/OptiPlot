"""Add a two-decade wavelength sweep so the scale diagnostics are reachable.

In the quantum limit a photodetector's responsivity is R = eta·lambda·q/hc, so it
is a power law in wavelength with exponent exactly 1: the log-log plot is the
figure that says so, and a straight line there is a claim about eta being flat
across the whole range. A UV-to-far-infrared sweep covers the two decades of
wavelength the diagnostic needs, and the 5.3-6.2 um region is left unmeasured
because that is where the atmosphere and the beamsplitter both cut - a cumulative
curve has to reach across it, which is the honest case to label.

Values come from a fixed formula plus seeded noise, not from any measurement.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_broadband_det.csv"

rng = np.random.default_rng(20260924)
wavelength = np.round(np.logspace(np.log10(250.0), np.log10(25000.0), 64), 1)

# R = eta * lambda * q / h c, with eta held at 0.70 across the range. hc/e is
# 1.2398 eV.um, so R[A/W] = eta * lambda[um] / 1.2398.
QUANTUM_LIMIT = 0.70 / 1.23984  # A/W per micrometre
clean = QUANTUM_LIMIT * (wavelength / 1000.0)
value = clean + rng.normal(0.0, 0.02, wavelength.size) * clean

frame = pd.DataFrame({"wavelength_nm": wavelength, "responsivity_a_w": np.round(value, 6)})
# the atmosphere is opaque and the FTIR beamsplitter has a turnover here, so
# nothing was measured; the rows stay as holes rather than being interpolated
blocked = frame.wavelength_nm.between(5300.0, 6200.0)
frame.loc[blocked, "responsivity_a_w"] = np.nan

frame.to_csv(out, index=False)
decades = np.log10(wavelength.max() / wavelength.min())
print(
    f"wrote {out.name}: {len(frame)} rows, {frame.wavelength_nm.min():g}"
    f"-{frame.wavelength_nm.max():g} nm = {decades:.2f} decades, "
    f"{int(blocked.sum())} holes, R from {frame.responsivity_a_w.min():.3f} to "
    f"{frame.responsivity_a_w.max():.3f} A/W"
)
