"""Add a laser LIV characterisation sample so dual_axis is reachable.

Current versus voltage and versus optical power on one sweep is among the most
common optoelectronic measurements, and no bundled example had two responses in
different units sharing an axis. Fixed seed; not any device's real data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_liv_sweep.csv"

rng = np.random.default_rng(20260921)
current = np.round(np.linspace(0.0, 60.0, 61), 2)  # mA

# sub-threshold leakage, then a kink near threshold, then a linear slope
voltage = 0.9 + 0.0025 * current + 0.055 * np.log1p(np.exp((current - 12.0) / 3.0))
power = np.where(current <= 12.0, 0.004 * current, 0.115 * (current - 12.0))
power = power + rng.normal(0.0, 0.012, power.size)
power = np.clip(power, 0.0, None)
# a detector saturation roll-off at high drive, so the curve is not perfectly linear
power = power * (1.0 - 0.0022 * np.clip(current - 40.0, 0.0, None))

frame = pd.DataFrame(
    {
        "drive_current_mA": current,
        "forward_voltage_V": np.round(voltage, 4),
        "optical_power_mW": np.round(power, 4),
    }
)
frame.to_csv(out, index=False)
print(f"wrote {out.name}: {len(frame)} rows")
print("units:", ["V", "mW"], "-> the two responses are not the same quantity")
