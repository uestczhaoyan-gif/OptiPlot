"""Add a vector-network-analyser response so the complex views are reachable.

A high-speed photodetector is characterised by its S21: one frequency sweep, two
columns, linear magnitude and phase in degrees. The magnitude is kept linear
rather than in dB because undoing a dB column needs the factor it was taken with,
and the column name does not say. That is a single complex quantity
stored as two real ones, which is the shape the Bode and complex-plane views
exist for - and the shape the engine previously had no way to notice.

The phase is left wrapped at ±180° on purpose, so the figure has to say it did
not unwrap anything. The roll-off is a first-order low-pass plus a relaxation
shoulder, so the complex-plane trace is an arc plus a second bend rather than a
textbook semicircle.

Values come from a fixed formula plus seeded noise, not from any measurement.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "examples" / "sample_vna_response.csv"

rng = np.random.default_rng(20260927)
frequency = np.round(np.geomspace(0.05, 67.0, 121), 4)  # GHz

# three poles: a carrier-trapping shoulder, the detector's RC roll-off, and the
# bond-pad resonance - together the phase runs past -180 degrees and wraps
response = 1.0 / ((1 + 1j * frequency / 12.0) * (1 + 1j * frequency / 2.2))
response *= 1.0 / (1 + 1j * frequency / 34.0)

amplitude = np.abs(response) * np.exp(rng.normal(0, 0.002, frequency.size))
phase_deg = (np.degrees(np.angle(response)) + rng.normal(0, 0.6, frequency.size) + 180) % 360 - 180

frame = pd.DataFrame(
    {
        "frequency_ghz": frequency,
        "s21_abs": np.round(amplitude, 6),
        "s21_phase_deg": np.round(phase_deg, 2),
    }
)
frame.to_csv(out, index=False)
seams = int(np.sum(np.abs(np.diff(phase_deg)) > 180))
print(
    f"wrote {out.name}: {len(frame)} points, {frequency.min():g}-{frequency.max():g} GHz "
    f"({np.log10(frequency.max() / frequency.min()):.1f} decades), "
    f"{seams} phase wrap seams, |S21| {amplitude.min():.4f} to {amplitude.max():.4f}"
)
