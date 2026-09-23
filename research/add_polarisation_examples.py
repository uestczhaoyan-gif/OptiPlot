"""Add a Mueller matrix and a Stokes spectrum sample.

Both new types need data whose columns name what they are, since that is the only
signal the recommender is allowed to use to claim a table is a Mueller matrix
rather than four-by-four numbers. The matrix here is the product of two standard
components, computed rather than typed, so its entries are consistent with each
other; the Stokes set varies its degree of polarisation across the band, which
puts points inside the sphere where a real measurement would.
"""

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]

delta = np.pi / 2.0
retarder_45 = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(delta), 0, -np.sin(delta)],
        [0, 0, 1, 0],
        [0, np.sin(delta), 0, np.cos(delta)],
    ]
)
polarizer_0 = 0.5 * np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
matrix = retarder_45 @ polarizer_0

columns = [f"m{i}{j}" for i in range(4) for j in range(4)]
out_matrix = root / "examples" / "sample_mueller.csv"
pd.DataFrame([dict(zip(columns, np.round(matrix.ravel(), 6)))]).to_csv(out_matrix, index=False)
print(f"wrote {out_matrix.name}: 1 row x {len(columns)} elements")

wavelength = np.round(np.linspace(1400.0, 1700.0, 31), 1)
# a polarised peak on an unpolarised background: the degree of polarisation is
# high at the resonance and falls towards the wings
peak = np.exp(-((wavelength - 1550.0) ** 2) / (2 * 60.0**2))
s0 = 1.0 + 2.0 * peak
dop = 0.25 + 0.7 * peak
azimuth = np.deg2rad(20.0 + 0.04 * (wavelength - 1550.0))
ellipticity = np.deg2rad(15.0 * peak)
out_stokes = root / "examples" / "sample_stokes.csv"
stokes = pd.DataFrame(
    {
        "wavelength_nm": wavelength,
        "S0": np.round(s0, 6),
        "S1": np.round(s0 * dop * np.cos(2 * ellipticity) * np.cos(2 * azimuth), 6),
        "S2": np.round(s0 * dop * np.cos(2 * ellipticity) * np.sin(2 * azimuth), 6),
        "S3": np.round(s0 * dop * np.sin(2 * ellipticity), 6),
    }
)
stokes.to_csv(out_stokes, index=False)
radius = np.sqrt(stokes.S1**2 + stokes.S2**2 + stokes.S3**2) / stokes.S0
print(f"wrote {out_stokes.name}: {len(stokes)} rows, DOP {radius.min():.2f}-{radius.max():.2f}")
