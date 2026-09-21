"""One canonical preview per figure type, for the figure-library view.

Each preview is drawn from a bundled example whose profile actually triggers
that type, so the gallery cannot advertise a type the recommender would never
offer on the shipped data.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import render, FIGURE_TYPES

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets" / "gallery"
out.mkdir(parents=True, exist_ok=True)

# ordered so the cheapest, most common optics cases come first; anything else
# that appears in examples/ is appended automatically, because a preview list
# someone has to remember to update is how a registered type ships with no image
PREFERRED = [
    "sample_spectrum",
    "sample_beam_map",
    "sample_polarization",
    "sample_replicates",
    "sample_explicit_error",
    "sample_devices",
    "sample_dense_scatter",
    "sample_workflow",
]
DATASETS = PREFERRED + sorted(
    {p.stem for p in (root / "examples").glob("*.csv")} - set(PREFERRED)
)

profiles = {}
for name in DATASETS:
    try:
        profiles[name] = analyze_file(root / "examples" / f"{name}.csv")
    except Exception as exc:
        print(f"  ! {name}: {exc}")

npy = analyze_file(root / "examples" / "sample_matrix.npy")
profiles["sample_matrix"] = npy

made, missing = {}, []
for kind in FIGURE_TYPES:
    hit = None
    for name, profile in profiles.items():
        rec = next((r for r in recommend(profile) if r.id == kind), None)
        if rec is not None:
            hit = (name, rec)
            break
    if hit is None:
        missing.append(kind)
        continue
    name, rec = hit
    try:
        render(profiles[name], rec, out / f"{kind}.png", options={"size": "double"}).clear()
        made[kind] = {
            "preview": f"assets/gallery/{kind}.png",
            "title": rec.title,
            "example": name,
            "encodings": {k: v for k, v in rec.encodings.items() if k != "columns"},
        }
        print(f"  ok   {kind:16s} <- {name}")
    except Exception as exc:
        missing.append(kind)
        print(f"  FAIL {kind}: {exc}")

print(f"\n{len(made)}/{len(FIGURE_TYPES)} figure types previewed")
if missing:
    print("MISSING:", missing)
