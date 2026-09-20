"""Render real figures with the existing engine so the UI mockup shows truth."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import render

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets"
out.mkdir(parents=True, exist_ok=True)

wanted = {
    "sample_spectrum.csv": ["spectrum_lines", "scatter_fit", "correlation", "distribution"],
    "sample_beam_map.csv": ["heatmap", "contour"],
    "sample_replicates.csv": ["errorbar", "box"],
    "sample_polarization.csv": ["polar"],
    "sample_devices.csv": ["box"],
    "sample_dense_scatter.csv": ["density"],
    "sample_workflow.csv": ["flow"],
    "sample_explicit_error.csv": ["errorbar"],
}

index = {}
for fname, kinds in wanted.items():
    profile = analyze_file(root / "examples" / fname)
    recs = {r.id: r for r in recommend(profile)}
    for kind in kinds:
        rec = recs.get(kind)
        if rec is None:
            print(f"  skip {fname}:{kind} (not recommended)")
            continue
        name = f"{fname.split('.')[0]}__{kind}"
        try:
            render(profile, rec, out / f"{name}.svg", options={"size": "double"}).clear()
            index[name] = {"file": f"assets/{name}.svg", "title": rec.title,
                           "tier": rec.tier, "tier_label": rec.tier_label,
                           "reason": rec.reason, "id": kind,
                           "encodings": dict(rec.encodings)}
            print(f"  ok   {name}")
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")

(out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
print("total:", len(index))
