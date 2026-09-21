"""Assemble mockup data (workbench candidates + figure library) and inject it."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import FIGURE_TYPES

here = Path(__file__).resolve().parent
root = here.parent
idx = json.loads((here / "assets" / "index.json").read_text(encoding="utf-8"))

TITLES, EXAMPLES = {}, {}
for path in sorted((root / "examples").glob("*.csv")):
    profile = analyze_file(path)
    for r in recommend(profile):
        TITLES.setdefault(r.id, r.title)
        EXAMPLES.setdefault(r.id, path.stem)
profile = analyze_file(root / "examples" / "sample_matrix.npy")
for r in recommend(profile):
    TITLES.setdefault(r.id, r.title)
    EXAMPLES.setdefault(r.id, "sample_matrix")

# refresh encodings straight from the engine so the mapping controls are real
for key, meta in idx.items():
    ds, kind = key.split("__")
    prof = analyze_file(root / "examples" / f"{ds}.csv")
    rec = next((r for r in recommend(prof) if r.id == kind), None)
    meta["encodings"] = dict(rec.encodings) if rec else {}

styles = json.loads((root / "catalog" / "styles.json").read_text(encoding="utf-8"))
lib = {}
for kind in FIGURE_TYPES:
    png = here / "assets" / "gallery" / f"{kind}.png"
    if not png.exists():
        continue
    lib[kind] = {
        "preview": f"assets/gallery/{kind}.png",
        "title": TITLES.get(kind, kind),
        "example": EXAMPLES.get(kind, ""),
        "subtypes": [
            {k: s[k] for k in ("id", "label", "data_schema", "recipe")}
            for s in styles
            if kind in s["patterns"]
        ],
    }

payload = json.dumps(
    {"figs": idx, "lib": lib}, ensure_ascii=False, separators=(",", ":")
)
html = (here / "mockup_template.html").read_text(encoding="utf-8")
assert "__DATA__" in html, "template lost its injection token"
(here / "mockup.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")

total = sum(len(v["subtypes"]) for v in lib.values())
print(f"mockup.html {(here / 'mockup.html').stat().st_size / 1024:.1f} KB")
print(f"图型 {len(lib)}/{len(FIGURE_TYPES)} · 画法条目合计 {total}")
for kind in FIGURE_TYPES:
    n = len(lib.get(kind, {}).get("subtypes", []))
    print(f"  {kind:16s} {n:>3} 种画法" + ("" if kind in lib else "   ← 无预览"))
