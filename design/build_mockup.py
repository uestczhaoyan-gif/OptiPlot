"""Assemble mockup data and inject it into the HTML template."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend

here = Path(__file__).resolve().parent
root = here.parent
idx = json.loads((here / "assets" / "index.json").read_text(encoding="utf-8"))

# refresh encodings straight from the engine
for key, meta in idx.items():
    ds, kind = key.split("__")
    profile = analyze_file(root / "examples" / f"{ds}.csv")
    rec = next((r for r in recommend(profile) if r.id == kind), None)
    meta["encodings"] = dict(rec.encodings) if rec else {}

styles_raw = json.loads((root / "catalog" / "styles.json").read_text(encoding="utf-8"))
sub = {}
for meta in idx.values():
    fid = meta["id"]
    hits = [s for s in styles_raw if fid in s["patterns"]]
    sub[fid] = [
        {k: s.get(k, "") for k in ("id", "label", "data_schema", "recipe")} for s in hits[:3]
    ]

payload = json.dumps({"figs": idx, "cases": sub}, ensure_ascii=False, separators=(",", ":"))
html = (here / "mockup_template.html").read_text(encoding="utf-8")
assert "__DATA__" in html, "template lost its injection token"
(here / "mockup.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")
print("mockup.html", round((here / "mockup.html").stat().st_size / 1024, 1), "KB")
