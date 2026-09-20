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

cases_raw = json.loads((root / "catalog" / "cases.json").read_text(encoding="utf-8-sig"))
sub = {}
for meta in idx.values():
    fid = meta["id"]
    hits = [c for c in cases_raw if fid in ["flow" if p == "flowchart" else p for p in c.get("patterns", [])]]
    sub[fid] = [
        {k: c.get(k, "") for k in ("label", "figure", "data_schema", "evidence", "source_url")}
        for c in hits[:2]
    ]

payload = json.dumps({"figs": idx, "cases": sub}, ensure_ascii=False, separators=(",", ":"))
html = (here / "mockup_template.html").read_text(encoding="utf-8")
assert "__DATA__" in html, "template lost its injection token"
(here / "mockup.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")
print("mockup.html", round((here / "mockup.html").stat().st_size / 1024, 1), "KB")
