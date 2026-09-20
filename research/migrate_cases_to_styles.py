"""One-time migration: catalog/cases.json -> catalog/styles.json.

Drops every per-record provenance field. What survives is the part the product
actually uses: which figure type, what columns it needs, and how to draw it.
Kept so collaborators can see where styles.json came from.
"""

from pathlib import Path
import json
import collections

root = Path(__file__).resolve().parents[1]
src = root / "catalog" / "cases.json"
dst = root / "catalog" / "styles.json"

raw = json.loads(src.read_text(encoding="utf-8-sig"))
dropped = collections.Counter()
seen = set()
styles = []

for case in raw:
    for field in case:
        if field not in {"label", "patterns", "data_schema", "recipe"}:
            dropped[field] += 1
    label = (case.get("label") or "").strip()
    recipe = (case.get("recipe") or "").strip()
    schema = (case.get("data_schema") or "").strip()
    patterns = [
        "flow" if p == "flowchart" else p for p in case.get("patterns", [])
    ]  # render.py's id for flow diagrams is "flow"
    if not (label and recipe and schema and patterns):
        dropped["<skipped: missing a required field>"] += 1
        continue
    key = (label, recipe, schema, tuple(patterns))
    if key in seen:
        dropped["<skipped: exact duplicate>"] += 1
        continue
    seen.add(key)
    styles.append(
        {"label": label, "patterns": sorted(patterns), "data_schema": schema, "recipe": recipe}
    )

# stable, content-neutral ids: grouped by primary pattern so ids read as
# "spectrum_lines-01" rather than leaking the paper they came from
by_pattern = collections.defaultdict(list)
for s in styles:
    by_pattern[s["patterns"][0]].append(s)
out = []
for pattern in sorted(by_pattern):
    for i, s in enumerate(sorted(by_pattern[pattern], key=lambda x: x["label"]), 1):
        out.append({"id": f"{pattern}-{i:02d}", **s})

dst.write_text(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
)

print(f"{len(raw)} cases -> {len(out)} styles")
print("dropped fields:", dict(dropped))
print("\nper figure type:")
for pattern, items in sorted(collections.Counter(s['patterns'][0] for s in out).items()):
    print(f"  {pattern:16s} {items}")
print(f"\nstyles.json {dst.stat().st_size/1024:.1f} KB "
      f"(cases.json was {src.stat().st_size/1024:.1f} KB)")
