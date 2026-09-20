"""Enumerate the plotting vocabulary in the local corpus - breadth, not ranking.

The question here is not "which of our types appear most often" but "what
quantities get plotted, against what, and using what noun". Every distinct
axis-quantity and every plot-noun is a candidate renderer, so the output is an
inventory to implement against rather than a scoreboard.
"""

from pathlib import Path
import json
import re
import collections

here = Path(__file__).resolve().parents[1]
rows = [
    json.loads(line)
    for line in (here / ".cache" / "zotero_captions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
text = " ".join(r["text"] for r in rows).lower()
per_paper = collections.defaultdict(set)

# 1. What things are plotted (the dependent quantities).
QUANT = re.compile(
    r"\b(?:spectra|spectrum|response|responsivity|detectivity|transmittance|transmission|"
    r"reflectance|absorbance|absorption|emission|quantum yield|efficiency|sensitivity|"
    r"noise equivalent|dark current|photocurrent|current|voltage|power|intensity|"
    r"field amplitude|electric field|magnetic field|near[- ]field|far[- ]field|"
    r"phase|amplitude|polarization|stokes|dolp|dop|docp|degree of polarization|"
    r"ellipticity|angle|wavelength|frequency|bandwidth|resolution|contrast|"
    r"temperature|carrier lifetime|decay|rise time|refractive index|permittivity|"
    r"loss|gain|fom|figure of merit|error|accuracy|confidence|auc|snr|fid)\b"
)
# 2. What it is plotted against (the independent variables).
VS = re.compile(
    r"(?:versus|vs\.?|as a function of|against|against the|in terms of|with increasing|"
    r"as a function)\s+([a-z][a-z0-9 _-]{2,45}?)(?=[,.;:]|\band\b|\bfor\b|\bat\b|$)"
)
# 3. What noun the authors use for the picture itself.
NOUN = re.compile(
    r"\b(spectra|spectrum|trace|traces|curve|curves|plot|plots|panel|panels|"
    r"map|maps|image|images|heatmap|heat map|contour|contours|profile|profiles|"
    r"cross[- ]section|cross[- ]sections|slice|slices|histogram|histograms|"
    r"distribution|distributions|pattern|patterns|diagram|diagrams|schematic|"
    r"schematics|illustration|illustrations|inset|insets|matrix|matrices|"
    r"surface|rendering|reconstruction|reconstructions|scan|scans|scan image|"
    r"eye diagram|constellation|polar plot|rose|smith|band structure|"
    r"dispersion|isotherm|nyquist|bode|waterfall|stacked|tmap|cmap)\b"
)

for r in rows:
    low = r["text"].lower()
    for m in QUANT.finditer(low):
        per_paper["quant:" + m.group(0)].add(r["key"])
    for m in VS.finditer(low):
        v = re.sub(r"\s+", " ", m.group(1)).strip()
        if 3 < len(v) < 45:
            per_paper["axis:" + v].add(r["key"])
    for m in NOUN.finditer(low):
        per_paper["noun:" + m.group(0)].add(r["key"])


def show(prefix, title, limit=999):
    items = [(k.split(":", 1)[1], len(v)) for k, v in per_paper.items() if k.startswith(prefix)]
    items.sort(key=lambda x: -x[1])
    print(f"\n=== {title}（不同论文数，共 {len(items)} 个不同说法）===")
    for name, n in items[:limit]:
        print(f"  {n:4d}  {name}")


print(f"{len(rows)} caption lines from {len({r['key'] for r in rows})} papers")
show("noun:", "图的形式（渲染器候选）— 这是宽度来源")
show("axis:", "被当作自变量的物理量（决定列编码契约）", 40)
show("quant:", "被当作因变量的物理量（决定 Y 轴候选）", 40)
