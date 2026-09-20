"""Recount caption evidence per distinct paper, not per caption line.

A paper with 14 band-structure panels is one paper, not fourteen. The first
pass counted lines, which overstates whatever any single group likes to draw.
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

COVERED = {
    "spectrum_lines": r"spectra|spectral|transmission spectrum|reflectance|absorbance|response curve|extinction",
    "scatter_fit": r"scatter|correlat|linear fit|versus|against",
    "density": r"density plot|hexbin",
    "errorbar": r"error bar|standard deviation|uncertaint|\bsem\b",
    "heatmap": r"heatmap|heat map|spatial distribution|map of|distribution of the|image of",
    "contour": r"contour",
    "matrix_heatmap": r"matrix",
    "polar": r"polar(?:ization)?[- ]dependen|angular dependen|polar pattern|radiation pattern",
    "distribution": r"histogram|distribution of",
    "box": r"box ?plot|violin",
    "correlation": r"correlation matrix|pairwise",
    "flow": r"schematic|diagram|flow|illustrates? the (?:setup|system|process)",
    "table": r"\btable\b",
}
CANDIDATE = {
    "device/structure schematic": r"schematic (?:top|cross|side|color|illustration|of)|cross[- ]sectional view|device structure",
    "band structure / dispersion": r"band structure|dispersion (?:relation|curve|diagram)|photonic band",
    "Stokes / polarisation state": r"stokes|degree of polarization|DOP|DOLP|DOCP|polarization state",
    "frequency response / Bode": r"frequency response|bode|transfer function|\b3dB bandwidth\b|cutoff frequency",
    "transient / time-resolved": r"transient|time[- ]resolved|rise time|decay (?:curve|time)|lifetime",
    "energy band diagram": r"band diagram|band alignment|energy level diagram",
    "residual / fit quality": r"residual",
    "Poincare sphere": r"poincar",
    "Mueller matrix": r"mueller",
    "linecut / profile": r"line ?cut|normalized (?:intensity|field) profile|field profile",
    "ROC / confusion": r"confusion matrix|receiver operating|\bauc\b",
}

papers = {r["key"] for r in rows}
print(f"distinct papers contributing captions: {len(papers)}")
print(f"caption lines: {len(rows)}\n")

cov_p = collections.Counter()
cand_p = collections.Counter()
for name, pat in COVERED.items():
    rx = re.compile(pat, re.I)
    cov_p[name] = len({r["key"] for r in rows if rx.search(r["text"])})
for name, pat in CANDIDATE.items():
    rx = re.compile(pat, re.I)
    cand_p[name] = len({r["key"] for r in rows if rx.search(r["text"])})

print("=== 现有图型：有多少篇论文的图注提到它（/ %d 篇）===" % len(papers))
for name, n in cov_p.most_common():
    print(f"  {n:4d}  {name}")

print("\n=== 画不出的表达：按论文数排序 ===")
for name, n in cand_p.most_common():
    print(f"  {n:4d}  {name}")
