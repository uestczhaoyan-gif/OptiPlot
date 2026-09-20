"""Mine figure captions from the local Zotero full-text cache.

Reads only. Zotero already stores a plain-text extraction of each PDF as
`.zotero-ft-cache`, so this never parses a PDF and never writes into the
Zotero directory.

Output discipline: caption text is copyrighted. The raw capture goes to
.cache/zotero_captions.jsonl, which is gitignored. Only aggregate counts and
candidate figure-type names — facts and techniques, not prose — are meant to
reach the repository, and only after a human rewrites them as recipes.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import collections

ZOTERO = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\PC\Zotero\storage")
OUT = Path(__file__).resolve().parents[1] / ".cache" / "zotero_captions.jsonl"

# Fig. 1 | Design of ...            (Nature family style titles)
TITLE = re.compile(r"^\s*(?:Fig(?:ure|\.)\s*(\d+)([a-z])?)\s*[|｜]\s*(.{6,180})$", re.M)
# Figure 3: The measured ...        /  Fig. 2a-b. Absorption spectra ...
BODY = re.compile(
    r"^\s*Fig(?:ure|\.)\s*\d+[a-z]?(?:[-–,]\s*\d*[a-z])*\s*[:.]\s+(.{30,600})$", re.M
)
# ... Figure 2d shows the absorption distribution of ...
SAYS = re.compile(
    r"Fig(?:ure|\.)\s*\d+[a-z]?\s+(?:a|b|c|d|e|f|g|h|and|to)?\s*"
    r"(?:shows?|plots?|depicts?|illustrates?|summariz\w+|present\w+|reports?|"
    r"display\w+|compares?|maps?|gives?|exhibits?)\s+(.{12,180}?)[.;]",
    re.I,
)

# What our 13 renderers already cover, as recognisable caption vocabulary.
COVERED = {
    "spectrum_lines": r"spectra|spectral|transmission spectrum|reflectance|absorbance|response curve|extinction",
    "scatter_fit": r"scatter|correlat|linear fit|versus|against",
    "density": r"density plot|hexbin|contour of the distrib",
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
COVERED_RE = {k: re.compile(v, re.I) for k, v in COVERED.items()}

# Caption vocabulary that our renderers cannot produce. Each hit is a candidate
# new figure type, counted rather than asserted.
CANDIDATE = {
    "waterfall / stacked spectra": r"waterfall|stacked spectr|series of spectra|offset (?:the )?spectra|3d plot of the spectr",
    "far-field radiation pattern (dB)": r"radiation pattern|far[- ]field pattern|directivity|back[- ]to[- ]front",
    "Poincare sphere": r"poincar[eé]",
    "Smith chart / impedance": r"smith chart|impedance matc|admittance",
    "band structure / dispersion": r"band structure|dispersion (?:relation|curve|diagram)|photonic band|\bk.?\b.?[\u03c9w]",
    "energy band diagram": r"band diagram|energy level diagram|band alignment",
    "Bode / frequency response": r"frequency response|bode|transfer function|\bdB\b.*phase|phase.*\bdB\b",
    "cumulative distribution": r"cumulative distribution|\bcdf\b",
    "ROC / confusion matrix": r"receiver operating|confusion matrix|\bauc\b",
    "dual-axis overlay": r"dual axis|second y[- ]axis|left and right ax",
    "residual / fit quality": r"residual",
    "vector field / arrow plot": r"vector field|arrow plot|electric field vector|poynting vector",
    "cross-section / linecut": r"cross[- ]section|line ?cut|slice along",
    "3D surface": r"3d surface|surface plot|three[- ]dimensional (?:map|plot)",
    "bar chart with error": r"bar (?:chart|graph|plot)",
    "pie / donut": r"pie chart|donut",
    "time series": r"time[- ]resolved|transient|time trace|dynamics of",
    "image grid / montage": r"montage|image grid|gallery of",
    "sankey / loss budget": r"sankey|loss budget|loss decomposition",
    " Mueller matrix": r"mueller",
    "jones / stokes vector": r"stokes|jones vector",
    "em / t-sne embedding": r"t[- ]sne|umap|embedding",
    "radar / spider": r"radar (?:chart|plot)|spider",
    "survival / reliability": r"survival (?:curve|plot)|weibull",
    "phase diagram": r"phase diagram|stability diagram",
    "nyquist": r"nyquist",
}
CANDIDATE_RE = {k: re.compile(v, re.I) for k, v in CANDIDATE.items()}


def main() -> int:
    if not ZOTERO.is_dir():
        print(f"not found: {ZOTERO}", file=sys.stderr)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)

    caches = sorted(ZOTERO.glob("*/.zotero-ft-cache"))
    print(f"{len(caches)} full-text caches under {ZOTERO}")

    covered = collections.Counter()
    cand = collections.Counter()
    cand_examples = collections.defaultdict(list)
    total_captions = 0
    papers_with = 0
    unreadable = 0

    with OUT.open("w", encoding="utf-8") as sink:
        for cache in caches:
            try:
                text = cache.read_text(encoding="utf-8", errors="replace")
            except OSError:
                unreadable += 1
                continue
            key = cache.parent.name

            titles = [(m.group(1), m.group(2), m.group(3).strip()) for m in TITLE.finditer(text)]
            bodies = [m.group(1).strip() for m in BODY.finditer(text)]
            says = [m.group(1).strip() for m in SAYS.finditer(text)]

            lines = [t[2] for t in titles] + bodies + says
            if not lines:
                continue
            papers_with += 1
            total_captions += len(lines)

            for ln in lines:
                hit = [name for name, rx in COVERED_RE.items() if rx.search(ln)]
                if hit:
                    covered[hit[0]] += 1
                for name, rx in CANDIDATE_RE.items():
                    if rx.search(ln):
                        cand[name] += 1
                        if len(cand_examples[name]) < 4:
                            cand_examples[name].append({"key": key, "text": ln[:200]})

            for ln in lines:
                sink.write(json.dumps({"key": key, "text": ln}, ensure_ascii=False) + "\n")

    print(f"papers with captions: {papers_with} | caption lines: {total_captions} | unreadable: {unreadable}")
    print(f"raw capture -> {OUT}  (gitignored; contains copyrighted prose)\n")

    print("=== 现有 13 类图型在图注中的出现频次（说明样例覆盖是否偏科）===")
    for name in COVERED:
        print(f"  {name:16s} {covered.get(name, 0)}")

    print("\n=== 当前渲染器画不出来的表达，按图注命中次数排序 ===")
    for name, n in cand.most_common():
        print(f"  {n:5d}  {name}")
    if not cand:
        print("  (无命中)")

    print("\n=== 排名靠前的候选：图注片段样本（仅供你判断，不入库）===")
    for name, n in cand.most_common(8):
        print(f"\n-- {name}  ({n})")
        for ex in cand_examples[name][:3]:
            print(f"   [{ex['key']}] {ex['text'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
