"""Render the same optics data under competing palettes so the choice is visual.

Panels: categorical line colours, sequential heatmaps, diverging signed maps,
and a grayscale reprint test (what the journal PDF looks like photocopied).
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets"
out.mkdir(parents=True, exist_ok=True)

FONT = next(
    (f for f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
     if f in {x.name for x in font_manager.fontManager.ttflist}),
    "DejaVu Sans",
)

PALETTES = {
    "Okabe-Ito（当前）": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"],
    "Tol bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"],
    "Tol vibrant": ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988"],
    "Tol muted": ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255"],
}
SEQ = ["viridis", "cividis", "magma", "inferno", "plasma"]
DIV = ["RdBu_r", "PiYG", "coolwarm", "seismic", "PRGn"]

spec = analyze_file(root / "examples" / "sample_spectrum.csv")
df = spec.data
ys = [c for c in df.columns if c != "wavelength_nm"][:3]

beam = analyze_file(root / "examples" / "sample_beam_map.csv")
b = beam.data[["x_um", "y_um", "intensity_au"]].dropna()
grid = b.pivot(index="y_um", columns="x_um", values="intensity_au")
Z = grid.to_numpy(float)
ZX = grid.columns.to_numpy(float)
ZY = grid.index.to_numpy(float)
signed = Z - Z.mean()

style = {
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "savefig.facecolor": "white",
}

with matplotlib.rc_context(style):
    fig = Figure(figsize=(11.5, 13.2), dpi=130, facecolor="white")
    bands = fig.subfigures(4, 1, height_ratios=[1.0, 0.95, 0.95, 1.05], hspace=0.34)

    # ── A 分类色 ───────────────────────────────────────────────────
    bands[0].suptitle("A  分类色 · 多通道光谱曲线（三器件响应，检查红绿色盲下是否可分）",
                      fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    axA = bands[0].subplots(1, 4)
    for ax, (name, cols) in zip(axA, PALETTES.items()):
        for i, col in enumerate(ys):
            d = df[["wavelength_nm", col]].dropna()
            ax.plot(d["wavelength_nm"], d[col], color=cols[i % len(cols)], lw=2.0, label=col)
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("wavelength / nm")
    axA[0].set_ylabel("response / a.u.")
    axA[0].legend(frameon=False, fontsize=8)

    # ── B 顺序色 ───────────────────────────────────────────────────
    bands[1].suptitle("B  顺序色 · 光束强度热图（单峰分布，看高低是否一眼可辨）",
                      fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    axB = bands[1].subplots(1, 5)
    for ax, cmap in zip(axB, SEQ):
        m = ax.pcolormesh(ZX, ZY, Z, shading="auto", cmap=cmap)
        ax.set_title(cmap, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        bands[1].colorbar(m, ax=ax, fraction=0.046, pad=0.03)

    # ── C 发散色 ───────────────────────────────────────────────────
    bands[2].suptitle("C  发散色 · 有符号量 ΔI（已减均值；检查无色是否真的落在 0）",
                      fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    axC = bands[2].subplots(1, 5)
    lim = float(np.nanmax(np.abs(signed)))
    for ax, cmap in zip(axC, DIV):
        m = ax.pcolormesh(ZX, ZY, signed, shading="auto", cmap=cmap,
                          norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
        ax.set_title(cmap, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        bands[2].colorbar(m, ax=ax, fraction=0.046, pad=0.03)

    # ── D 灰度重印检验 ─────────────────────────────────────────────
    bands[3].suptitle("D  灰度检验 · 上排原图，下排转灰度（黑白打印/影印后还能分辨高低吗）",
                      fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.09)
    axD = bands[3].subplots(2, 5, gridspec_kw={"hspace": 0.34})
    for j, cmap in enumerate(SEQ):
        m = axD[0, j].pcolormesh(ZX, ZY, Z, shading="auto", cmap=cmap)
        axD[0, j].set_title(cmap, fontsize=9.5)
        rgb = m.cmap(m.norm(Z))
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        axD[1, j].imshow(gray, origin="lower", aspect="auto", cmap="gray",
                         vmin=Z.min(), vmax=Z.max())
        for i in range(2):
            axD[i, j].set_xticks([]); axD[i, j].set_yticks([])
        axD[1, j].set_ylabel("gray", fontsize=8)

    fig.savefig(out / "palette_comparison.png", facecolor="white")
    print("wrote", out / "palette_comparison.png")

print("\n=== 顺序色带灰度单调性（相邻亮度差；越小越难分辨）===")
for name in SEQ:
    cm = matplotlib.colormaps[name]
    rgb = cm(np.linspace(0, 1, 256))
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    d = np.abs(np.diff(lum))
    verdict = "灰度单调" if d.min() > 1e-4 else "存在不可分辨区段"
    print(f"{name:10s} mean|dL|={d.mean():.4f}  min|dL|={d.min():.5f}  -> {verdict}")
