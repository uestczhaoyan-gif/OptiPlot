"""Render one dataset under competing data-series settings, for eyeballing.

Companion to the typography, axes and legend sheets; covers group D.
"""

from pathlib import Path
import io
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import render
from optiplot.style import Style

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets"
out.mkdir(parents=True, exist_ok=True)

profile = analyze_file(root / "examples" / "sample_spectrum.csv")
rec = next(r for r in recommend(profile) if r.id == "spectrum_lines")

CASES = [
    ("默认：1.65 pt，线型循环，无标记", Style()),
    ("加粗 4 pt", Style(line_width=4.0)),
    ("细线 0.6 pt", Style(line_width=0.6)),
    ("线型不循环（只靠颜色区分）", Style(vary_line_style=False)),
    ("实心圆标记 6 pt", Style(marker="o", marker_size=6.0)),
    ("空心方块 8 pt", Style(marker="s", marker_size=8.0, marker_fill="none")),
    ("标记抽稀：每 10 点一个", Style(marker="o", marker_every=10)),
    ("阶梯线（离散调谐/步进数据）", Style(step=True)),
    ("只有点、不连线", Style(show_lines=False, show_points=True, marker="o")),
    ("半透明 0.35", Style(series_alpha=0.35)),
    ("三角标记 + 加粗 + 不循环", Style(marker="^", line_width=2.6, vary_line_style=False)),
    ("左半填充标记（Tol 调色板）", Style(marker="s", marker_size=11, marker_fill="left",
                                          palette="tol_bright")),
]

TILES = []
for name, style in CASES:
    fig = render(profile, rec, style=style.replace(size="single"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    fig.clear()
    buf.seek(0)
    TILES.append((name, Image.open(buf).convert("RGB")))

W, H = TILES[0][1].size
COLS, ROWS = 3, 4
PAD, CAP = 14, 30
sheet = Image.new(
    "RGB", (COLS * (W + PAD) + PAD, ROWS * (H + CAP + PAD) + PAD + 54 + CAP), "#FBFBFA"
)
draw = ImageDraw.Draw(sheet)
try:
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    cap_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
except OSError:
    title_font = cap_font = ImageFont.load_default()

draw.text(
    (PAD, 14),
    "OptiPlot · N3a D 数据系列 — 同一份三通道光谱数据，只换线宽/线型/标记",
    font=title_font,
    fill="#1A1A18",
)
for i, (name, img) in enumerate(TILES):
    r, c = divmod(i, COLS)
    x = PAD + c * (W + PAD)
    y = 54 + PAD + r * (H + CAP + PAD)
    sheet.paste(img, (x, y))
    draw.text((x + 2, y + H + 6), name, font=cap_font, fill="#55534E")

path = out / "series_comparison.png"
sheet.save(path)
print("wrote", path, sheet.size)
