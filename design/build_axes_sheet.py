"""Render one dataset under competing axis settings, for eyeballing.

Companion to build_typography_sheet.py; covers group C.
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

profile = analyze_file(root / "examples" / "sample_replicates.csv")
rec = next(r for r in recommend(profile) if r.id == "errorbar")

CASES = [
    ("默认：左+下脊线，刻度朝外，无网格", Style()),
    ("四条脊线", Style(spines="all")),
    ("脊线加粗 2.2", Style(spines="all", spine_width=2.2)),
    ("刻度朝内", Style(tick_direction="in")),
    ("刻度朝内 + 次刻度", Style(tick_direction="in", tick_minor=True)),
    ("长刻度 11 pt", Style(tick_length=11.0)),
    ("主网格（压在数据下方）", Style(grid="major")),
    ("主+次网格，网格浮在数据上方", Style(grid="both", tick_minor=True, grid_under_data=False)),
    ("科学计数 10²", Style(sci_power=2)),
    ("刻度保留 1 位小数", Style(tick_precision=1)),
    ("X 范围手动 2–8", Style(x_min=2.0, x_max=8.0)),
    ("Y 对称于零（全正数据，慎用）", Style(y_zero_centered=True)),
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
    "OptiPlot · N3a C 坐标轴参数对比 — 同一份重复测量数据、同一图型，只换轴样式",
    font=title_font,
    fill="#1A1A18",
)
for i, (name, img) in enumerate(TILES):
    r, c = divmod(i, COLS)
    x = PAD + c * (W + PAD)
    y = 54 + PAD + r * (H + CAP + PAD)
    sheet.paste(img, (x, y))
    draw.text((x + 2, y + H + 6), name, font=cap_font, fill="#55534E")

path = out / "axes_comparison.png"
sheet.save(path)
print("wrote", path, sheet.size)
