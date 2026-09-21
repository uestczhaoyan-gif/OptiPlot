"""Render one dataset under competing legend and colour settings, for eyeballing.

Companion to build_typography_sheet.py and build_axes_sheet.py; covers groups E
and F.
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
    ("默认：Okabe-Ito，图例无边框", Style()),
    ("Okabe-Ito + 带框图例", Style(legend_frame=True, legend_frame_alpha=0.9)),
    ("Okabe-Ito + 图例置右下、两列", Style(legend="lower right", legend_columns=2)),
    ("无图例", Style(legend="none")),
    ("Tol bright", Style(palette="tol_bright")),
    ("Tol vibrant", Style(palette="tol_vibrant")),
    ("Tol muted", Style(palette="tol_muted")),
    ("自定义三色", Style(palette="custom", custom_colors="#1B3A5C,#C0392B,#27AE60")),
    ("图例句柄加长 + 间距放宽", Style(legend_handle_length=3.4, legend_label_spacing=1.1)),
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
COLS, ROWS = 3, 3
PAD, CAP = 14, 30
sheet = Image.new("RGB", (COLS * (W + PAD) + PAD, ROWS * (H + CAP + PAD) + PAD + 54), "#FBFBFA")
draw = ImageDraw.Draw(sheet)
try:
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    cap_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
except OSError:
    title_font = cap_font = ImageFont.load_default()

draw.text(
    (PAD, 14),
    "OptiPlot · N3a E 图例 + F 颜色 — 同一份三通道光谱数据，只换图例与调色板",
    font=title_font,
    fill="#1A1A18",
)
for i, (name, img) in enumerate(TILES):
    r, c = divmod(i, COLS)
    x = PAD + c * (W + PAD)
    y = 54 + PAD + r * (H + CAP + PAD)
    sheet.paste(img, (x, y))
    draw.text((x + 2, y + H + 6), name, font=cap_font, fill="#55534E")

path = out / "legend_colour_comparison.png"
sheet.save(path)
print("wrote", path, sheet.size)
