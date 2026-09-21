"""Render the same figure under each named preset, for eyeballing group I."""

from pathlib import Path
import io
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import render
from optiplot.style import Style, STYLE_PRESETS

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets"
out.mkdir(parents=True, exist_ok=True)

profile = analyze_file(root / "examples" / "sample_replicates.csv")
rec = next(r for r in recommend(profile) if r.id == "errorbar")

NOTES = {
    "default": "default · 现状基线",
    "journal": "journal · 9pt 细线 四面框 次刻度 600dpi 紧裁",
    "slide": "slide · 17pt 粗线 两列图例 主网格",
    "poster": "poster · 24pt 4pt 线宽 带框图例",
}

TILES = []
for name in STYLE_PRESETS:
    style = Style.from_preset(name)
    fig = render(profile, rec, style=style)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    fig.clear()
    buf.seek(0)
    TILES.append((NOTES[name], Image.open(buf).convert("RGB")))

# tile sizes differ per preset because the canvas preset differs; scale to match
TARGET_W = 620
TILES = [(n, im.resize((TARGET_W, int(im.height * TARGET_W / im.width)), Image.LANCZOS))
         for n, im in TILES]

W = TARGET_W
H = max(im.height for _, im in TILES)
PAD, CAP = 16, 32
sheet = Image.new("RGB", (2 * (W + PAD) + PAD, 2 * (H + CAP + PAD) + PAD + 54), "#FBFBFA")
draw = ImageDraw.Draw(sheet)
try:
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    cap_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
except OSError:
    title_font = cap_font = ImageFont.load_default()

draw.text((PAD, 14), "OptiPlot · N3a I 样式预设 — 同一份重复测量数据，四个内置预设",
          font=title_font, fill="#1A1A18")
for i, (name, img) in enumerate(TILES):
    r, c = divmod(i, 2)
    x = PAD + c * (W + PAD)
    y = 54 + PAD + r * (H + CAP + PAD)
    sheet.paste(img, (x, y))
    draw.text((x + 2, y + img.height + 8), name, font=cap_font, fill="#55534E")

path = out / "presets_comparison.png"
sheet.save(path)
print("wrote", path, sheet.size)
print("\n每个预设改动了哪些字段（相对 default）：")
base = Style().to_dict()
for name in STYLE_PRESETS:
    diff = {k: v for k, v in Style.from_preset(name).to_dict().items() if base[k] != v}
    print(f"  {name:9s} {len(diff):2d} 项  {sorted(diff)}")
