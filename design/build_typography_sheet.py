"""Render one dataset under competing typography settings, for eyeballing.

Not a test - this exists so the typography presets can be judged on a real
optics figure instead of a list of numbers. Each case is rendered to its own
PNG by the normal pipeline, then tiled with PIL; no reaching into figure
internals, so what you see is exactly what render() produces.
"""

from pathlib import Path
import io
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optiplot import analyze_file, recommend
from optiplot.render import render
from optiplot.style import Style

root = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent / "assets"
out.mkdir(parents=True, exist_ok=True)

profile = analyze_file(root / "examples" / "sample_beam_map.csv")
rec = next(r for r in recommend(profile) if r.id == "heatmap")

# Matplotlib cannot fall back per glyph, and a CJK font carries Latin shapes, so
# "family" only reaches Latin text when the CJK font is dropped. The two halves
# of this sheet show exactly that split - same data, same figure type.
CASES = [
    ("中文标题 · 雅黑（默认）", Style(), "cjk"),
    ("中文标题 · 宋体 SimSun", Style(cjk="songti"), "cjk"),
    ("中文标题 · 楷体 KaiTi", Style(cjk="kaiti"), "cjk"),
    ("纯英文 · serif Times", Style(family="serif", cjk="none"), "latin"),
    ("纯英文 · mono Consolas", Style(family="mono", cjk="none"), "latin"),
    ("纯英文 · 基准 8 pt", Style(cjk="none", font_size=8), "latin"),
    ("纯英文 · 基准 14 pt", Style(cjk="none", font_size=14), "latin"),
    ("纯英文 · 仅刻度 1.45×", Style(cjk="none", tick_scale=1.45), "latin"),
    ("纯英文 · 标签加粗+留白", Style(cjk="none", label_weight="bold", label_pad=14), "latin"),
]

CJK_TEXT = {
    "title": "光束强度二维分布",
    "xlabel": "横向位置 x / μm",
    "ylabel": "纵向位置 y / μm",
}
LATIN_TEXT = {
    "title": "Beam intensity",
    "xlabel": "transverse position x / um",
    "ylabel": "transverse position y / um",
}
TILES = []
for name, style, mode in CASES:
    options = CJK_TEXT if mode == "cjk" else LATIN_TEXT
    fig = render(profile, rec, style=style.replace(size="single"), options=options)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    fig.clear()
    buf.seek(0)
    TILES.append((name, Image.open(buf).convert("RGB")))

W, H = TILES[0][1].size
COLS, ROWS = 3, 3
PAD, CAP = 14, 30
sheet = Image.new(
    "RGB",
    (COLS * (W + PAD) + PAD, ROWS * (H + CAP + PAD) + PAD + 54 + CAP),
    "#FBFBFA",
)
draw = ImageDraw.Draw(sheet)
try:
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    cap_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
except OSError:
    title_font = cap_font = ImageFont.load_default()

draw.text(
    (PAD, 14),
    "OptiPlot · N3a 排版参数对比 — 同一份光束扫描数据、同一图型，只换样式参数",
    font=title_font,
    fill="#1A1A18",
)
for i, (name, img) in enumerate(TILES):
    r, c = divmod(i, COLS)
    x = PAD + c * (W + PAD)
    y = 54 + PAD + r * (H + CAP + PAD)
    sheet.paste(img, (x, y))
    draw.text((x + 2, y + H + 6), name, font=cap_font, fill="#55534E")

path = out / "typography_comparison.png"
sheet.save(path)
print("wrote", path, sheet.size)

# report the resolved numbers so the visual claim is checkable
print("\n每格实际解析出的字号（pt）与字体栈：")
for name, style, mode in CASES:
    rc = style.rc_params()
    print(
        f"  {name:26s} base={rc['font.size']:5.1f} "
        f"tick={rc['xtick.labelsize']:5.2f} label={rc['axes.labelsize']:5.2f} "
        f"labelpad={rc['axes.labelpad']:4.1f} stack={style.font_stack()}"
    )
