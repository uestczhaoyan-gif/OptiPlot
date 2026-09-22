"""Hand-written drawing recipes for the model-fit options.

A recipe states what has to be true for a drawing to be honest. The corpus tells
us which shapes appear in the literature; whether a fit may be reported as a
lifetime is a judgement about measurement, so it is written rather than mined.
"""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("spectrum_lines", "时间分辨衰变的单指数拟合",
     "delay_ps, photoluminescence",
     "勾选单指数后必须同时看残差面板：残差若在后段成系统鼓起，说明衰减不是单指数或基线没扣对，"
     "此时报告出的 τ 是拟合的产物而不是载流子寿命。"),
    ("spectrum_lines", "双指数分量的取舍",
     "delay_ns, intensity",
     "第二个分量只有在单指数残差里出现结构时才值得引入；引入后 τ₂ 落到扫描范围之外"
     "（图注会明说），它实际起的是常数作用，应退回单指数。"),
    ("spectrum_lines", "高斯还是洛伦兹线型",
     "wavelength_nm, transmission",
     "两者在峰顶几乎同形、在两翼分明不同，因此判据只能看残差的两翼符号；"
     "只用 R² 比较会得出反直觉的结论，因为 R² 被峰顶主导。"),
    ("scatter_fit", "Malus 定律与消光比",
     "analyzer_angle_deg, output_power",
     "先确认角度单位（模型按度或弧度参与拟合，选错单位时曲线仍能贴合但阶数会翻倍）；"
     "消光比要由同一轮扫描的最大最小值直接给出，不要用拟合参数相减再相除。"),
    ("spectrum_lines", "折射率色散的振子位置",
     "wavelength_um, refractive_index",
     "单振子色散式只有在数据接近该振子时才能同时定出 n∞、S 与 λ₀；"
     "远离极点时曲线仍然贴合而 λ₀ 会乱跑，图注里的量纲判据就是为这种情况准备的。"),
    ("scatter_fit", "拟合结果的报告口径",
     "x, y",
     "报告参数时同时给出扫描范围与 R²，并注明本项目不给参数置信区间——"
     "协方差假设的是独立同分布高斯噪声，而仪器读数的噪声通常不是。"),
]

existing = {(s["label"], s["recipe"]) for s in styles}
by_prefix = {
    p: sum(1 for s in styles if s["patterns"][0] == p) for p in {s["patterns"][0] for s in styles}
}
added = 0
for pattern, label, schema, recipe in NEW:
    if (label, recipe) in existing:
        continue
    by_prefix[pattern] = by_prefix.get(pattern, 0) + 1
    styles.append(
        {
            "id": f"{pattern}-{by_prefix[pattern]:02d}",
            "label": label,
            "patterns": [pattern],
            "data_schema": schema,
            "recipe": recipe,
        }
    )
    added += 1

styles.sort(key=lambda s: (s["patterns"][0], s["id"]))
styles_path.write_text(
    json.dumps(styles, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
)
print(f"added {added} recipes; total {len(styles)}")
