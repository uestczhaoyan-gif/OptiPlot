"""Hand-written recipes for normalised surfaces and threshold masking."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("heatmap_normalized", "消去光谱包络的逐行归一化",
     "wavelength_nm, temperature_K, transmittance",
     "每个温度处沿波长各自标准化，消去随温度变化的整体透过率水平，只比较谱形；"
     "此时颜色不再是实测值，不同温度之间同色不代表同值，色标必须写明口径。"),
    ("heatmap_normalized", "归一化口径要写进图注",
     "x_position_mm, y_position_mm, field_au",
     "行标准化与列标准化回答的是两个不同问题，图上看不出区别；"
     "必须写明按哪个坐标归一、以及消掉了什么，否则读者会拿颜色反推绝对大小。"),
    ("heatmap_normalized", "整行无变化时不画",
     "wavelength_nm, angle_deg, reflectance",
     "标准差为零的行除以零后留空而不是报 0；若图中出现整行空白，"
     "说明该行本来就是平的，不要改成强行填充。"),
    ("heatmap", "阈值屏蔽与缺测分开着色",
     "x_um, y_um, intensity_au",
     "被阈值挡掉的格子与从未测过的格子必须两种颜色：前者说“测了但很小”，"
     "后者说“没有数”。屏蔽数量要写在色标上，否则读者会把灰块当成背景。"),
    ("heatmap", "阈值按实测值判定",
     "x_um, y_um, intensity_au",
     "屏蔽阈值与归一化无关，判的是实测值：设 0.05 就是 5% 的测量量，"
     "不是 5% 的某行标准差；先屏蔽后归一化的顺序会影响结论，不要反过来。"),
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
