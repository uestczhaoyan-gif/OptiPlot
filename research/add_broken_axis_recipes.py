"""Hand-written recipes for the broken-axis figure."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("broken_spectrum", "双量程拼起来的宽谱",
     "wavelength_nm, response",
     "两段量程之间确实没有数据时才断轴；断口两侧各展开读数，共用纵轴刻度所以峰高仍可比，"
     "但跨过断口的斜率与间距不再有意义，不要在两瓣之间量任何东西。"),
    ("broken_spectrum", "断口必须可见",
     "wavelength_nm, response",
     "断口标记与省略区间是这张图的必需项而不是装饰：抹掉它们，读者会把两条独立坐标"
     "当成一条连续轴来读，从而在断口两侧编出根本不存在的趋势。"),
    ("broken_spectrum", "不要用断轴压缩想省略的区间",
     "wavelength_nm, response",
     "如果空洞是仪器本来能测而你没测，那是采样问题，应补测或改用连续轴把空白露出来；"
     "只有量程切换这类物理上不可能有数据的地方才值得断轴。"),
    ("broken_spectrum", "多响应列共用断口",
     "wavelength_nm, signal_a, signal_b",
     "所有列在同一处断开，断口位置取自横轴而不是某一列；若某列在空洞一侧不足三个点，"
     "该列那一侧不画而不是把整条轴挪到别处断。"),
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
