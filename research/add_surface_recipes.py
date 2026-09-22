"""Hand-written drawing recipes for the surface views added to group B."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("heatmap_contours", "模式场叠加结构等值线",
     "x_um, y_um, field",
     "颜色保留每个格点的实测值，等值线只标出各等值面的经过位置；"
     "线的位置由相邻格点插值得到，因此不能把两线之间的区域当作已测，"
     "需要数值时看色标而不是数线条。"),
    ("heatmap_contours", "光束质量图的等值线间隔",
     "x_um, y_um, intensity_au",
     "等值线条数由 contour_levels 决定，与色标层数共用；"
     "光束尾部很平，层数一多会在低值区糊成一片，此时应减少层数或改用对数色标。"),
    ("heatmap_contours", "顺序色带上的等值线颜色",
     "x_um, y_um, intensity_au",
     "顺序色带大部分面积偏暗，等值线必须用浅色才看得见；"
     "发散热色带中部偏亮，改用深色。auto 就是按是否跨零来选的，跨零时深色更稳。"),
    ("heatmap_marginals", "光束剖面与二维分布同图",
     "x_um, y_um, intensity_au",
     "上、右两条剖面是沿另一轴求均值，用来判断光束是否散光；"
     "均值把整列起伏压成一个数，看离散程度要回到热图，别用剖面读峰值功率。"),
    ("heatmap_marginals", "为什么只有两条剖面",
     "x_um, y_um, intensity_au",
     "上边与下边是同一条曲线的重复，左边与右边同理，因此只画上与右；"
     "若确实要四边，应让其中一对显示分位数而不是再画一遍均值。"),
    ("heatmap", "缺测格涂灰而不是留白",
     "x_um, y_um, intensity_au",
     "留白与色带低端的颜色可能相同，读者会把没测的地方读成低值；"
     "missing_fill 给缺测格一个不可能属于色带的颜色，前提是色带本身不含该灰阶。"),
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
