"""Hand-written recipes for the Mueller matrix and Poincaré sphere views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("mueller_matrix", "Mueller 矩阵的发散色标",
     "m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33",
     "矩阵元素有正负且符号有物理含义，色标必须以零为中心、对称于最大绝对值；"
     "用顺序色带会把负元素读成「数值小」而不是「反号」。"),
    ("mueller_matrix", "互易性只报偏差不下结论",
     "m00..m33",
     "图上给出 M 与其转置的最大偏差出现在哪一对元素；"
     "该偏差是否意味着样品非互易，取决于材料是否含磁光或手性组分，这是使用者的判断。"),
    ("mueller_matrix", "归一化与否要写明",
     "m00..m33",
     "文献常报 M/M00 的归一化矩阵；本图按实测值着色并在色标注明与 M00 同量纲，"
     "两种口径混用会让消光比一类的读数差几个数量级。"),
    ("poincare_sphere", "偏振态随波长的轨迹",
     "wavelength_nm, S0, S1, S2, S3",
     "以 S0 归一后画在球内，轨迹顺序即波长顺序；"
     "点不强行投影到球面——半径就是偏振度，压到球面等于宣称完全偏振。"),
    ("poincare_sphere", "球外点的处理",
     "wavelength_nm, S0, S1, S2, S3",
     "|S|/S0 > 1 说明 Stokes 参数不自洽（噪声、定标漂移或串扰），"
     "图上按实测量画出并在标题计数，不做归一化修正，否则掩盖了定标问题。"),
    ("poincare_sphere", "视角是数据的一部分",
     "S0, S1, S2, S3",
     "三维轨迹的可读性取决于视角：正对某条轴时球面退化成圆盘、轨迹重叠。"
     "view_elevation 与 view_azimuth 需要按数据调，定稿前换一个角度复核。"),
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
