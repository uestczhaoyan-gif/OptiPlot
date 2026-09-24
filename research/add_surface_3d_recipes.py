"""Hand-written recipes for the three-dimensional surface views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("surface_3d", "形状用三维、数值用热图",
     "x_um, y_um, intensity_au",
     "透视投影下高度不能对着刻度读，前面还会挡住后面；这张图回答「形状像什么」，"
     "要取具体数值请回到热图或等高线。"),
    ("surface_3d", "纵向拉伸只改形状",
     "x_um, y_um, intensity_au",
     "z_exaggeration 改的是外框比例而不是 z 值，刻度仍是实测量；"
     "只要不等于 1 就在标题写明，因为此时任何斜率、倾角与锥度的目视判断都失效。"),
    ("surface_3d", "盒型比例不声称真实",
     "x_um, y_um, intensity_au",
     "z 与 x/y 量纲不同，不存在「正确的」长宽高比；外框归一成立方体再加声明的拉伸，"
     "避免把单位选择伪装成样品的几何。"),
    ("surface_3d", "视角必须复核",
     "x_um, y_um, intensity_au",
     "正对某条轴时表面退化成一片、脊与谷互换显眼程度；定稿前用 view_elevation "
     "与 view_azimuth 至少换一个角度确认看到的不是遮挡造成的假象。"),
    ("surface_with_contour", "立体与俯视并置",
     "x_um, y_um, intensity_au",
     "左看形状右读数值，两幅共用同一批格点与同一色标；"
     "等高线一侧按 contour_levels 分层，层数不够时俯视会显得比立体更平滑。"),
    ("surface_with_contour", "不要只留立体那一半",
     "x_um, y_um, intensity_au",
     "把右半裁掉单独发立体图，就失去了这张图存在的理由——"
     "并置是为了承认透视读不出数值。"),
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
