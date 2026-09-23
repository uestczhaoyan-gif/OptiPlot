"""Hand-written recipes for the dB and dual-view polar figures."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("polar_db", "天线与散射的 dB 方向图",
     "theta_deg, far_field_W",
     "半径改为相对最大值的 dB，旁瓣与后瓣才从外圈上分离出来；"
     "0 dB 参考值与径向深度必须写在图上，否则读者无法把曲线还原成功率。"),
    ("polar_db", "功率还是场幅决定对数因子",
     "theta_deg, far_field_W",
     "功率量用 10·log10，场幅量用 20·log10，选错是 2 倍标度误差；"
     "列名分不出两者，图上会标出当前用的因子，请核对后必要时改 db_factor。"),
    ("polar_db", "零值与深旁瓣的截断",
     "theta_deg, far_field_W",
     "0 与负值取对数无定义，低于径向范围的点会被丢弃；"
     "数量写在标题里，若占比很大说明径向范围设得太浅而不是数据有问题。"),
    ("polar_and_cartesian", "超过一圈的旋转扫描",
     "rotator_angle_deg, detected_power_W",
     "扫描跨度大于一圈时，不同圈的同一角度落在同一条射线上，"
     "极坐标单独看不出某点属于哪一圈；右侧直角面板保留展开后的顺序。"),
    ("polar_and_cartesian", "同一数据的两种读法",
     "angle_deg, response",
     "极坐标看对称性，直角看峰高与背景比；两者并列时曲线必须同源同序，"
     "不要在任一侧做平滑或插值，否则两张图讲的不是同一批测量。"),
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
