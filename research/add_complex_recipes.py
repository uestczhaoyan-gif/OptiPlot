"""Hand-written recipes for the complex-frequency-response views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("bode", "两个面板必须共用一条频率轴",
     "frequency_ghz, s21_db, s21_phase_deg",
     "幅度与相位各自成面板时，如果横轴不锁在一起，读者会把相位上某个频率挪到幅度的另一个频率去对齐。"
     "引擎让两个面板共享同一条轴并且只在下方标数字，就是为了断掉这种错读。"),
    ("bode", "相位展开会改掉仪器读数",
     "frequency_ghz, s21_db, s21_phase_deg",
     "缠绕在 ±180° 内的相位是网络分析仪直接给出的数；展开（unwrap）把跳变之后整段平移，"
     "曲线连续了，但绝对值不再是任何一次测量。默认不展开，图上写明检测到几处跳变；"
     "要展开得用户显式勾选。"),
    ("bode", "分贝因子用错就是两倍标度误差",
     "frequency_ghz, h_re, h_im",
     "从实部虚部换算幅度时，取分贝要用 20·log10（场幅）还是 10·log10（功率）取决于这一列是什么。"
     "引擎用 db_factor 并在纵轴标签与说明里写出用的哪个数；列名本身已带 dB 时直接采用，不再换算。"),
    ("nyquist", "等比例不是审美要求",
     "h_re, h_im",
     "实部与虚部同量纲，一张非等比例的框会把一个弛豫半圆拉成椭圆，而椭圆看起来正好像两个弛豫过程。"
     "这张图强制等比例，代价是画布留白。"),
    ("nyquist", "轨迹的方向必须标出来",
     "h_re, h_im",
     "复平面上的线没有横轴可读，起点与终点不标就等于把一条有方向的轨迹交给读者猜。"
     "引擎按数据行序画出实心起点与空心终点，所以行序必须就是频率序。"),
    ("nyquist", "缺读数的地方断开，不补值",
     "s11_re, s11_im",
     "某一行实部或虚部缺失时，轨迹在那里断开而不是连线补一个点：补出来的那段会被读成测量结果，"
     "断掉的那段只说明没测。断开的点数写在图上。"),
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
