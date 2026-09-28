"""Hand-written recipes for the shape, interval and device-table views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("qq_norm", "中间那段几乎总是直的，信息在两端",
     "responsivity_A_W, device",
     "样本分位数对正态分位数，中段落在直线上并不稀奇——中心区被大量点平均掉。真正要看的是最"
     "上和最下那几个点离开直线多远，它们同时也是抽样波动最大的地方。"),
    ("qq_norm", "点位约定会挪动尾部",
     "responsivity_A_W, device",
     "同一个样本用 (i − 0.5)/n 与 i/(n + 1) 两种点位，端点的横坐标会差出小半个标准差。n 小于"
     "二十时这个差别肉眼可见，所以图上写清了用的是哪一种；比较两张 Q-Q 图之前先比较它们的约定。"),
    ("qq_norm", "看着直不等于来自正态总体",
     "lifetime_hours",
     "小样本下偏态或双峰的点列也能在中间段近似成一条线。这张图回答「形状像不像正态」，不回答"
     "「是不是正态」；要下后者结论得靠检验与已知量级的功效，那是另一件事，图上也不给 p 值。"),
    ("forest", "区间重叠不代表两组没有差异",
     "efficiency_percent, batch",
     "每组的区间是它自己均值的置信区间，两条重叠时两均值之差的标准误比单个的小得多。判断差异"
     "要做两两比较，不能拿两根棒目测交不交——这是森林图最常见的误读，方向还偏保守。"),
    ("forest", "用点而不是柱，是为了不必从 0 画起",
     "threshold_voltage_V, wafer",
     "柱状用面积编码大小，基线一截断就把 3% 的差画成两倍；点加区间用位置编码，纵轴不需要 0。"
     "要比较的都是同一量、数值又远离零时，这种画法比柱子更省画布也更不容易骗人。"),
    ("forest", "区间宽是读数少还是分散大",
     "dark_current_nA, device",
     "区间宽度是 t 乘标准误，标准误里含 1/√n。一根特别宽的棒可能只说明那组测得少。图上把每组"
     "的 n 写出来；要比较各组自身的分散程度，请看 SD 或回到点阵。"),
    ("group_metric_heatmap", "颜色跨列不可比",
     "device, responsivity_A_W, dark_current_nA, bandwidth_GHz",
     "每个指标各自归一，所以同一颜色在响应度列里表示「本批最高」，在暗电流列里也表示「本批最高」"
     "——而后者是坏事。读这张图必须先看列名与好坏方向，别把颜色当综合评分。"),
    ("group_metric_heatmap", "最差=0、最好=1 是构造出来的",
     "device, responsivity_A_W, dark_current_nA, bandwidth_GHz",
     "min–max 归一让每列两端顶满色带，哪怕实际只差 2%。器件只有两三个时整张图必然铺满极端颜色，"
     "那不代表差异大。要保留量级信息就改用 z 分数，或者把原始数值一并写在格子里——这里两个都做了。"),
    ("group_metric_heatmap", "不随器件变化的指标进不了这张图",
     "device, responsivity_A_W, dark_current_nA, bandwidth_GHz",
     "某列在全部器件上取值相同时，归一化的分母为零，它既没有颜色也没有比较价值。引擎把它从候选"
     "里剔出并在数据检查里点名，而不是画一条纯色列——那会让读者以为这个指标被核过了。"),
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
