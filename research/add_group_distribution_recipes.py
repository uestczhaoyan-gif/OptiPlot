"""Hand-written recipes for the group and distribution comparison views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("ecdf", "分位数直接读，不必先选分箱",
     "responsivity_A_W, device",
     "直方图要先决定箱宽，经验累积分布什么都不先决定：某个值以下有多少比例，就在纵轴上找到那个"
     "高度、水平读到横轴。要报中位数、四分位或阈值分布，这张图少一层会改变结论的假设。"),
    ("ecdf", "两级之间没有数据，就不画成连续",
     "responsivity_A_W, device",
     "阶梯从 0.5 跳到 0.9 只说明这 40% 的观测落在同一个点上，不说明中间那些值出现过。把阶梯"
     "抹平就是替样本编出一段它没有测量的区间。"),
    ("ecdf", "组间样本量不同时，台阶粗细不是分布差异",
     "threshold_voltage_V, batch",
     "一个台阶的高度是 1/n。n=8 的组每步抬 12.5%，n=200 的组几乎成一条平滑线——把这两条画在"
     "一起比形状之前，先确认读者不会把台阶粗细读成分散程度。"),
    ("beeswarm", "横向没有含义，但同值排开有",
     "responsivity_A_W, device",
     "随机抖动把两个完全相同的读数甩到不同横向位置，点的位置就成了噪声；这里按高度分带后对称"
     "排开，同一列里能数出有几个读数撞在同一个值上。横向仍然不携带信息，别把它当成第四维。"),
    ("beeswarm", "样本小的时候不要先算分位数",
     "efficiency_percent, batch",
     "五个点画箱线，中位数与上下四分位都取自单个观测，箱体宽度只是在猜总体。点阵把这些点原样"
     "摆出来，读者看到的样本量就是真正的样本量。"),
    ("beeswarm", "点太多时这张图会糊",
     "dark_current_nA, device",
     "分组超过 20 组或单组上千点时，排开的堆叠会连成一片，此时该回到箱线看汇总或改用直方图看"
     "形状；引擎对组数做了限制并在超出时截断，截断信息写在组标签上。"),
    ("group_bar", "SD 与 SEM 是两种语气",
     "responsivity_A_W, device",
     "同样一批数据，SEM 的误差棒比 SD 短 √n 倍。SD 说的是读数有多分散，SEM 说的是均值估得多准，"
     "两者都常被写成「误差棒」。图例必须写清用的哪个，换掉它往往就把「差异不明显」变成「差异显著」。"),
    ("group_bar", "柱状必须从 0 起画",
     "responsivity_A_W, device",
     "柱高比的是绝对量级，纵轴一旦从接近最小值处截断，3% 的差别就能看起来像两倍。引擎把纵轴"
     "下界钉在 0（数据含负值时钉在 0 与最小值之间），这不是保守，是柱状图唯一自洽的画法。"),
    ("group_bar", "一柱一棒之外什么都没了",
     "efficiency_percent, batch",
     "均值与误差棒丢掉了双峰、离群点、每组真实样本量与读数顺序。一个均值 0.8、SD 0.2 的组可能"
     "是 30 个稳定读数，也可能是 25 个 0.7 加 5 个 1.5——要区分这两种，得回到点阵、箱线或直方图。"),
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
