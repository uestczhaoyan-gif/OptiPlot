"""Hand-written recipes for the cloud-comparison views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("scatter_marginals", "边缘用的是配对行，不是各自的全体",
     "drive_current_mA, forward_voltage_V",
     "两条边缘直方图只统计两列同时有读数的行，所以它们的总数与散点一致。若分别对两列各画一次"
     "直方图（各自允许缺失），条数可能更多——那是另一张图，两个数字不能混着讲。"),
    ("scatter_marginals", "分箱一改，形状就可能一改",
     "drive_current_mA, forward_voltage_V",
     "边缘分布对分箱数敏感，长尾那一侧尤其明显。默认按 sqrt(n) 取箱并在图上写明当前箱数；"
     "要比较两份数据的边缘分布，必须先把 hist_bins 设成同一个值，否则比的是分箱方案。"),
    ("scatter_marginals", "点糊成一片时先改用分箱",
     "reference_V, measured_V",
     "上千个点叠在一起时散点看不出密度，边缘也救不了主面板；这种量级该用六边形分箱看联合分布。"
     "反过来，如果目的只是各自的分布而不是二者的关系，两条单变量直方图比这张三联图更省读。"),
    ("bland_altman", "先确认同一行就是同一样本",
     "power_ref_W, power_meter_W",
     "逐行配对是这张图唯一的前提，而表格里没有任何东西能证明它。两次测量若各自按顺序独立记录"
     "（比如都按波长排过序），差值就是两条无关序列相减，图上看起来仍然完整。先确认配对来源再谈界限。"),
    ("bland_altman", "一致性界限不是合格区间",
     "power_ref_W, power_meter_W",
     "均值 ± 1.96×SD 说的是「两种测法在这个范围内互换不影响单次读数」，它来自数据本身；"
     "允许多大偏差是规格或临床判断，不由这张图给出。把界限当成公差是这套方法最常见的误读。"),
    ("bland_altman", "这张图不替你检查自己的假设",
     "power_ref_W, power_meter_W",
     "界限要成立，得同时满足上下限是常数、差值近似正态。图上不给出"落在界外的点数"这类统计量："
     "SD 不是稳健量，一个粗大离群点会把两条界限一起撑宽，那个点随即回到界内，计数看起来反而正常。"
     "要判断假设就看点云的形状，这正是这张图画出来的东西。"),
    ("bland_altman", "谁减谁决定正负",
     "power_ref_W, power_meter_W",
     "差值取「第一列 − 第二列」，所以正的偏差均值表示第一列系统性偏高。换顺序整张图上下翻转，"
     "结论的符号跟着翻；报告时把顺序写进图注，别留给读者猜。"),
    ("pairs", "相关系数看不出弯",
     "drive_current_mA, forward_voltage_V, optical_power_mW",
     "激光器 L-I 曲线与正向电压的 Pearson r 可以到 +1.00，而阈值拐点在成对散点里一眼可见。"
     "成对图就是相关矩阵的逐点版本：凡是打算用「r=0.99」下结论的地方，先看那一格里的点长什么样。"),
    ("pairs", "每格的样本量可以不同",
     "drive_current_mA, forward_voltage_V, optical_power_mW",
     "每一格只用该两列同时有读数的行，所以格与格的 n 不一样，对角线的 n 又是该列自己的有效点数。"
     "这些数字写在每格上方；两列从不共现时格子留空并标注，而不是画一条看起来像数据的线。"),
    ("pairs", "列数封顶在 5",
     "drive_current_mA, forward_voltage_V, optical_power_mW",
     "面板数按列数的平方增长，6 列就是 36 格、每格不到一枚硬币大。超过 5 列请改用相关矩阵定位"
     "值得细看的少数对，再对那几对单独画散点。"),
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
