"""Hand-written drawing recipes for the axis-scale and cumulative views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("log_log", "斜率要等比方才谈得上目视",
     "wavelength_nm, responsivity_a_w",
     "双对数上幂律是一条直线，但斜率是几何量：同一批数据换一张更方的画布就会看起来"
     "更陡。要用电眼判断指数，先开 log_aspect_equal 让两轴等比，此时 45° 即 k=1；"
     "要报出数值仍然得拟合 power_law，等比方不给数。"),
    ("log_log", "两个斜率就是两个机制",
     "wavelength_nm, responsivity_a_w",
     "直线在某一段成立、在另一段转折不成立时，转折点本身就是要报告的东西（散射机制"
     "改变、膜厚干涉截止）。不要用一条幂律跨过转折硬拟合，那会把两个机制平均成一个"
     "不存在的中间指数。"),
    ("log_log", "指数由拟合给出，不由坐标轴给出",
     "wavelength_nm, responsivity_a_w",
     "对数轴只承诺「看起来直」，它不检验残差。点名 power_law 之后必须看残差条：随机"
     "散布才支持幂律，成系统的大值残余说明形状不对，哪怕 R² 已经是三位数的 9。"),
    ("log_log", "一个零读数就使整列不能取对数",
     "wavelength_nm, responsivity_a_w",
     "对数轴画不出零与负值，所以这张图的前提是全列为正。引擎把「有非正值」当成图型"
     "不可用而报错，不是悄悄把那几个点丢掉——丢掉暗电流扣除后变负的点，等于把定标问题"
     "从图上擦掉。"),
    ("semi_log", "下降一个数量级的横轴长度就是答案",
     "delay_ns, photoluminescence_au",
     "半对数上指数衰减 y = A·exp(-x/τ) 是直线，下降一个数量级所需的横轴跨度等于 "
     "τ·ln10；读出这段跨度比读截距稳，因为截距受早期非指数段影响。"),
    ("semi_log", "直线排除不了幂律尾部",
     "delay_ns, photoluminescence_au",
     "拖尾在半对数上看着接近直线，并不唯一地支持指数：幂律衰减在有限区间上也能弯成"
     "近似直线。要分开两者，请看拟合残差面板的曲率，而不是看这条轴直不直。"),
    ("semi_log", "纵轴按数量级读，噪声在小值端变大",
     "wavelength_nm, responsivity_a_w",
     "对数纵轴上两点之间的垂直距离表示比值而不是差值，所以绝对噪声在小信号一端被"
     "放大成更大的视觉抖动。若尾段抖动比前段明显，那不是样品变了，是坐标在放大。"),
    ("cumulative_response", "末值是已扫区间的累计",
     "wavelength_nm, responsivity_a_w",
     "累积曲线的终点只算到最后一个实测横坐标，不是全量程总量；把它当作总量报告，"
     "等于把扫描范围当成样品的性质。要谈总量得先说明积分区间。"),
    ("cumulative_response", "跨缺测的那些段要数出来",
     "wavelength_nm, responsivity_a_w",
     "积分只走相邻实测点之间的梯形，缺测处按两端实测值线性计入并在图上写出跨了几段。"
     "一个跨过大气吸收带的长段和一堆小段加起来的数值可能一样，但可信度不一样。"),
    ("cumulative_response", "累积永远单调，形状要回原始曲线看",
     "wavelength_nm, responsivity_a_w",
     "累积曲线把峰和谷都抹平成斜率变化，它回答「哪一段贡献了多少」，不回答形状。"
     "归一化成占比时按曲线自身的最大偏离归一（不是按末值），否则有正负抵消的列会被"
     "除爆；不同列想放在同一张图比较，只有占比模式有意义。"),
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
