"""Hand-written recipes for the curve-family difference and re-scaling views."""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    ("difference_family", "参考条是定义不是测量",
     "theta_deg, wavelength_nm, reflectance",
     "被选作参考的那条曲线减完恒等于零，画成一条醒目的实线就会被读成「测到了零响应」。"
     "引擎把它换成灰色虚线基准线，并在图上写明参考是谁；要强调参考本身，请另画原始曲线图。"),
    ("difference_family", "先确认同一批坐标，再看少了几个点",
     "theta_deg, wavelength_nm, reflectance",
     "逐点相减只在两条曲线都有读数的坐标上成立。缺测处不插值补齐，图上会给出「多少个差值点"
     "未定义」；如果这个数字很大，说明两条曲线本来就不是同一批网格测的，应该先重测而不是先作差。"),
    ("difference_family", "ΔT/T 用相对模式，分母近零处会屏蔽",
     "delay_ps, probe_intensity, pump_state",
     "泵浦-探测的纵轴惯例是 (本条−参考)/参考。参考值接近零的行算出的相对差是除法的产物而不是"
     "测量，这些行被屏蔽并计数；报告 ΔT/T 时要把屏蔽了多少行一起写出来。"),
    ("difference_family", "换参考条就是换符号",
     "temperature_K, wavelength_nm, electroluminescence",
     "差值取「本条 − 参考」，所以选哪条当参考决定所有曲线的正负方向。默认是文件里的第一条，"
     "而仪器常常从高往低扫；用 reference 明确点名参考水平，别默认第一条就是物理上的初态。"),
    ("curves_normalized", "归一化之后不能再谈强度",
     "temperature_K, wavelength_nm, electroluminescence",
     "每条曲线除以自己的最大值之后，淬灭、增益阈值、响应度标定这类结论就没有依据了——它们正是"
     "被除掉的那个量。归一化图只能说明线形与峰位的关系，绝对强度必须回到原始曲线图去讲。"),
    ("curves_normalized", "按峰值还是按面积，问的是两个问题",
     "device_id, wavelength_nm, photoluminescence",
     "除以最大绝对值比的是峰的形状、宽度和位置；除以积分比的是能量在波长上的分布，此时最高的峰"
     "可以超过 1。两种归一化不能混用同一句话描述，图上的纵轴标签要跟着换。"),
    ("curves_normalized", "归一化会抬起噪声底",
     "theta_deg, wavelength_nm, reflectance",
     "除数越小，同样的绝对噪声在图上就越显眼。若某条曲线的最大偏离本来就小，归一化后它的抖动会"
     "盖过别的曲线，看起来像「这个角度的样品更不均匀」，实际只是除法放大了噪声。"),
    ("curves_normalized", "除数取不到就整条拒绝",
     "channel_id, wavelength_nm, transmittance",
     "最大值为零、或面积为零（正负抵消）的列没有归一化可言。引擎选择报错而不是把这条悄悄跳过，"
     "因为一张少了一条曲线的归一化图，看上去和一张完整的图没有区别。"),
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
