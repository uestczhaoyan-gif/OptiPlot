"""Append drawing recipes for the four new group-A figure types.

Written by hand rather than generated: a recipe is a statement about how to draw
honestly, and the corpus only tells us what to draw, not what caveat belongs with
it. No source attribution is carried, per the product's stance.
"""

from pathlib import Path
import json

here = Path(__file__).resolve().parents[1]
styles_path = here / "catalog" / "styles.json"
styles = json.loads(styles_path.read_text(encoding="utf-8"))

NEW = [
    # --- dual_axis ---
    ("dual_axis", "透反射同框",
     "wavelength_nm, T_percent, R_percent",
     "透射率与反射率共用波长轴但各占一条纵轴；两轴刻度独立，曲线看似同步不能当作二者相关的证据，要论证关系须改用散点图。"),
    ("dual_axis", "LIV 与正向电压",
     "drive_current_mA, optical_power_mW, forward_voltage_V",
     "光功率与正向电压对同一驱动电流作图；开启电压与阈值拐点应分别读各自轴，不要把两条曲线的交点解释为物理阈值。"),
    ("dual_axis", "谱位与线宽演化",
     "temperature_K, peak_wavelength_nm, fwhm_nm",
     "峰位与半高宽随温度同框；二者量纲不同故分轴，若只想表达趋势应拆成上下两个子图共享横轴。"),
    # --- spectral_derivative ---
    ("spectral_derivative", "吸收边导数定位",
     "wavelength_nm, absorbance",
     "对吸收谱求导以定位带隙边沿；微分放大噪声，须先确认原始数据平滑度，必要时同时给出原谱与导数两个子图。"),
    ("spectral_derivative", "消光谱一阶导稳频",
     "wavelength_nm, saturation_signal",
     "饱和吸收一阶导曲线用于锁频参考零点；过零点位置即参考，须标注扫描方向与磁场条件，且采样必须严格单调无重复波长。"),
    ("spectral_derivative", "椭偏参数拐点",
     "wavelength_nm, psi_deg",
     "Ψ 随波长的导数用于识别色散反常区；导数曲线不能替代原始 Ψ/Δ 双量展示，应作为附子图而非替换。"),
    # --- peak_evolution ---
    ("peak_evolution", "变角度共振色散",
     "theta_deg, wavelength_nm, reflectance",
     "每个角度取反射 dip 的中心波长与半高宽，得到随角度的色散与展宽；峰位于扫描边界的角度必须跳过并计数，不得把扫描端点当作共振。"),
    ("peak_evolution", "温漂峰位与线宽",
     "temperature_K, wavelength_nm, electroluminescence",
     "发光峰位与 FWHM 随温度变化，用于提取热淬灭趋势；半高宽以曲线端点的较高本底为参考，不以零为参考。"),
    ("peak_evolution", "多峰样品的峰跟踪",
     "gate_voltage_V, wavelength_nm, transmission",
     "存在多个峰时必须先指定跟踪哪一个（最接近某波长、或最高），否则逐角度取最大值会让峰位在两个峰之间跳变；本图型默认取内部极值，多峰须人工核对。"),
    # --- spectral_difference ---
    ("spectral_difference", "损耗工程差分透射谱",
     "wavelength_nm, T_with_antenna, T_without_antenna",
     "以加载纳米天线结构与空白结构的透射率逐点相减；零参考线必须可见，正负两侧不宜共用单色顺序色带。"),
    ("spectral_difference", "正交偏振二色性差",
     "angle_deg, R_left, R_right",
     "两个正交圆偏振响应之差直接给出圆二色性；两列必须来自同一角度网格，不得插值对齐，缺测角度整行舍弃并注明。"),
    ("spectral_difference", "变温差分光谱",
     "temperature_K, spectrum_T2, spectrum_T1",
     "先各自扣除本底再作差，否则差值里混着基线漂移；参考曲线取哪一个温度要在图上写明。"),
    # --- spectral_ratio ---
    ("spectral_ratio", "偏振消光比",
     "angle_deg, I_parallel, I_cross",
     "平行与交叉偏振输出之比随检偏角变化；分母接近零的角度必须屏蔽并在曲线上标出屏蔽段，否则比值尖峰是除法产物。"),
    ("spectral_ratio", "参考通道归一化谱",
     "wavelength_nm, I_sample, I_reference",
     "样品信号除以参考通道以扣除光源包络；应同时给出参考通道自身的噪声水平，否则归一化把噪声一起放大。"),
    ("spectral_ratio", "双波长比值测温标定",
     "temperature_K, peak_short_intensity, peak_long_intensity",
     "两发射峰强度比对温度的标定曲线；先在每个温度上成对计算比值再聚合，不要分别拟合两条曲线后相除。"),
    # --- spectral_envelope ---
    ("spectral_envelope", "批次透射包络",
     "wavelength_nm, device_01, device_02, device_03",
     "多器件同批测量的取值范围用填充带加中位数线表达；必须写明这是极差而不是标准差，器件数少时极差会严重高估分散程度。"),
    ("spectral_envelope", "变角度光谱展宽包络",
     "wavelength_nm, theta_0deg, theta_10deg, theta_20deg",
     "连续角度切片的上下包络用于显示随角度的展宽与漂移；若各角度曲线相互交叉，包络会掩盖顺序信息，应改回多曲线。"),
    ("spectral_envelope", "重复测量范围带",
     "wavelength_nm, rep_01, rep_02, rep_03",
     "同一条件重复测量的最小–最大带叠加均值线；重复次数少于 5 时优先改用误差棒而不是包络带。"),
    # --- energy_axis ---
    ("energy_axis", "波长与光子能量双刻度",
     "wavelength_nm, response",
     "同一组曲线下方标波长、上方标光子能量；副轴只是同一数据的换单位，不得对任一轴单独设范围或重采样。"),
    ("energy_axis", "吸收边带隙读数",
     "wavelength_nm, absorbance",
     "在吸收边附近用 eV 副轴直接读出带隙位置；副轴刻度应由主轴范围换算而来，不要为凑整数 eV 而截断主轴。"),
    ("energy_axis", "近红外到中红外的连续换轴",
     "wavelength_nm, transmittance",
     "跨波段时 eV 刻度在长波端严重压缩，需检查副轴刻度是否退化重叠；重叠时改用对数横轴或分段，而不是删掉刻度。"),
]

existing = {(s["label"], s["recipe"]) for s in styles}
by_prefix = {p: sum(1 for s in styles if s["id"].startswith(p + "-")) for p in
             {"spectral_difference", "spectral_ratio", "spectral_envelope", "energy_axis"}}

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
covered = sorted({p for s in styles for p in s["patterns"]})
print("covered patterns:", covered)
