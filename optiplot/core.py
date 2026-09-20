"""Local data inspection and explainable, deterministic figure suggestions.

Scores rank structural matches; they are not probabilities or scientific advice.
The input is copied and missing values are retained for renderers to filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import io
import re

import numpy as np
import pandas as pd


@dataclass
class DataProfile:
    path: str = ""
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    angle_columns: list[str] = field(default_factory=list)
    axis_columns: list[str] = field(default_factory=list)
    grid_like: bool = False
    repeated_x: bool = False
    notes: list[str] = field(default_factory=list)
    data: object = None
    constant_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    error_columns: dict = field(default_factory=dict)
    grid_columns: list[str] = field(default_factory=list)
    angle_units: dict = field(default_factory=dict)
    missing_counts: dict = field(default_factory=dict)
    source_column: str | None = None
    target_column: str | None = None
    matrix_like: bool = False


@dataclass(frozen=True)
class Recommendation:
    id: str
    title: str
    score: int
    reason: str
    encodings: dict = field(default_factory=dict)


def _tokens(name: str) -> set[str]:
    # Boundaries matter: intensity, efficiency and pixel do not imply x or y.
    normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return set(re.findall(r"[a-z]+|[α-ω]+|[\u4e00-\u9fff]+", normalized.lower()))


def _has(name: str, words: set[str], chinese: tuple = ()) -> bool:
    return bool(_tokens(name) & words) or any(w in name for w in chinese)


def _is_angle(name: str) -> bool:
    return _has(name, {"angle", "theta", "phi", "azimuth", "θ", "φ"}, ("角度", "方位角"))


def _axis_rank(name: str) -> int:
    tokens = _tokens(name)
    if tokens & {"wavelength", "lambda", "frequency", "freq", "wavenumber"} or any(
        w in name for w in ("波长", "频率", "波数")
    ):
        return 0
    if tokens & {"time", "delay", "duration"} or any(w in name for w in ("时间", "延时")):
        return 1
    if tokens & {"x"}:
        return 2
    if tokens & {
        "position",
        "distance",
        "voltage",
        "current",
        "temperature",
        "concentration",
        "power",
    } or any(w in name for w in ("位置", "距离", "电压", "温度", "浓度", "功率")):
        return 3
    if _is_angle(name):
        return 4
    if tokens & {"y"}:
        return 5
    return 99


def _identifier(name: str) -> bool:
    return _has(name, {"id", "index", "identifier"}, ("序号", "编号")) or str(name).lower() in {
        "row",
        "row_number",
    }


def _unique_names(columns) -> list[str]:
    names, used = [], set()
    for i, col in enumerate(columns):
        base = str(col).strip() or f"column_{i + 1}"
        name, suffix = base, 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
        used.add(name)
    return names


def _read_text(path: Path, header: bool) -> pd.DataFrame:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError("文件为空，至少需要一行数据。")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("无法解码文本文件，请保存为 UTF-8 或 GB18030。") from exc
    sample = "\n".join(line for line in text.splitlines() if line.strip())[:16384]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        first = sample.splitlines()[0]
        delimiter = next((d for d in ("\t", ",", ";") if d in first), None)
    if delimiter:
        rows = [
            row
            for row in csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
            if row and any(v.strip() for v in row)
        ]
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("各行列数不一致，请检查分隔符、缺失的分隔符或未配对引号。")
        separator = delimiter
    else:
        lines = [line for line in text.splitlines() if line.strip()]
        split = [re.split(r"\s+", line.strip()) for line in lines]
        separator = (
            r"\s+" if len(split[0]) > 1 and all(len(row) == len(split[0]) for row in split) else ","
        )
    try:
        df = pd.read_csv(
            io.StringIO(text), sep=separator, engine="python", header=0 if header else None
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise ValueError(f"无法解析数据表：{exc}") from exc
    if not header:
        df.columns = [f"column_{i + 1}" for i in range(len(df.columns))]
    return df


def _array_frame(array, name="value") -> pd.DataFrame:
    arr = np.asarray(array)
    if arr.dtype.kind not in "biuf":
        raise ValueError("只支持实数数组；对象、复数或结构体请先导出为数值表。")
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    if arr.ndim != 2:
        raise ValueError("数组必须是一维或二维；请先选择高维数组中的一个切片。")
    return pd.DataFrame(arr, columns=[f"column_{i + 1}" for i in range(arr.shape[1])])


def analyze_file(path: str | Path, header: bool = True, sheet_name=0) -> DataProfile:
    """Load CSV/TSV/TXT, XLSX, NPY or a simple MATLAB v4-v7.2 file.

    Equal-length MAT vectors become named columns. Otherwise the largest real
    numeric array is selected, with an explicit note. v7.3/HDF5 is unsupported.
    """
    source, notes = Path(path), []
    matrix_like = False
    suffix = source.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt", ".dat"}:
            df = _read_text(source, header)
        elif suffix in {".xlsx", ".xls"}:
            if sheet_name is None:
                raise ValueError("请选择一个工作表。")
            df = pd.read_excel(source, header=0 if header else None, sheet_name=sheet_name)
            if not header:
                df.columns = [f"column_{i + 1}" for i in range(len(df.columns))]
        elif suffix == ".npy":
            array = np.load(source, allow_pickle=False)
            df = _array_frame(array)
            matrix_like = array.ndim == 2 and min(array.shape) >= 2
            notes.append(
                "NPY 按行作为观测、按列作为变量读取；数组没有物理量名称或单位，请在绘图设置中确认。"
            )
        elif suffix == ".mat":
            from scipy.io import loadmat

            try:
                variables = loadmat(source)
            except NotImplementedError as exc:
                raise ValueError("暂不支持 MATLAB v7.3/HDF5，请另存为 -v7 MAT 或 CSV。") from exc
            arrays = {
                k: np.asarray(v)
                for k, v in variables.items()
                if not k.startswith("__")
                and isinstance(v, np.ndarray)
                and v.dtype.kind in "biuf"
                and v.size > 0
                and v.ndim <= 2
            }
            if not arrays:
                raise ValueError("MAT 文件中没有可读取的实数向量或二维数组。")
            vectors = {k: v.reshape(-1) for k, v in arrays.items() if v.ndim == 1 or 1 in v.shape}
            if len(vectors) == len(arrays) and len({len(v) for v in vectors.values()}) == 1:
                df = pd.DataFrame(vectors)
                notes.append("已将 MAT 中等长实数向量读为命名数据列。")
            else:
                selected = max(arrays, key=lambda k: arrays[k].size)
                arr = arrays[selected]
                df = _array_frame(arr.reshape(-1) if selected in vectors else arr, selected)
                matrix_like = selected not in vectors and min(arr.shape) >= 2
                notes.append(
                    f"MAT 选择了最大实数数组 {selected}，形状 {arr.shape}；其他变量未合并。"
                )
        else:
            raise ValueError("不支持此格式。请使用 CSV、TSV、TXT、XLSX、NPY 或 MAT。")
    except FileNotFoundError:
        raise ValueError(f"找不到数据文件：{source}") from None
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError(f"读取此格式需要安装相应依赖：{exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"文本格式错误：{exc}") from exc
    result = analyze_dataframe(df, str(source))
    result.matrix_like = matrix_like
    result.notes[:0] = notes
    return result


def analyze_dataframe(df: pd.DataFrame, path="") -> DataProfile:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("analyze_dataframe 需要 pandas.DataFrame。")
    if df.empty or len(df.columns) == 0:
        raise ValueError("数据表为空，至少需要一行数据和一列变量。")
    data = df.copy(deep=True)
    data.columns = _unique_names(data.columns)
    data = data.replace(r"^\s*$", np.nan, regex=True)
    notes, numeric, inf_count = [], [], 0
    for col in data.columns:
        values = data[col]
        if pd.api.types.is_complex_dtype(values):
            raise ValueError(f"列 {col} 包含复数，请先选择幅值、相位、实部或虚部。")
        if pd.api.types.is_bool_dtype(values):
            continue
        converted = pd.to_numeric(values, errors="coerce")
        nonmissing = values.notna()
        # Mixed categories such as 'control, 1, 2' stay categorical.
        if nonmissing.any() and converted[nonmissing].notna().all():
            inf_count += int(np.isinf(converted.to_numpy(dtype=float)).sum())
            data[col] = converted.replace([np.inf, -np.inf], np.nan)
            if data[col].notna().any():
                numeric.append(col)
    if inf_count:
        notes.append(f"发现 {inf_count} 个正/负无穷值，已按缺失值处理。")
    if not data.notna().any().any():
        raise ValueError("数据表没有有效观测值（全部为空或无穷）。")
    missing = {col: int(data[col].isna().sum()) for col in data.columns if data[col].isna().any()}
    if missing:
        notes.append(
            f"共有 {sum(missing.values())} 个缺失单元格；各图仅使用相关列的有效观测，不自动插值。"
        )
    constant = [col for col in numeric if data[col].nunique(dropna=True) <= 1]
    ids = [col for col in numeric if _identifier(col)]
    if constant:
        notes.append("常量列不参与相关性或自动坐标选择：" + "、".join(constant))
    if ids:
        notes.append("编号列不参与自动数值关系推荐：" + "、".join(ids))
    errors = {}
    for col in numeric:
        match = re.match(r"^(.*?)[_\s.-]+(sd|sem|std|stderr)$", col, re.I)
        if match:
            response = next((n for n in numeric if n.lower() == match.group(1).lower()), None)
            if response:
                kind = "sem" if match.group(2).lower() in {"sem", "stderr"} else "sd"
                errors[col] = {"response": response, "kind": kind}
                if (data[col].dropna() < 0).any():
                    notes.append(f"误差列 {col} 含负值，不推荐作为误差棒使用。")
    eligible = [
        col for col in numeric if col not in ids and col not in constant and col not in errors
    ]
    axes = sorted((col for col in eligible if _axis_rank(col) < 99), key=_axis_rank)
    if not axes:
        axes = [col for col in eligible if _ordered(data[col])][:1]
    angles = [col for col in eligible if _is_angle(col)]
    units = {}
    for col in angles:
        tokens = _tokens(col)
        if tokens & {"rad", "radian", "radians"} or "弧度" in col:
            units[col] = "rad"
        elif tokens & {"deg", "degree", "degrees"} or "°" in col or "度" in col:
            units[col] = "deg"
        else:
            units[col] = "deg" if data[col].abs().max() > 2 * np.pi + 1e-6 else "rad"
            notes.append(
                f"{col} 未注明角度单位，暂按 {'度' if units[col] == 'deg' else '弧度'}显示；请在设置中确认。"
            )
    grid = _find_grid(data, eligible)
    repeated = bool(axes and data[axes[0]].dropna().duplicated().any())
    cats = [col for col in data.columns if col not in numeric]
    source = next(
        (c for c in data.columns if str(c).strip().lower() in {"source", "from", "源", "起点"}),
        None,
    )
    target = next(
        (c for c in data.columns if str(c).strip().lower() in {"target", "to", "目标", "终点"}),
        None,
    )
    if not numeric:
        notes.append("未发现有效数值列；可查看数据表，source/target 边列表可生成流程图。")
    if len(data) == 1:
        notes.append("只有一行观测，无法估计分布、趋势或重复测量误差。")
    return DataProfile(
        path=str(path),
        n_rows=len(data),
        n_cols=len(data.columns),
        columns=list(data.columns),
        numeric_columns=numeric,
        categorical_columns=cats,
        angle_columns=angles,
        axis_columns=axes,
        grid_like=bool(grid),
        repeated_x=repeated,
        notes=notes,
        data=data,
        constant_columns=constant,
        id_columns=ids,
        error_columns=errors,
        grid_columns=grid,
        angle_units=units,
        missing_counts=missing,
        source_column=source,
        target_column=target,
    )


def _ordered(values) -> bool:
    values = values.dropna()
    return (
        len(values) >= 4
        and values.nunique() >= 4
        and (values.is_monotonic_increasing or values.is_monotonic_decreasing)
    )


def _find_grid(data, eligible) -> list[str]:
    # Coincidental products of category counts are not measured surfaces.
    xs = [c for c in eligible if _tokens(c) & {"x"}]
    ys = [c for c in eligible if _tokens(c) & {"y"}]
    axes = sorted((c for c in eligible if _axis_rank(c) < 99), key=_axis_rank)
    candidates = [(x, y) for x in xs for y in ys if x != y]
    candidates += [
        (x, y) for i, x in enumerate(axes) for y in axes[i + 1 :] if (x, y) not in candidates
    ]
    for x, y in candidates:
        for z in eligible:
            if z in {x, y}:
                continue
            valid = data[[x, y, z]].dropna()
            nx, ny = valid[x].nunique(), valid[y].nunique()
            if nx >= 3 and ny >= 3 and len(valid) == nx * ny and not valid.duplicated([x, y]).any():
                return [x, y, z]
    return []


def _valid_count(data, columns) -> int:
    return len(data[list(dict.fromkeys(columns))].dropna())


def recommend(profile: DataProfile, catalog_path=None) -> list[Recommendation]:
    """Return at most eight renderable choices. No regression is automatic."""
    p, out = profile, []
    data = p.data
    if not isinstance(data, pd.DataFrame) or data.empty:
        return []
    eligible = [
        c
        for c in p.numeric_columns
        if c not in p.constant_columns and c not in p.id_columns and c not in p.error_columns
    ]

    def add(identifier, title, score, reason, encodings):
        out.append(
            Recommendation(
                identifier, title, score, reason + "（结构启发式评分，非科学结论。）", encodings
            )
        )

    if (
        p.source_column
        and p.target_column
        and _valid_count(data, [p.source_column, p.target_column])
    ):
        nodes = pd.unique(
            data[[p.source_column, p.target_column]].dropna().astype(str).to_numpy().ravel()
        )
        if len(nodes) <= 18:
            add(
                "flow",
                "流程 / 关系图",
                98,
                "存在 source/target 边列表，节点和箭头可直接由已提供关系构建",
                {"source": p.source_column, "target": p.target_column},
            )
    if p.n_rows == 1:
        add("table", "数据表", 85, "只有一行观测，表格可完整保留数值与标签", {"columns": p.columns})
        return sorted(out, key=lambda r: r.score, reverse=True)[:8]

    group = next(
        (c for c in p.categorical_columns if 2 <= data[c].nunique() <= min(12, p.n_rows // 2)), None
    )
    x = next((c for c in p.axis_columns if c in eligible), eligible[0] if eligible else None)
    responses = [c for c in eligible if c != x and c not in p.angle_columns]
    y = responses[0] if responses else None
    if p.matrix_like and p.n_rows >= 2 and len(p.numeric_columns) >= 2:
        add(
            "matrix_heatmap",
            "数组矩阵热图",
            93,
            "导入的是二维无坐标数组，可按行列索引预览数值；请确认行列对应的物理含义",
            {"columns": p.numeric_columns},
        )
    if p.grid_like and len(p.grid_columns) == 3:
        gx, gy, gz = p.grid_columns
        enc = {"x": gx, "y": gy, "z": gz}
        add(
            "heatmap",
            "二维参数热图",
            97,
            "两个坐标列的唯一观测覆盖完整网格，颜色可编码第三个测量量",
            enc,
        )
        add("contour", "填色等高线", 87, "完整的二维网格支持等值线；无须猜测缺失采样点", enc.copy())

    if p.angle_columns and not p.grid_like:
        theta = p.angle_columns[0]
        radius = next((c for c in eligible if c != theta and c not in p.axis_columns), None)
        if radius is None:
            radius = next((c for c in eligible if c != theta), None)
        if (
            radius
            and _valid_count(data, [theta, radius]) >= 3
            and (data[radius].dropna() >= 0).all()
        ):
            enc = {"theta": theta, "r": radius, "angle_unit": p.angle_units.get(theta, "deg")}
            if group:
                enc["group"] = group
            add(
                "polar",
                "极坐标响应图",
                98,
                "列名含角度信息且径向值非负；请核对角度单位及周期含义",
                enc,
            )

    if x and y and _valid_count(data, [x, y]) >= 2:
        enc = {"x": x, "y": y}
        if group:
            enc["group"] = group
        add(
            "scatter_fit",
            "散点关系图",
            77,
            "至少两个数值变量有配对观测，可检查关系；默认不拟合",
            enc,
        )
        if _valid_count(data, [x, y]) >= 300:
            add(
                "density",
                "二维密度 / 六边形分箱",
                89,
                "配对观测较多，分箱可减轻散点重叠；颜色表示各区域样本数",
                {"x": x, "y": y},
            )
        line_ys = [c for c in responses if _valid_count(data, [x, c]) >= 3]
        keys = [x] + ([group] if group else [])
        line_unique = not data[keys].dropna().duplicated(keys).any()
        if (
            line_ys
            and p.n_rows >= 4
            and not p.grid_like
            and line_unique
            and (_axis_rank(x) < 99 or _ordered(data[x]))
        ):
            enc = {"x": x, "y": line_ys[:8]}
            if group:
                enc["group"] = group
            add(
                "spectrum_lines",
                "有序响应 / 光谱曲线",
                91,
                "横轴名称含扫描轴信息，或观测按横轴有序；曲线显示采样点间变化",
                enc,
            )

    error_added = False
    for err, info in p.error_columns.items():
        response = info["response"]
        error_x = next((c for c in p.axis_columns if c != response and c != err), None)
        if error_x is None:
            error_x = next((c for c in eligible if c != response), None)
        if error_x and _valid_count(data, [error_x, response, err]) >= 2:
            valid = data[[error_x, response, err]].dropna()
            if (valid[err] >= 0).all():
                enc = {"x": error_x, "y": response, "error": err, "error_kind": info["kind"]}
                if group:
                    enc["group"] = group
                add(
                    "errorbar",
                    f"均值与显式 {info['kind'].upper()} 误差棒",
                    95,
                    "存在与响应列配对且非负的误差列，保留其已有统计定义",
                    enc,
                )
                error_added = True
                break
    if not error_added and x and y and not p.grid_like and x in p.axis_columns:
        keys = [x] + ([group] if group else [])
        valid = data[keys + [y]].dropna()
        counts = valid.groupby(keys, observed=True)[y].count()
        if (counts >= 2).sum() >= 2:
            enc = {"x": x, "y": y, "error_kind": "sd", "replicates": True}
            if group:
                enc["group"] = group
            add(
                "errorbar",
                "重复观测均值 ± SD",
                94,
                "至少两个横轴取值具有组内重复观测，可计算样本标准差；请确认这些确为重复实验",
                enc,
            )

    if eligible:
        value = y or eligible[0]
        if data[value].notna().sum() >= 3:
            add(
                "distribution",
                "数值分布直方图",
                73 if len(eligible) == 1 else 62,
                "有效观测可用直方图检查分布和离群值；分箱会影响外观",
                {"value": value},
            )
        if group and _valid_count(data, [group, value]) >= 4:
            sizes = data[[group, value]].dropna().groupby(group, observed=True)[value].count()
            if (sizes >= 2).sum() >= 2:
                add(
                    "box",
                    "分组箱线图与原始点",
                    88,
                    "分类列中至少两组有重复数值观测，可比较中位数、四分位与原始点",
                    {"group": group, "value": value},
                )
    if len(eligible) >= 3 and p.n_rows >= 5:
        # Bound pairwise inspection for wide detector/spectral arrays. The GUI
        # still exposes every column for manual selection.
        usable = [c for c in eligible[:16] if data[c].notna().sum() >= 3]
        paired = [
            (a, b)
            for i, a in enumerate(usable)
            for b in usable[i + 1 :]
            if _valid_count(data, [a, b]) >= 3
        ]
        if paired:
            used = [c for c in usable if any(c in pair for pair in paired)]
            add(
                "correlation",
                "数值变量相关矩阵",
                69,
                "多个变化数值列可检查线性相关；使用成对有效观测，相关不代表因果",
                {"columns": used[:16]},
            )
    add(
        "table",
        "数据表",
        48 if eligible else 85,
        "表格保留原始标签、数值与缺失项，适合逐项核对和小样本呈现",
        {"columns": p.columns},
    )
    return sorted(out, key=lambda r: r.score, reverse=True)[:8]
