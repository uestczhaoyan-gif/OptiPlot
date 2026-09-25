"""Local data inspection and explainable, deterministic figure suggestions.

Scores rank structural matches; they are not probabilities or scientific advice.
The input is copied and missing values are retained for renderers to filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import io
import math
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


TIER_LABELS = {"high": "高", "medium": "中", "low": "低"}
TIER_ORDER = ("high", "medium", "low")
# How many candidates one dataset may put forward. Measured, not aesthetic: the
# widest shipped shape asks for about a dozen, so anything lower starts deleting
# views the engine had already justified.
MAX_CANDIDATES = 16


@dataclass(frozen=True)
class Recommendation:
    """One drawable option.

    `tier` states how well the data meets this figure type's preconditions:
    high means every precondition holds, medium means it can be drawn but rests
    on an assumption the user should confirm, low means it is technically
    available and usually not the best expression. `rank` only breaks ties
    inside a tier and is not shown to users.
    """

    id: str
    title: str
    tier: str
    reason: str
    encodings: dict = field(default_factory=dict)
    rank: int = 0

    @property
    def tier_label(self) -> str:
        return TIER_LABELS[self.tier]


def _tokens(name: str) -> set[str]:
    # Boundaries matter: intensity, efficiency and pixel do not imply x or y.
    normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return set(re.findall(r"[a-z]+|[α-ω]+|[\u4e00-\u9fff]+", normalized.lower()))


def _has(name: str, words: set[str], chinese: tuple = ()) -> bool:
    return bool(_tokens(name) & words) or any(w in name for w in chinese)


def _is_angle(name: str) -> bool:
    return _has(name, {"angle", "theta", "phi", "azimuth", "θ", "φ"}, ("角度", "方位角"))


# A trailing token only counts as a unit if it really is one. Matching on length
# alone reads the series label in `device_A` as a unit and then declares
# device_A and device_B incomparable, which silently kills the difference and
# ratio options on exactly the data they are most useful for.
UNITS = {
    # length
    "nm", "um", "µm", "mm", "cm", "m", "mkm", "in", "px", "angstrom", "a0",
    # energy
    "ev", "kev", "mev", "gev", "j", "nj", "uj", "mj", "kj", "cm-1", "cm1",
    # time
    "fs", "ps", "ns", "us", "µs", "ms", "s", "min", "h", "hr", "d",
    # frequency
    "hz", "khz", "mhz", "ghz", "thz", "phz",
    # electrical
    "v", "mv", "uv", "µv", "kv", "a", "ma", "ua", "µa", "na", "pa", "k_a",
    "w", "mw", "uw", "µw", "nw", "gw", "db", "dbm", "dbc",
    # photometric / radiometric
    "au", "arb", "counts", "cps", "rps", "lumens", "lux", "w_m2", "w_cm2",
    # other
    "deg", "degree", "rad", "percent", "pct", "%", "k", "mk", "g", "mg", "ug",
    "µg", "kg", "mol", "mmol", "umol", "m", "ohm", "omega", "f", "pf", "nf",
}


# A one-letter tail is only a unit when the stem names the quantity that letter
# measures. "A" is the ampere, but a trailing A/B/C is far more often a series
# label, so device_A stays unit-less while forward_voltage_V really does carry
# volts -- and a laser's current/voltage/power sweep is exactly the case where
# two responses need separate axes.
LETTER_UNITS = {
    "v": {"voltage", "potential", "bias"},
    "a": {"current", "amperage"},
    "w": {"power"},
    "k": {"temperature"},
    "s": {"time", "delay", "duration"},
    "g": {"mass", "weight"},
    "m": {"length", "distance", "position", "absorbance", "transmittance"},
    "d": {"diameter", "thickness", "depth"},
}


def _unit_of(name: str) -> str | None:
    """Trailing unit token, only if it is actually a known unit: `x_um` -> "um",
    but `device_A` -> None."""
    parts = re.split(r"[_\s.]+", str(name).strip())
    if len(parts) < 2:
        return None
    tail = parts[-1].lower().replace("²", "2").replace("³", "3")
    if tail not in UNITS:
        return None
    if len(tail) >= 2 or _tokens(name) & LETTER_UNITS.get(tail, set()):
        return tail
    return None


def _comparable(a: str, b: str) -> bool:
    """Whether two columns are the same kind of quantity, which is what makes a
    difference or ratio meaningful.

    Subtracting a position from an intensity is arithmetically valid and
    physically empty, so the pair has to agree on unit when both name one, and
    neither may be an independent variable.
    """
    if _axis_rank(a) < 99 or _axis_rank(b) < 99:
        return False
    ua, ub = _unit_of(a), _unit_of(b)
    if ua and ub:
        return ua == ub
    # No unit on either side: require a shared leading stem, so device_A and
    # device_B pair but alpha and beta do not. Being unable to tell is a reason
    # to withhold the option, not to offer it.
    if not ua and not ub:
        sa = re.split(r"[_\s.]+", str(a))[0].lower()
        sb = re.split(r"[_\s.]+", str(b))[0].lower()
        return sa == sb
    return False


def _distinct_quantity(a: str, b: str) -> bool:
    """Two columns whose named units differ, which is what could justify giving
    them separate axes.

    Deliberately conservative: both must carry a recognised unit. Columns in the
    same unit belong on one axis, and splitting them across two is exactly how a
    dual-axis plot manufactures a relationship that the data does not show.
    """
    ua, ub = _unit_of(a), _unit_of(b)
    if not (ua and ub and ua != ub):
        return False
    # A bare x/y token names a coordinate of the frame rather than a response, so
    # the cross-section of a beam map is not a second quantity to give its own
    # scale. Quantity words that can go either way -- voltage, current, power --
    # stay eligible, which is what a laser's I-V-P sweep needs.
    return all(_axis_rank(c) not in (2, 5) for c in (a, b))


def _parameter_group(data, x, candidates, n_rows: int) -> str | None:
    """A second scan axis whose values index repeated blocks of the first one.

    Instrument exports store the angle, temperature or delay step as a number, so
    no column reads as categorical and the curve family collapses into one
    zigzag. The uniqueness test is what separates "fifteen spectra at fifteen
    angles" from "the same point measured fifteen times" -- and replicates need
    error bars, not a family of lines.
    """
    best = None
    for c in candidates:
        levels = data[c].dropna().nunique()
        if not 2 <= levels <= 24 or levels * 4 > n_rows:
            continue
        pairs = data[[x, c]].dropna()
        if pairs.empty or pairs.duplicated().any():
            continue
        if best is None or levels < best[1]:
            best = (c, levels)
    return best[0] if best else None


def _has_interior_extremum(values) -> bool:
    """Whether a curve turns around inside its own range rather than running
    monotonically to an edge. A peak that sits on the boundary is a truncated
    scan, not a resonance."""
    v = np.asarray(values, dtype=float)
    if v.size < 5:
        return False
    i = int(np.argmax(v))
    j = int(np.argmin(v))
    interior_max = 0 < i < v.size - 1 and v[i] > max(v[0], v[-1])
    interior_min = 0 < j < v.size - 1 and v[j] < min(v[0], v[-1])
    return interior_max or interior_min


def _is_wavelength(name: str) -> bool:
    """Only a wavelength can be relabelled as photon energy via 1239.84/λ.
    Frequency and wavenumber axes are rank-0 scan axes too but invert differently,
    so they must not share this check."""
    return _has(
        name,
        {"wavelength", "lambda", "nm"},
        ("波长",),
    )


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


def _strictly_ordered(values) -> bool:
    """Monotonic is not enough to differentiate: a repeated coordinate makes the
    finite-difference denominator zero."""
    v = values.dropna().to_numpy(dtype=float)
    return v.size >= 4 and bool(np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0))


def _level_text(value) -> str:
    """A group level as it should read in prose: 25 not 25.0, A not 0.0."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _decade_span(values) -> float:
    """How many factors of ten a column covers, or 0.0 when it cannot be logged.

    A log axis drops non-positive values rather than showing them, so a column
    with a single zero in it does not have a small span; it has no span. The
    caller treats 0.0 as "this view is not offerable", never as "flat".
    """
    v = values.dropna().to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.any(v <= 0):
        return 0.0
    return float(math.log10(v.max() / v.min()))


_MUELLER_CELLS = tuple(f"m{i}{j}" for i in range(4) for j in range(4))


def _named_block(columns, keys):
    """Map a set of expected short names onto real column names, tolerating the
    separators and case instrument exports add between them."""
    lookup = {re.sub(r"[_\s\-.]", "", str(c)).lower(): c for c in columns}
    return [lookup[k] for k in keys if k in lookup] if all(k in lookup for k in keys) else None


def _mueller_columns(columns) -> list[str] | None:
    """The 16 elements of a Mueller matrix, but only when the table says so.

    A 4x4 block of numbers is not evidence of a matrix -- most four-by-four
    tables are not -- so the claim is made by the column names and nothing else.
    """
    return _named_block(columns, _MUELLER_CELLS)


def _stokes_columns(columns) -> list[str] | None:
    return _named_block(columns, ("s0", "s1", "s2", "s3"))


def _widest_x_gap(x):
    """The largest empty interval in a scan axis, measured against the sampling
    spacing.

    A gap earns a broken axis only if it is far wider than the interval the data
    is otherwise sampled at. Anything smaller is a sparse region, and a
    continuous line over it is honest; breaking the axis there would buy canvas
    nobody needed and invite the reader to compare slopes across a seam.
    """
    v = np.unique(pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy())
    if v.size < 6:
        return None
    steps = np.diff(v)
    median = float(np.median(steps))
    if median <= 0:
        return None
    i = int(np.argmax(steps))
    if steps[i] < 8.0 * median:
        return None
    return float(v[i]), float(v[i + 1]), float(steps[i] / median)


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


def _partial_grid(data, eligible):
    """(x, y, z, coverage, holes) for a regular scan that is mostly but not fully
    sampled.

    A dropped reading in an angle-by-wavelength map leaves one blank cell, and a
    heatmap can show that honestly. A scatter cloud is not that: the coverage
    ratio is what separates a grid with holes from points that never formed a
    grid, and the two need different plots.
    """
    axes = sorted((c for c in eligible if _axis_rank(c) < 99), key=_axis_rank)
    best = None
    for i, x in enumerate(axes):
        for y in axes[i + 1 :]:
            for z in eligible:
                if z in {x, y}:
                    continue
                valid = data[[x, y, z]].dropna()
                nx, ny = valid[x].nunique(), valid[y].nunique()
                if nx < 3 or ny < 3 or valid.duplicated([x, y]).any():
                    continue
                cells = nx * ny
                coverage = len(valid) / cells
                if 0.6 <= coverage < 1.0 and (best is None or coverage > best[3]):
                    best = (x, y, z, coverage, cells - len(valid))
    return best


def _peak_trackable(p: DataProfile, data) -> dict | None:
    """First (parameter, wavelength, response) triple whose curves have a peak
    that can be tracked, or None.

    Each parameter setting needs enough points to locate an extremum, and the
    extremum has to be interior rather than on the edge of the scan - a "peak"
    pinned to the last measured wavelength is a truncated sweep, and tracking it
    would report the scan limit as a resonance.
    """
    used = set(p.error_columns) | set(p.id_columns) | set(p.constant_columns)
    waves = [c for c in p.numeric_columns if _is_wavelength(c) and c not in used]
    others = [c for c in p.numeric_columns if c not in used and c not in waves]
    for wave in waves:
        for param in others:
            rest = [c for c in others if c != param]
            for value in rest:
                rows = data[[param, wave, value]].dropna()
                if rows.empty:
                    continue
                sizes = rows.groupby(param)[wave].count()
                if len(sizes) < 3 or (sizes < 5).any():
                    continue
                tracked = 0
                for _, group in rows.groupby(param):
                    ordered = group.sort_values(wave)
                    if _has_interior_extremum(ordered[value].to_numpy()):
                        tracked += 1
                if tracked >= max(3, int(0.6 * len(sizes))):
                    return {"param": param, "wave": wave, "value": value}
    return None


def _valid_count(data, columns) -> int:
    return len(data[list(dict.fromkeys(columns))].dropna())


def recommend(profile: DataProfile) -> list[Recommendation]:
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

    def add(identifier, title, tier, reason, encodings, rank):
        grouped_by = encodings.get("group")
        if grouped_by:
            # Grouping is inferred from the column structure, so the figure has to
            # name what it grouped by rather than leave it to the legend title.
            reason += f"；按 {grouped_by} 的 {data[grouped_by].dropna().nunique()} 个取值分组"
        out.append(Recommendation(identifier, title, tier, reason, encodings, rank))

    def by_tier(items):
        # Sixteen, not eight: the widest shipped shape (a complete grid, and a
        # multi-column spectrum) already asks for twelve candidates, so a cap of
        # eight was silently deleting offered views rather than trimming noise.
        # The number is measured against `examples/`, and the tiers still order
        # what survives.
        return sorted(items, key=lambda r: (TIER_ORDER.index(r.tier), r.rank))[
            :MAX_CANDIDATES
        ]

    mueller = _mueller_columns(p.numeric_columns)
    if mueller:
        add(
            "mueller_matrix",
            "Mueller 矩阵 4×4",
            "high",
            "列名给出 m00…m33 全部 16 个元素，按 4×4 展示；色标以零为中心并对称于最大绝对值，"
            "因为负元素在偏振里有符号含义而不是「数值小」；"
            + (f"当前 {p.n_rows} 行，只显示第一行" if p.n_rows > 1 else "每个元素标出数值"),
            {"cells": mueller},
            0,
        )
    if p.matrix_like and p.n_rows >= 2 and len(p.numeric_columns) >= 2:
        add(
            "matrix_heatmap",
            "数组矩阵热图",
            "high",
            "导入的是二维无坐标数组，可按行列索引预览数值；请确认行列对应的物理含义",
            {"columns": p.numeric_columns},
            3,
        )
    if p.grid_like and len(p.grid_columns) == 3:
        gx, gy, gz = p.grid_columns
        enc = {"x": gx, "y": gy, "z": gz}
        dims = (data[gx].nunique(), data[gy].nunique())
        if min(dims) >= 5:
            add(
                "surface_3d",
                "三维表面图",
                "medium",
                f"{gx} × {gy} 的完整网格（{dims[0]}×{dims[1]}）可以立起来看整体形状；"
                "但透视投影下高度无法对着刻度读、前面会挡住后面，"
                "要取数值请改用热图或等高线",
                enc.copy(),
                3,
            )
            add(
                "surface_with_contour",
                "三维表面 + 俯视等高线",
                "medium",
                "左侧立体看形状，右侧俯视等高线读数值，两幅共用同一批格点；"
                "单独一张三维图不能同时给出可读的高度，这个配对就是为了补上那一条",
                enc.copy(),
                3,
            )
        add(
            "heatmap",
            "二维参数热图",
            "high",
            "两个坐标列的唯一观测覆盖完整网格，颜色可编码第三个测量量",
            enc,
            1,
        )
        add(
            "contour",
            "填色等高线",
            "medium",
            "完整的二维网格支持等值线；填色会在采样点之间插值，"
            "若采样间隔内物理量并不连续请改用热图",
            enc.copy(),
            1,
        )
        add(
            "heatmap_contours",
            "热图叠加等高线",
            "medium",
            "颜色保留每个格点的实测值，等高线只标出Levels 的经过位置；"
            "线的位置由采样点间插值得到，因此不能把两条线之间的区域当作已测",
            enc.copy(),
            1,
        )
        add(
            "heatmap_marginals",
            "热图加边缘剖面",
            "medium",
            "在主图上方与右方各给一条沿另一轴求均值的剖面；"
            "均值会把整列的起伏压成一个数，看离散程度请回到热图或改用投影的分位数",
            enc.copy(),
            1,
        )
        add(
            "heatmap_normalized",
            "逐行（列）归一化热图",
            "medium",
            f"每个 {gy} 处沿 {gx} 各自标准化后成图，用于消去随 {gy} 变化的整体水平、"
            "只比较各行形状；颜色此时不再是实测值，"
            "不同行之间同色不代表同值，默认口径可在 normalize 里改成按另一轴",
            enc.copy(),
            2,
        )
    else:
        partial = _partial_grid(data, eligible)
        if partial:
            gx, gy, gz, coverage, holes = partial
            add(
                "heatmap",
                "二维参数热图（缺测留白）",
                "medium",
                f"{gx} × {gy} 的规则扫描只覆盖 {coverage:.0%} 的格点，缺 {holes} 个；"
                "热图把缺测格留白而不是插值补齐，因此不能对全区域积分或求均值，"
                "等值线在这种网格上会跨缺测区插值，故不推荐",
                {"x": gx, "y": gy, "z": gz},
                1,
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
                "high",
                "存在 source/target 边列表，节点和箭头可直接由已提供关系构建",
                {"source": p.source_column, "target": p.target_column},
                0,
            )
    if p.n_rows == 1:
        add(
            "table",
            "数据表",
            "high",
            "只有一行观测，表格可完整保留数值与标签",
            {"columns": p.columns},
            0,
        )
        return by_tier(out)

    group = next(
        (c for c in p.categorical_columns if 2 <= data[c].nunique() <= min(12, p.n_rows // 2)), None
    )
    x = next((c for c in p.axis_columns if c in eligible), eligible[0] if eligible else None)
    if x and group is None:
        group = _parameter_group(
            data, x, [c for c in p.axis_columns if c != x], p.n_rows
        )
    responses = [c for c in eligible if c != x and c != group and c not in p.angle_columns]
    y = responses[0] if responses else None
    stokes = _stokes_columns(p.numeric_columns)
    if stokes:
        add(
            "poincare_sphere",
            "Poincaré 球与偏振态轨迹",
            "high",
            f"{stokes[1]}…{stokes[3]} 以 {stokes[0]} 归一后画进球内；"
            "点落在球内不强行投影到球面——半径本身就是偏振度，"
            "S0≤0 或分量不全的行不参与；若有点在球外说明 Stokes 不自洽，仍按实测量画出",
            {"stokes": stokes, "label": x or "measurement"},
            0,
        )
    if p.angle_columns and not p.grid_like:
        theta = p.angle_columns[0]
        radius = next((c for c in eligible if c != theta and c not in p.axis_columns), None)
        if radius is None:
            radius = next((c for c in eligible if c != theta), None)
        repeats = bool(
            radius and data[[theta, radius]].dropna().duplicated(theta).any()
        )
        polar_group = None
        if repeats:
            # One radius per angle is what makes a polar response a curve. Where
            # several share an angle the figure is only honest once something
            # splits them -- a channel label, or the second scan axis -- and the
            # split has to leave each group tracing the circle once.
            for candidate in [group] + [c for c in p.axis_columns if c != theta]:
                if not candidate or candidate == theta:
                    continue
                trio = data[[theta, radius, candidate]].dropna()
                levels = trio[candidate].nunique()
                if 2 <= levels <= 8 and not trio.duplicated([theta, candidate]).any():
                    polar_group = candidate
                    break
        if (
            radius
            and _valid_count(data, [theta, radius]) >= 3
            and (data[radius].dropna() >= 0).all()
            and (not repeats or polar_group is not None)
        ):
            enc = {"theta": theta, "r": radius, "angle_unit": p.angle_units.get(theta, "deg")}
            # Grouping a polar plot by its own angle leaves one radius per group,
            # which is a scatter of dots with a legend, not a comparison.
            if polar_group:
                enc["group"] = polar_group
            elif group and group != theta:
                enc["group"] = group
            add(
                "polar",
                "极坐标响应图",
                "high",
                "列名含角度信息且径向值非负；请核对角度单位及周期含义",
                enc,
                2,
            )
            span = float(data[theta].dropna().max() - data[theta].dropna().min())
            full = 360.0 if p.angle_units.get(theta, "deg") == "deg" else 2 * math.pi
            if span > full:
                # More than one turn lands on the same ray, so the figure cannot
                # show which pass a given point belongs to.
                add(
                    "polar_and_cartesian",
                    "极坐标 + 直角双显示",
                    "high",
                    f"{theta} 的扫描跨度 {span:g} 超过一圈，"
                    "不同圈的同一角度落在同一条射线上，极坐标单独看不出是哪一圈；"
                    "右侧直角面板保留展开后的顺序",
                    enc.copy(),
                    1,
                )
            positive = data[radius].dropna()
            # Zero is allowed and reported: a pattern with a true null still has a
            # meaningful dB skirt, and the renderer counts what it cannot place.
            if len(positive) >= 3 and (positive >= 0).all():
                add(
                    "polar_db",
                    "分贝刻度方向图",
                    "medium",
                    f"半径改为 10·log10({radius}/最大值)，旁瓣与后瓣才看得见——"
                    "线性半径会把它们压成看不见的一圈；"
                    f"若 {radius} 是场幅而不是功率，须把 db_factor 改成 20，"
                    "用错因子是 2 倍的标度误差，图上会标出用的是哪一个",
                    enc.copy(),
                    2,
                )

    if x and y and _valid_count(data, [x, y]) >= 2:
        enc = {"x": x, "y": y}
        if group:
            enc["group"] = group
        add(
            "scatter_fit",
            "散点关系图",
            "medium",
            "至少两个数值变量有配对观测，可检查关系；默认不拟合",
            enc,
            3,
        )
        # Hexbin counts samples per region. On a complete measurement grid every
        # cell holds one sample, so the map is uniform and the heatmap already
        # shows the measured value directly.
        if _valid_count(data, [x, y]) >= 300 and not p.grid_like:
            add(
                "density",
                "二维密度 / 六边形分箱",
                "high",
                "配对观测较多，分箱可减轻散点重叠；颜色表示各区域样本数",
                {"x": x, "y": y},
                5,
            )
        line_ys = [c for c in responses if _valid_count(data, [x, c]) >= 3]
        keys = [x] + ([group] if group else [])
        line_unique = not data[keys].dropna().duplicated(keys).any()
        # A complete grid disqualifies one line, because joining the raster order
        # zigzags across the map. Once a second scan axis names the groups, each
        # group is its own ordered sweep and the family is the honest view.
        if (
            line_ys
            and p.n_rows >= 4
            and line_unique
            and (not p.grid_like or group)
            and (_axis_rank(x) < 99 or _ordered(data[x]))
        ):
            enc = {"x": x, "y": line_ys[:8]}
            if group:
                enc["group"] = group
            add(
                "spectrum_lines",
                "有序响应 / 光谱曲线",
                "high",
                "横轴名称含扫描轴信息，或观测按横轴有序；曲线显示采样点间变化",
                enc,
                6,
            )
            # A labelled peak belongs to a single curve. Once a second scan axis
            # splits the rows into a family, the per-curve peak is what
            # peak_evolution reports, and labelling one here would put a marker
            # on a zigzag through fifteen spectra.
            if not group and not p.grid_like:
                peaked = [
                    c
                    for c in line_ys
                    if _has_interior_extremum(data[[x, c]].dropna().sort_values(x)[c])
                ]
                if peaked:
                    add(
                        "peak_annotation",
                        "曲线与峰位标注",
                        "high",
                        f"{peaked[0]} 在扫描范围内存在内部极值，可标注峰位与半高宽；"
                        "峰位由三点抛物线插值给出，半高宽以曲线端点本底为参考，"
                        "两个半高交点任一落在扫描之外时不给宽度",
                        {"x": x, "y": peaked[:4]},
                        6,
                    )
            # The same family stacked instead of overlaid. Offered only for one
            # response: eight columns times fifteen angles is a wall, not a plot.
            if group and len(line_ys) >= 1 and 2 <= data[group].dropna().nunique() <= 20:
                add(
                    "stacked_curves",
                    "堆叠曲线族",
                    "high",
                    f"把 {group} 的每条曲线按固定偏移逐条上移，重叠区不再互相遮挡，"
                    "适合看峰形随参数的移动；每条曲线的基线只是排版偏移，"
                    "不是物理零点，纵坐标绝对值只能在本条曲线内部解读",
                    {"x": x, "y": line_ys[:1], "group": group},
                    7,
                )

        gap = None if p.grid_like else _widest_x_gap(data[x])
        if gap:
            lo, hi, ratio = gap
            bridged = []
            for c in line_ys:
                xs = data[[x, c]].dropna()[x].to_numpy()
                if int((xs <= lo).sum()) >= 3 and int((xs >= hi).sum()) >= 3:
                    bridged.append(c)
            if bridged:
                add(
                    "broken_spectrum",
                    "断轴光谱曲线",
                    "high",
                    f"{x} 在 {lo:g}–{hi:g} 之间是空的（约为采样间隔的 {ratio:g} 倍），"
                    "连续坐标轴会把大部分画布留给这段空洞；断轴后两侧各自展开读数，"
                    "两侧共用纵轴刻度所以峰高仍可比，横向斜率跨过断口不可比",
                    {"x": x, "y": bridged[:8], "gap": [lo, hi]},
                    7,
                )

        # --- derived comparisons between response columns on the same scan grid.
        # All of these require the columns to share x values row by row; nothing
        # is interpolated onto a common grid, because that would invent samples.
        # They also require the columns to measure the same quantity, because
        # subtracting a position from an intensity is arithmetically valid and
        # physically empty.
        measured = [c for c in line_ys if _valid_count(data, [x, c]) >= 4]
        pairs = [
            (a, b)
            for i, a in enumerate(measured)
            for b in measured[i + 1 :]
            if _comparable(a, b)
        ]
        if pairs:
            a, b = pairs[0]
            both = data[[x, a, b]].dropna()
            if len(both) >= 4:
                denc = {"x": x, "y": [a, b]}
                if group:
                    denc["group"] = group
                add(
                    "spectral_difference",
                    "两列差值 A − B",
                    "medium",
                    f"{a} 与 {b} 是同一网格上的同类量，可逐点相减（有效配对 {len(both)} 行）；"
                    "顺序决定符号，请确认是 A−B 而非 B−A",
                    denc,
                    0,
                )
                denominator = both[b].to_numpy()
                live = int((np.abs(denominator) > 1e-12 * max(np.abs(denominator).max(), 1e-30)).sum())
                if live >= 4:
                    add(
                        "spectral_ratio",
                        "两列比值 A / B",
                        "medium",
                        f"{a} 与 {b} 同类，可逐点相除；{len(both) - live} 行因分母接近零被屏蔽，"
                        "比值在屏蔽点附近不可信",
                        denc,
                        1,
                    )
        alike = [c for c in measured if c and pairs and _comparable(pairs[0][0], c)]
        if len(alike) >= 3:
            add(
                "spectral_envelope",
                "多列包络带（最小–最大）",
                "medium",
                f"{len(alike)} 列同类且同网格，可显示取值范围；"
                "这是极差不是标准差，列数少时会高估分散程度",
                {"x": x, "y": alike[:12]},
                2,
            )

        # A family of curves is also a family of *deviations*: both views below
        # align members on the coordinates they actually share, and neither
        # invents a point where one member is missing.
        n_levels = data[group].dropna().nunique() if group else 0
        grouped_family = bool(group and line_ys and 2 <= n_levels <= 20)
        if grouped_family or len(alike) >= 2:
            if grouped_family:
                family = {"x": x, "y": line_ys[:1], "group": group}
                members = f"{group} 的 {n_levels} 条曲线"
                first = data[group].dropna().iloc[0]
                reference = f"文件里第一条 {group}={_level_text(first)}，用 reference 可换"
            else:
                family = {"x": x, "y": alike[:8]}
                members = f"{alike[0]} 等 {len(alike)} 列"
                reference = f"第一列 {alike[0]}，用 reference 可换"
            add(
                "difference_family",
                "逐条减参考曲线",
                "medium",
                f"{members}落在同一批横坐标上，可全部减去参考条（{reference}）；"
                "差值取「本条 − 参考」所以上升读成正，参考条按定义是恒零线、"
                "不代表测到了零，两条曲线在某点任一方缺失则该点不作差而不插值补齐",
                dict(family),
                8,
            )
            add(
                "curves_normalized",
                "曲线归一化重标",
                "medium",
                f"{members}可各除以自己的最大偏离后叠在一起比线形与峰位；"
                "这一步会丢掉绝对强度差，而强度差常常正是结果（荧光淬灭、增益阈值、"
                "响应度标定），所以归一化图不能代替原始曲线图",
                dict(family),
                9,
            )
        if _is_wavelength(x) and (data[x].dropna() > 0).all():
            add(
                "energy_axis",
                "波长轴 + 光子能量副轴",
                "medium",
                f"{x} 为正的波长量，可加第二条上横轴按 E = 1239.84/λ 标注光子能量；"
                "副轴只是同一数据的另一种单位，不增加信息",
                {"x": x, "y": measured[:8] or line_ys[:8], "secondary_unit": "eV"},
                3,
            )

        dual = [
            (a, b)
            for i, a in enumerate(measured)
            for b in measured[i + 1 :]
            if _distinct_quantity(a, b)
        ]
        if dual:
            a, b = dual[0]
            both = data[[x, a, b]].dropna()
            if len(both) >= 4:
                add(
                    "dual_axis",
                    "双纵轴（不同量纲）",
                    "medium",
                    f"{a} 与 {b} 单位不同（{_unit_of(a)} / {_unit_of(b)}），可各占一条纵轴；"
                    "两条轴的比例独立可调，**曲线看起来同步不代表二者相关**，"
                    "要论证关系请改用散点图",
                    {"x": x, "y": [a, b], "right": b},
                    4,
                )

        if measured and _strictly_ordered(data[x]) and _valid_count(data, [x, measured[0]]) >= 8:
            add(
                "spectral_derivative",
                "响应导数 d/dλ",
                "medium",
                f"{x} 单调递增，可对 {len(measured)} 列求导以定位拐点与肩峰；"
                "微分会放大噪声，若原始数据本身有抖动请先说明再使用",
                {"x": x, "y": measured[:8]},
                5,
            )

        # --- scale diagnostics. A logged axis is not a cosmetic toggle: it asserts
        # that one particular function family is being tested, so the reason names
        # the family and what the axis still cannot settle.
        x_span = _decade_span(data[x])
        if x_span >= 2.0:
            loggable = [c for c in measured if _decade_span(data[c]) >= 1.0]
            if loggable:
                add(
                    "log_log",
                    "双对数曲线（幂律诊断）",
                    "medium",
                    f"{x} 全为正值且跨 {x_span:.1f} 个数量级，{loggable[0]} 等 "
                    f"{len(loggable)} 列同样全为正值，可画双对数；幂律 y = C·x^k 在这里"
                    "是一条直线，但目视斜率随画布长宽比改变，要报出 k 得点名 power_law "
                    "拟合，log_aspect_equal 只能让目视斜率与 k 对上而不给出数值",
                    {"x": x, "y": loggable[:8]},
                    6,
                )
        y_span = max([_decade_span(data[c]) for c in measured], default=0.0)
        decayed = [c for c in measured if _decade_span(data[c]) >= 1.0]
        if decayed:
            add(
                "semi_log",
                "半对数曲线（指数诊断）",
                "medium",
                f"{decayed[0]} 等 {len(decayed)} 列全为正值且纵轴跨 {y_span:.1f} 个数量级，"
                "线性纵轴上尾段会被压成贴着零线；半对数下指数衰减 y = A·exp(-x/τ) 是一条"
                "直线，但直线只说明与指数族相符——它与幂律尾部区分不开，要分开请看拟合"
                "残差面板而不是这条轴",
                {"x": x, "y": decayed[:8]},
                7,
            )

        if _strictly_ordered(data[x]) and measured:
            integrated = [c for c in measured if _valid_count(data, [x, c]) >= 8]
            if integrated:
                add(
                    "cumulative_response",
                    "累积响应曲线",
                    "medium",
                    f"{x} 严格单调，{integrated[0]} 等 {len(integrated)} 列可沿横轴累积成一条"
                    "单调曲线，用来回答「前一半区间贡献了多少」；积分只走相邻实测点之间的"
                    "梯形，缺测处断线不桥接，因此末值是已扫区间的累计而不是全量程总量，"
                    f"纵轴单位是 {integrated[0]} 单位乘以 {x} 单位",
                    {"x": x, "y": integrated[:8]},
                    8,
                )

    # Peak tracking is a reduction, not a grid view: it needs a wavelength axis,
    # a second swept parameter, and a response, and it deliberately does not
    # require the grid to be complete, since a missing reading only costs
    # resolution along one curve.
    track = _peak_trackable(p, data)
    if track:
        add(
            "peak_evolution",
            "峰位与半高宽随参数演化",
            "medium",
            f"对每个 {track['param']} 取值在 {track['wave']} 上定位 {track['value']} 的极值，"
            "得到峰位与半高宽两条曲线；这是把二维图压成一维趋势，"
            "多峰或无清晰峰的行会被跳过并计数，请核对被跳过的角度/温度",
            track,
            6,
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
                    "high",
                    "存在与响应列配对且非负的误差列，保留其已有统计定义",
                    enc,
                    4,
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
                "medium",
                "至少两个横轴取值具有组内重复观测，可计算样本标准差；请确认这些确为重复实验",
                enc,
                0,
            )

    if eligible:
        value = y or eligible[0]
        if data[value].notna().sum() >= 3:
            only_number = len(eligible) == 1
            add(
                "distribution",
                "数值分布直方图",
                "medium" if only_number else "low",
                "有效观测可用直方图检查分布和离群值；分箱会影响外观",
                {"value": value},
                4 if only_number else 1,
            )
        if group and _valid_count(data, [group, value]) >= 4:
            sizes = data[[group, value]].dropna().groupby(group, observed=True)[value].count()
            if (sizes >= 2).sum() >= 2:
                add(
                    "box",
                    "分组箱线图与原始点",
                    "high",
                    "至少两组有重复数值观测，可比较中位数、四分位与原始点",
                    {"group": group, "value": value},
                    7,
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
                "low",
                "多个变化数值列可检查线性相关；使用成对有效观测，相关不代表因果",
                {"columns": used[:16]},
                0,
            )
    only_table_works = not eligible
    add(
        "table",
        "数据表",
        "high" if only_table_works else "low",
        "表格保留原始标签、数值与缺失项，适合逐项核对和小样本呈现",
        {"columns": p.columns},
        0 if only_table_works else 2,
    )
    return by_tier(out)
