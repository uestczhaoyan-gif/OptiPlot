"""Physical models a user can name and fit, and the fit itself.

Nothing here runs unless the user picks a model by id. The engine's stance is
that a curve through data is a claim, not a rendering, so the choice stays with
the person who knows what the columns are; what this module adds is the ability
to make that claim properly, with the residuals alongside it.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitModel:
    """One model: what it is called, what it computes, and how to start it.

    `guess` and `bounds` see the data because every sensible starting value here
    is read off the curve itself - a peak height, a half-width, an endpoint
    baseline. A fixed initial vector would make the fit's success depend on
    whichever units the instrument happened to export.
    """

    id: str
    label: str
    formula: str
    parameters: tuple[str, ...]
    evaluate: Callable[..., np.ndarray]
    guess: Callable[..., list[float]]
    min_points: int
    bounds: Callable[[np.ndarray, np.ndarray], tuple[list[float], list[float]]] | None = None
    starts: Callable[..., list[list[float]]] | None = None
    x_positive: bool = False
    x_unit: bool = False
    x_hint: tuple[str, ...] = ()
    y_hint: tuple[str, ...] = ()
    caveat: str = ""


def _span(x):
    return float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0


def _baseline(x, y):
    """The flatter end of the curve, used as the offset a decay settles to."""
    head, tail = np.nanmean(y[: max(1, len(y) // 5)]), np.nanmean(y[-max(1, len(y) // 5) :])
    return float(tail if abs(tail) < abs(head) else head)


def _single_exp(x, A, tau, C, x_factor=1.0):
    return C + A * np.exp(-x / tau)


def _double_exp(x, A1, tau1, A2, tau2, C, x_factor=1.0):
    return C + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)


def _power_law(x, A, k, C, x_factor=1.0):
    return C + A * np.power(x, k)


def _gaussian(x, A, mu, sigma, C, x_factor=1.0):
    return C + A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _lorentzian(x, A, mu, gamma, C, x_factor=1.0):
    return C + A * gamma**2 / ((x - mu) ** 2 + gamma**2)


def _drude_lorentz(x, n_inf, strength, lambda0, x_factor=1.0):
    """One oscillator's contribution to a dispersion curve, in wavelength form.

    The oscillator term is written as S·λ₀²/(λ²−λ₀²) rather than S·λ²/(λ²−λ₀²)
    because the latter tends to a constant far from the pole, and a constant that
    the background parameter can already absorb leaves the two of them without a
    unique answer no matter how good the curve looks.
    """
    return n_inf + strength * lambda0**2 / (x**2 - lambda0**2)


def _malus(x, A, order, phase, C, x_factor=1.0):
    return C + A * np.cos(x_factor * order * x + phase) ** 2


def _peak_shape_guess(x, y, fraction):
    """Amplitude, centre and width read off the tallest interior excursion."""
    step = float(np.median(np.diff(np.sort(x)))) or 1.0
    i = int(np.argmax(y))
    base = float(np.median(np.concatenate([y[:3], y[-3:]])))
    height = float(y[i]) - base
    half = base + height / 2.0
    above = x[y > half]
    width = float(above.max() - above.min()) if above.size > 1 else 4.0 * step
    return height, float(x[i]), max(width / 2.0, step), base


def _malus_starts(x, y, x_factor=1.0):
    """Several starting orders, because cos² is periodic and a curve showing two
    humps could be cos²(θ) over 360° or cos²(9θ) over 40°.

    The textbook low orders are tried alongside the orders that would place a
    whole number of periods inside this particular span, which is the only way
    the guess can mean anything without knowing the angle unit in advance.
    """
    span = _span(x) or 1.0
    amplitude = float(np.nanmax(y) - np.nanmin(y))
    floor = float(np.nanmin(y))
    # A curve of n points cannot show more than about n/4 periods of anything, so
    # that is where the search stops; below it every order is tried, because the
    # cheap ones are the physical ones and there is no way to tell from one hump
    # whether this is cos²(θ) over 360° or cos²(9θ) over 40°.
    limit = max(8, min(len(x) // 4, 48))
    periodic = {m * math.pi / (x_factor * span) for m in range(1, limit + 1)}
    orders = sorted({0.5, 1.0, 2.0, 3.0, 4.0} | {o for o in periodic if np.isfinite(o) and o > 0})
    return [[amplitude, order, 0.0, floor] for order in orders]


MODELS: tuple[FitModel, ...] = (
    FitModel(
        id="single_exponential",
        label="单指数衰减",
        formula="y = C + A·exp(−x/τ)",
        parameters=("A", "tau", "C"),
        evaluate=_single_exp,
        guess=lambda x, y, x_factor=1.0: [float(y[0] - _baseline(x, y)), _span(x) / 3.0, _baseline(x, y)],
        bounds=lambda x, y: ([-np.inf, 1e-12 * max(_span(x), 1.0), -np.inf], [np.inf] * 3),
        min_points=4,
        x_hint=("time", "delay", "t_ns", "t_ps", "duration"),
        y_hint=("decay", "lifetime", "intensity", "signal", "photoluminescence"),
        caveat="τ 只有在数据跨过数个 τ 且末端真正趋平时才可辨识；扫描在衰减完成前截断时，C 会把 τ 拉长。",
    ),
    FitModel(
        id="double_exponential",
        label="双指数衰减",
        formula="y = C + A₁·exp(−x/τ₁) + A₂·exp(−x/τ₂)",
        parameters=("A1", "tau1", "A2", "tau2", "C"),
        evaluate=_double_exp,
        guess=lambda x, y, x_factor=1.0: [
            float(y[0] - _baseline(x, y)),
            _span(x) / 10.0,
            float(_baseline(x, y) - y[-1]) or 0.1,
            _span(x) / 1.5,
            _baseline(x, y),
        ],
        bounds=lambda x, y: (
            [-np.inf, 1e-12 * max(_span(x), 1.0), -np.inf, 1e-12 * max(_span(x), 1.0), -np.inf],
            [np.inf] * 5,
        ),
        min_points=8,
        x_hint=("time", "delay", "t_ns", "t_ps", "duration"),
        y_hint=("decay", "lifetime", "intensity", "signal", "photoluminescence"),
        caveat="两个时间常数接近时不可分辨；先确认单指数残差是否已有结构，再决定要不要第二个分量。",
    ),
    FitModel(
        id="power_law",
        label="幂律",
        formula="y = C + A·xᵏ",
        parameters=("A", "k", "C"),
        evaluate=_power_law,
        guess=lambda x, y, x_factor=1.0: [float(np.nanmax(y) - np.nanmin(y)), 1.0, float(np.nanmin(y))],
        bounds=lambda x, y: ([-np.inf, -20.0, -np.inf], [np.inf, 20.0, np.inf]),
        min_points=5,
        x_positive=True,
        x_hint=("distance", "power", "intensity", "fluence", "radius"),
        y_hint=("intensity", "field", "signal", "efficiency"),
        caveat="横轴跨度不足一个数量级时 A 与 C 互相吸收，k 只在跨数量级数据上稳定。",
    ),
    FitModel(
        id="gaussian",
        label="高斯线型",
        formula="y = C + A·exp(−(x−μ)²/2σ²)",
        parameters=("A", "mu", "sigma", "C"),
        evaluate=_gaussian,
        guess=lambda x, y, x_factor=1.0: list(_peak_shape_guess(x, y, 0.5)),
        bounds=lambda x, y: (
            [-np.inf, float(np.nanmin(x)), 1e-9, -np.inf],
            [np.inf, float(np.nanmax(x)), _span(x) * 10.0, np.inf],
        ),
        min_points=6,
        x_hint=("wavelength", "frequency", "angle", "detuning"),
        y_hint=("intensity", "transmission", "reflectance", "absorbance", "signal"),
        caveat="基线 C 与峰高相关；峰宽接近采样间隔时 σ 由网格决定而不是由样品决定。",
    ),
    FitModel(
        id="lorentzian",
        label="洛伦兹线型",
        formula="y = C + A·γ²/((x−μ)²+γ²)",
        parameters=("A", "mu", "gamma", "C"),
        evaluate=_lorentzian,
        guess=lambda x, y, x_factor=1.0: list(_peak_shape_guess(x, y, 0.5)),
        bounds=lambda x, y: (
            [-np.inf, float(np.nanmin(x)), 1e-9, -np.inf],
            [np.inf, float(np.nanmax(x)), _span(x) * 10.0, np.inf],
        ),
        min_points=6,
        x_hint=("wavelength", "frequency", "angle", "detuning"),
        y_hint=("intensity", "transmission", "reflectance", "absorbance", "signal"),
        caveat="基线 C 与峰高相关；两翼若被相邻峰占据，γ 会吸收邻峰强度。",
    ),
    FitModel(
        id="drude_lorentz",
        label="Drude-Lorentz 色散",
        formula="n = n∞ + S·λ₀²/(λ²−λ₀²)",
        parameters=("n_inf", "strength", "lambda0"),
        evaluate=_drude_lorentz,
        guess=lambda x, y, x_factor=1.0: [float(np.nanmin(y)), _span(x) * 10.0, float(np.nanmin(x)) / 2.0],
        min_points=5,
        x_positive=True,
        x_hint=("wavelength", "lambda", "nm"),
        y_hint=("refractive", "index", "n", "epsilon", "dielectric"),
        caveat="远离极点时只有 n∞ 与 S·λ₀² 的组合可辨识；单独的 λ₀ 要求数据接近该极点，否则曲线看起来对而参数无意义。",
    ),
    FitModel(
        id="malus",
        label="Malus 定律",
        formula="y = C + A·cos²(k·θ + φ)",
        parameters=("A", "k", "phi", "C"),
        evaluate=_malus,
        guess=lambda x, y, x_factor=1.0: [float(np.nanmax(y) - np.nanmin(y)), 1.0, 0.0, float(np.nanmin(y))],
        starts=_malus_starts,
        bounds=lambda x, y: (
            [-np.inf, 1e-6, -4.0 * np.pi, -np.inf],
            [np.inf, 50.0, 4.0 * np.pi, np.inf],
        ),
        min_points=6,
        x_unit=True,
        x_hint=("angle", "theta", "analyzer", "polarizer", "deg"),
        y_hint=("power", "intensity", "transmission"),
        caveat="φ 与 k 的奇偶简并给出同一条曲线；消光比接近零时 A 与 C 高度相关，此时应直接报最小值而不是报比值。",
    ),
)

MODEL_BY_ID = {m.id: m for m in MODELS}


@dataclass(frozen=True)
class FitResult:
    """A fitted model plus the evidence for and against it.

    `residual` is not optional decoration: a model can score R² = 0.99 and still
    be the wrong shape, and the residual column is the only place that shows it.
    """

    model: FitModel
    params: dict = field(default_factory=dict)
    r2: float = float("nan")
    rmse: float = float("nan")
    residual: np.ndarray = field(default_factory=lambda: np.array([]))
    x_factor: float = 1.0
    notes: tuple[str, ...] = ()

    def curve(self, x):
        return self.model.evaluate(np.asarray(x, dtype=float), *self._vector(), x_factor=self.x_factor)

    def _vector(self):
        return [self.params[name] for name in self.model.parameters]

    def parameter_text(self) -> str:
        return ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())

    def label(self) -> str:
        return f"{self.model.label}: {self.parameter_text()}; R²={self.r2:.3f}"


def model_for(identifier: str) -> FitModel:
    try:
        return MODEL_BY_ID[identifier]
    except KeyError:
        raise ValueError(
            f"未知拟合模型 {identifier!r}；可选：" + "、".join(MODEL_BY_ID)
        ) from None


def fit_model(model: FitModel, x, y, x_factor: float = 1.0) -> FitResult:
    """Least-squares fit of one named model, or a Chinese-language refusal.

    Refusals matter more than results here: a fit that silently returns garbage
    is the exact failure this project exists to avoid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < model.min_points:
        raise ValueError(
            f"{model.label} 需要至少 {model.min_points} 个有效数据点，当前只有 {x.size} 个。"
        )
    if np.unique(x).size < len(model.parameters):
        raise ValueError(
            f"{model.label} 有 {len(model.parameters)} 个参数，"
            f"但横轴只有 {np.unique(x).size} 个不同取值，参数不可辨识。"
        )
    if model.x_positive and float(np.nanmin(x)) <= 0:
        raise ValueError(f"{model.label} 要求横轴全部为正（含零时幂律或色散式无定义）。")

    notes: list[str] = []
    if model.id == "drude_lorentz":
        return _fit_drude_branches(x, y, model)
    lower, upper = ([], []) if model.bounds is None else model.bounds(x, y)
    starts = (
        model.starts(x, y, x_factor=x_factor)
        if model.starts
        else [model.guess(x, y, x_factor=x_factor)]
    )
    best = _best_start(model, x, y, starts, lower, upper, x_factor)
    if best is None:
        raise ValueError(
            f"{model.label} 未收敛：从数据估出的 {len(starts)} 组初值都找不到极小值，"
            "请检查横轴单位与数据是否真的服从该模型。"
        )
    popt, pcov = best
    return _package(model, x, y, popt, pcov, x_factor, notes)


def _least_squares(model, x, y, p0, lower, upper, x_factor):
    try:
        popt, pcov = curve_fit(
            lambda xx, *p: model.evaluate(xx, *p, x_factor=x_factor),
            x,
            y,
            p0=_inside(p0, lower, upper),
            bounds=(lower, upper) if lower else (-np.inf, np.inf),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(model.evaluate(x, *popt, x_factor=x_factor))):
        return None
    return popt, pcov


def _best_start(model, x, y, starts, lower, upper, x_factor):
    """Least squares from every candidate start, keeping the deepest minimum.

    A periodic model has many local minima and a multi-exponential one has
    near-degenerate ones; reporting whichever the first guess happened to reach
    would make the answer depend on the guess rather than on the data.
    """
    best = None
    for p0 in starts:
        found = _least_squares(model, x, y, p0, lower, upper, x_factor)
        if found is None:
            continue
        popt, pcov = found
        ssr = float(np.sum((y - model.evaluate(x, *popt, x_factor=x_factor)) ** 2))
        if not np.isfinite(ssr):
            continue
        if best is None or ssr < best[0]:
            best = (ssr, popt, pcov)
    return None if best is None else (best[1], best[2])


def _fit_drude_branches(x, y, model):
    """Fit with the oscillator pole on the short side, then the long side.

    The choice is not a free parameter: the pole must stay outside the data or
    the curve is fitted through an asymptote. Which side it is on is a physical
    claim about the material, so the rejected branch is reported as a note.
    """
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    starts = [[float(np.nanmin(y)), hi - lo, lo / 2.0], [float(np.nanmin(y)), hi - lo, hi * 2.0]]
    best = None
    for lower, upper, side in (
        ([0.0, -np.inf, 0.0], [np.inf, np.inf, lo * 0.999], "短波侧"),
        ([0.0, -np.inf, hi * 1.001], [np.inf, np.inf, np.inf], "长波侧"),
    ):
        found = _best_start(model, x, y, starts, lower, upper, 1.0)
        if found is None:
            continue
        popt, pcov = found
        ssr = float(np.sum((y - model.evaluate(x, *popt)) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, popt, pcov, side)
    if best is None:
        raise ValueError(
            f"{model.label} 未收敛：共振波长 λ₀ 无法落在数据范围之外，"
            "说明数据跨越了吸收峰，此时 n 的实部本身不再服从单振子色散式。"
        )
    _, popt, pcov, side = best
    other = "长波侧" if side == "短波侧" else "短波侧"
    return _package(
        model, x, y, popt, pcov, 1.0,
        [f"共振波长 λ₀ 取在数据范围的{side}；另一分支（{other}）残差更大，未采用。"],
    )


def _inside(guess, lower, upper):
    """Nudge initial values into the bounds; scipy rejects a start outside them."""
    out = []
    for i, value in enumerate(guess):
        if lower and np.isfinite(lower[i]):
            value = max(value, lower[i] * (1.0 + 1e-6) + 1e-12)
        if upper and np.isfinite(upper[i]):
            value = min(value, upper[i] * (1.0 - 1e-6) - 1e-12)
        out.append(value if np.isfinite(value) else 0.0)
    return out


def _package(model, x, y, popt, pcov, x_factor, notes):
    predicted = model.evaluate(x, *popt, x_factor=x_factor)
    residual = y - predicted
    total = float(np.sum((y - y.mean()) ** 2))
    ssr = float(np.sum(residual**2))
    r2 = 1.0 - ssr / total if total > 0 else float("nan")
    if total <= 0:
        notes.append("响应列没有变化，R² 无定义。")
    if not np.all(np.isfinite(pcov)):
        notes.append("参数协方差无法估计：模型可能过参数化或存在简并，参数值不应作为唯一解引用。")
    if np.isfinite(r2) and r2 < 0.8:
        notes.append(f"R²={r2:.3f} 偏低，模型形状与数据不符，请检查残差面板。")
    return FitResult(
        model=model,
        params=dict(zip(model.parameters, [float(v) for v in popt])),
        r2=float(r2),
        rmse=float(np.sqrt(ssr / max(1, y.size))),
        residual=residual,
        x_factor=x_factor,
        notes=tuple(notes),
    )
