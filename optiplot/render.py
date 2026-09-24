"""Deterministic publication renderers. No fitting, smoothing or interpolation by default."""

from pathlib import Path
import math
from typing import NamedTuple
import warnings
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:  # normal import, as part of the optiplot package
    from .style import EXPORT_FORMATS, Style
    from .models import MODEL_BY_ID, fit_model, model_for
except ImportError:  # standalone copy inside an exported reproducible bundle
    from style import EXPORT_FORMATS, Style
    from models import MODEL_BY_ID, fit_model, model_for

_MODEL_IDS = frozenset(MODEL_BY_ID)
FIT_CHOICES = ("none", "linear", *MODEL_BY_ID)
# Types a fit can be applied to. The rest ignore the option, so an interface has
# to say so rather than leave a control that does nothing look like it worked.
FITTABLE = ("scatter_fit", "spectrum_lines")

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
# The authoritative list of drawable figure types. A new branch in _draw is not
# a supported figure type until it appears here; catalog/styles.json may only
# reference these ids.
FIGURE_TYPES = (
    "spectrum_lines",
    "peak_annotation",
    "stacked_curves",
    "broken_spectrum",
    "spectral_difference",
    "spectral_ratio",
    "spectral_envelope",
    "spectral_derivative",
    "energy_axis",
    "dual_axis",
    "peak_evolution",
    "scatter_fit",
    "density",
    "errorbar",
    "heatmap",
    "heatmap_contours",
    "heatmap_marginals",
    "heatmap_normalized",
    "contour",
    "matrix_heatmap",
    "surface_3d",
    "surface_with_contour",
    "polar",
    "polar_db",
    "polar_and_cartesian",
    "mueller_matrix",
    "poincare_sphere",
    "distribution",
    "box",
    "correlation",
    "flow",
    "table",
)

# Views that paint a measured two-dimensional grid as colour on a flat axes.
# Named for what they route, so a future three-dimensional "surface" type cannot
# be swept in by the word alone.
PAINTED_GRID_KINDS = (
    "heatmap",
    "heatmap_contours",
    "heatmap_marginals",
    "heatmap_normalized",
    "contour",
    "matrix_heatmap",
)
# Which coordinate is held fixed when each line of the grid is standardised.
NORMALIZATIONS = ("per_y", "per_x")
# Angular response views: one polar frame, one dB frame, and one that pairs the
# polar frame with the same data on a straight axis.
POLAR_KINDS = ("polar", "polar_db", "polar_and_cartesian")


# A difference crosses zero by construction, so a log axis there is meaningless;
# the others are ratios or positive-valued spectra.
LOG_AXIS_TYPES = (
    "spectrum_lines",
    "peak_annotation",
    "spectral_ratio",
    "spectral_envelope",
    "energy_axis",
    "scatter_fit",
    "density",
    "errorbar",
    "distribution",
)


def _finite(df, names):
    cols = list(dict.fromkeys(names))
    out = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("所选变量没有完整且有限的数值配对。")
    return out


def _spectral_pair(df, e):
    """Rows where the scan axis and exactly two response columns are all present.

    The two columns are subtracted or divided row by row, so a shared grid is a
    precondition, not a detail: nothing here resamples one series onto the other's
    wavelengths.
    """
    xname = e["x"]
    ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
    if len(ys) != 2:
        raise ValueError("差值与比值图需要恰好两个响应列，当前给出 " + str(len(ys)) + " 个。")
    d = df[[xname] + ys].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d.dropna()
    if len(d) < 2:
        raise ValueError("两列在同一采样网格上没有足够的有效配对行。")
    return d, xname, ys[0], ys[1]


def _legend(ax, style, **overrides):
    """One place that decides whether a legend exists at all.

    Everything else about it - frame, font size, columns, handle length,
    spacing - arrives from Style through rcParams, so a renderer branch only
    ever passes a per-type placement override.
    """
    if style.legend == "none":
        return None
    return ax.legend(**style.legend_kwargs(), **overrides)


def _colorbar(fig, m, ax, label, style, dedicated=False):
    """Artist-level styling: one of the two routing points rcParams cannot reach.

    `dedicated` means `ax` is a colour-bar-only axes already placed by a
    gridspec; shrinking a parent axes is a different operation and the two must
    not be confused, or the bar steals canvas from the panel beside it.
    """
    if dedicated:
        cb = fig.colorbar(m, cax=ax, label=label)
    else:
        cb = fig.colorbar(
            m, ax=ax, label=label, fraction=style.colorbar_thickness, pad=style.colorbar_pad
        )
    cb.outline.set_linewidth(0.7)
    cb.ax.yaxis.label.set_fontsize(style.resolved_font_size() * style.colorbar_scale)
    return cb


def _group_colours(style, n: int) -> list:
    """Cycle the palette while it covers the series; past that, sample the figure
    colour map evenly.

    Reusing a colour inside a parameter family is worse than a slightly less
    distinct gradient: fifteen angles of one coating are neither one curve nor a
    repeated measurement, and two curves drawn in the same blue read as either.
    """
    palette = style.colors
    if n <= len(palette):
        return [palette[i % len(palette)] for i in range(n)]
    cmap = matplotlib.colormaps[style.cmap if style.cmap != "auto" else "viridis"]
    return [cmap(x) for x in np.linspace(0.05, 0.95, n)]


class Peak(NamedTuple):
    """An interior turning point, its half-maximum width and where the width was
    measured.

    `left` and `right` travel with the number so a caller can draw the width
    rather than only print it, and a reader can see which two crossings the
    FWHM came from.
    """

    position: float
    value: float
    width: float
    is_max: bool | None
    level: float
    left: float
    right: float

    @property
    def found(self) -> bool:
        return np.isfinite(self.position)


NO_PEAK = Peak(np.nan, np.nan, np.nan, None, np.nan, np.nan, np.nan)


def _break_marks(left, right, style):
    """Slashes across the two facing spines, and those spines removed.

    A broken axis is the one place where the figure has to advertise its own
    lie: without the marks a reader carries one continuous horizontal scale
    across the seam and compares slopes that were never on the same axis.
    """
    d = 0.018
    for panel, side in ((left, 1.0), (right, 0.0)):
        for y in (0.0, 1.0):
            panel.plot(
                (side - d, side + d),
                (y - d * 1.7, y + d * 1.7),
                transform=panel.transAxes,
                color=style.spine_color,
                clip_on=False,
                linewidth=float(style.spine_width) * 1.2,
            )
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)


def _fit_overlay(ax, residual_ax, df, xname, yname, choice, palette, style, x_factor=1.0):
    """Draw a named model over the data, and its residuals in the strip below.

    The strip is not an option the caller can drop: a model that looks right is
    precisely the case where the difference has to be visible, and a residual
    column with structure in it is the only cheap signal that the shape is wrong.
    """
    d = _finite(df, [xname, yname])
    x, y = d[xname].to_numpy(), d[yname].to_numpy()
    result = fit_model(model_for(choice), x, y, x_factor=x_factor)
    grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 300)
    colour = palette[1 % len(palette)]
    ax.plot(
        grid,
        result.curve(grid),
        color=colour,
        linewidth=max(0.9, float(style.line_width) * 0.85),
        label=result.label(),
        zorder=max(int(style.series_zorder) + 1, 3),
    )
    if residual_ax is None:
        return result
    residual_ax.axhline(0.0, color="#7A94AB", lw=0.9, ls="--", zorder=1)
    residual_ax.vlines(x, 0.0, result.residual, color=colour, linewidth=0.9, zorder=2)
    residual_ax.set_ylabel("残差", fontsize=style.resolved_font_size() * 0.9)
    residual_ax.tick_params(labelsize=style.resolved_font_size() * 0.85)
    if result.notes:
        residual_ax.set_title(
            "；".join(result.notes),
            loc="left",
            fontsize=style.resolved_font_size() * 0.85,
            color="#B3261E",
        )
    return result


def _grid_of(df, e, numeric):
    """Shared (x, y, z) preparation for every surface view of a two-dimensional scan.

    Duplicate coordinates are refused rather than averaged: silently choosing a
    summary for repeated points is how a plot stops being the data.
    """
    if "x" not in e or "y" not in e or "z" not in e:  # a coordinate-free array
        z = df[e.get("columns", numeric)].to_numpy(dtype=float)
        return np.arange(z.shape[1]), np.arange(z.shape[0]), z, "Column index", "Row index", "Value"
    xn, yn, zn = e["x"], e["y"], e["z"]
    d = _finite(df, [xn, yn, zn])
    if d.duplicated([xn, yn]).any():
        raise ValueError("二维坐标存在重复观测；请先明确汇总方式。")
    table = d.pivot(index=yn, columns=xn, values=zn).sort_index().sort_index(axis=1)
    if min(table.shape) < 2:
        raise ValueError("二维图至少需要每个方向有两个坐标。")
    return table.columns.to_numpy(), table.index.to_numpy(), table.to_numpy(), xn, yn, zn


def _is_signed(z) -> bool:
    """Whether the map crosses zero, which is what decides diverging versus
    sequential colour and, with it, which line colour stays visible."""
    return bool(np.nanmin(z) < 0 < np.nanmax(z))


def _surface_colormap(z, style):
    cmap = style.cmap
    signed = _is_signed(z)
    norm = None
    if cmap == "auto":
        cmap = "RdBu_r" if signed else "viridis"
    if cmap == "RdBu_r" and signed:
        limit = float(np.nanmax(np.abs(z)))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    return cmap, norm


def _paint_surface(ax, ux, uy, z, style, masked=None):
    """The measured grid as colour, plus a layer for cells hidden by a threshold.

    A hidden cell and an unmeasured cell must not share a colour: one says
    "there is no number here", the other "there is one, and it was small". The
    first is the colour map's own bad-cell colour; the second is drawn over it as
    a separate flat layer, so both can appear on one map.
    """
    cmap, norm = _surface_colormap(z, style)
    painted = matplotlib.colormaps[cmap]
    if style.missing_fill != "none":
        painted = painted.with_extremes(bad=style.missing_fill)
    mesh = ax.pcolormesh(
        ux,
        uy,
        np.ma.masked_invalid(z),
        cmap=painted,
        norm=norm,
        shading=style.image_shading,
    )
    hidden = 0
    if masked is not None and masked.any():
        hidden = int(masked.sum())
        veil = np.ma.masked_where(~masked, np.zeros_like(np.asarray(z, dtype=float)))
        ax.pcolormesh(
            ux,
            uy,
            veil,
            cmap=matplotlib.colors.ListedColormap([style.masked_fill]),
            vmin=0.0,
            vmax=1.0,
            shading=style.image_shading,
        )
    return mesh, hidden


def _threshold_mask(z, style):
    """Cells hidden by `mask_below`, or None when no threshold is set."""
    if style.mask_below is None:
        return None
    return np.abs(z) < abs(float(style.mask_below))


def _normalised_surface(z, per):
    """Standardise each line of the grid along one coordinate.

    `per` names the coordinate held fixed: "per_y" normalises each row across x,
    which divides out whatever varies with y and leaves the row-to-row shape
    comparable. The result stops being a measured value -- two equally coloured
    cells in different rows are only equal relative to their own row.
    """
    axis = 1 if per == "per_y" else 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(z, axis=axis, keepdims=True)
        spread = np.nanstd(z, axis=axis, keepdims=True)
    flat = np.where(np.asarray(spread) > 0, spread, np.nan)
    return (z - centre) / flat


def _mean_over(z, axis):
    """Mean along one axis, with an entirely unmeasured line staying NaN instead
    of emitting a warning and a zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(z, axis=axis)


def _marginal_profiles(top, right, ux, uy, z, xn, yn, style):
    """One profile per axis: what the map averages to when read along the other.

    Only two panels, not four. The profile across the top and the one across the
    bottom are the same numbers, and printing them twice buys nothing but ink.
    """
    colour = style.colors[0 % len(style.colors)]
    weight = max(0.8, float(style.line_width) * 0.8)
    top.plot(ux, _mean_over(z, 0), color=colour, linewidth=weight)
    right.plot(_mean_over(z, 1), uy, color=colour, linewidth=weight)
    # Each caption belongs on the axis carrying the profile's own values: the top
    # panel reads vertically, the right one horizontally against the shared y.
    top.set_ylabel(f"沿 {yn} 均值", fontsize=style.resolved_font_size() * 0.82)
    right.set_xlabel(f"沿 {xn} 均值", fontsize=style.resolved_font_size() * 0.82)
    for panel in (top, right):
        panel.tick_params(labelsize=style.resolved_font_size() * 0.8)
        panel.grid(False)
    for label in top.get_xticklabels():
        label.set_visible(False)
    for label in right.get_yticklabels():
        label.set_visible(False)
    # No manual limits here. The panels share their axes with the map, and a limit
    # set on the child propagates back and clips the outermost measured cells,
    # which pcolormesh extends half a cell past the last coordinate.


def _labelled_contours(ax, ux, uy, z, style):
    """Contour lines, optionally numbered where they run.

    "auto" picks by the map's polarity: a sequential map is dark over most of its
    area, so dark lines disappear exactly where the outer levels are, while a
    diverging map is pale through its middle where the lines carry information.
    """
    colour = style.contour_line_color
    if colour == "auto":
        colour = "#20303C" if _is_signed(z) else "#FFFFFF"
    lines = ax.contour(
        ux,
        uy,
        z,
        levels=style.contour_levels,
        colors=colour,
        linewidths=max(0.4, float(style.line_width) * 0.45),
    )
    if style.contour_labels:
        ax.clabel(
            lines,
            inline=True,
            fontsize=style.resolved_font_size() * 0.82,
            fmt=lambda value: f"{value:.3g}",
        )
    return lines


def _peak_and_fwhm(wave, value) -> Peak:
    """Locate the interior extremum of a curve and its full width at half maximum.

    Returns NaNs in `position` when the curve has no interior turning point. The
    half-maximum is measured against the higher of the two curve endpoints rather
    than zero, because a reflectance dip sitting on a pedestal of 0.9 is not 90%
    deep.
    """
    wave = np.asarray(wave, dtype=float)
    value = np.asarray(value, dtype=float)
    order = np.argsort(wave)
    wave, value = wave[order], value[order]
    if wave.size < 5:
        return NO_PEAK
    i_max, i_min = int(np.nanargmax(value)), int(np.nanargmin(value))
    # Both turning points are candidates and the deeper one wins. Preferring the
    # maximum outright lets a reflectance dip be reported at the tallest noise
    # spike, because a curve that is flat everywhere except at its dip has a
    # thousand interior samples that all "exceed" their neighbours' endpoints.
    candidates = []
    if 0 < i_max < wave.size - 1 and value[i_max] > max(value[0], value[-1]):
        candidates.append((value[i_max] - max(value[0], value[-1]), i_max, True))
    if 0 < i_min < wave.size - 1 and value[i_min] < min(value[0], value[-1]):
        candidates.append((min(value[0], value[-1]) - value[i_min], i_min, False))
    if not candidates:
        return NO_PEAK
    excursion, idx, is_max = max(candidates)
    amplitude = value[idx]
    pedestal = max(value[0], value[-1]) if is_max else min(value[0], value[-1])
    half = amplitude + (pedestal - amplitude) / 2.0

    def crossing(indices):
        for position in indices:
            low, high = value[position], value[position + 1]
            if (low - half) * (high - half) <= 0 and high != low:
                frac = (half - low) / (high - low)
                return wave[position] + frac * (wave[position + 1] - wave[position])
        return np.nan

    left = crossing(range(idx - 1, -1, -1))
    right = crossing(range(idx, wave.size - 1))
    width = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
    position, height = _refine(wave, value, idx, is_max)
    return Peak(position, height, width, is_max, float(half), float(left), float(right))


def _refine(wave, value, idx, is_max):
    """Sub-sample extremum position and height from the parabola through the peak
    and its two neighbours.

    The nearest sample can be a whole grid step away from the true resonance, so
    an angle-resolved track would otherwise advance in flat stair-steps and a
    peak 8 nm from its neighbour would read as unmoved.
    """
    if idx < 1 or idx >= wave.size - 1:
        return wave[idx], value[idx]
    x = wave[idx - 1 : idx + 2]
    y = value[idx - 1 : idx + 2]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or len(set(x)) < 3:
        return wave[idx], value[idx]
    a, b, c = np.polyfit(x - x[1], y, 2)
    if a == 0 or (a > 0) != (not is_max):
        return wave[idx], value[idx]
    offset = -b / (2 * a)
    if not x[0] <= x[1] + offset <= x[2]:
        return wave[idx], value[idx]
    return float(x[1] + offset), float(c - b * b / (4 * a))


def kind_for_log(kind, axis):
    return kind == "distribution" and axis == "x"


def _peak_label(xname: str, peak: Peak) -> str:
    """Two lines at most: where the turning point is, and how wide it is.

    The width is dropped when either half-maximum crossing falls outside the
    scan, because a truncated width is not a measurement of anything.
    """
    lines = [f"{xname} = {peak.position:.6g}"]
    if np.isfinite(peak.width):
        lines.append(f"FWHM = {peak.width:.4g}")
    return "\n".join(lines)


def render(profile, rec, output=None, options=None, style=None):
    """Draw one recommendation and return the Figure.

    `style` is the presentation half of the plumbing; when omitted it is split
    out of `options`, so existing callers that pass one flat dict keep working.
    Pass a Style directly when the caller already owns one.
    """
    opts = options or {}
    if style is None:
        style, opts = Style.from_options(opts)
    elif not isinstance(style, Style):
        style = Style.from_dict(style)
    enc = dict(rec.encodings)
    enc.update(opts.get("encodings", {}))
    df = profile.data if isinstance(profile.data, pd.DataFrame) else pd.DataFrame(profile.data)
    fit_choice = str(opts.get("fit", enc.get("fit", "none"))).strip()
    if fit_choice not in FIT_CHOICES:
        # A misspelled model must not degrade into "no fit": the user asked for a
        # curve through the data and would receive a plot that looks answered.
        raise ValueError(
            f"未知拟合选项 {fit_choice!r}；可选 none、linear、" + "、".join(MODEL_BY_ID)
        )
    with matplotlib.rc_context(style.rc_params()):
        fig = Figure(
            figsize=style.size_inches(), dpi=style.preview_dpi, **style.figure_kwargs()
        )
        marginal_axes = None
        residual_ax = None
        companion_ax = None
        colour_axis = None
        if rec.id == "heatmap_marginals":
            # The colour bar gets its own column: a colorbar built from `ax`
            # shrinks only that axes, which would pull the map out of alignment
            # with the profile panels that share its axes.
            edge = max(0.05, min(0.8, float(style.marginal_height)))
            grid = fig.add_gridspec(
                2,
                3,
                width_ratios=[1.0, edge, max(0.06, style.colorbar_thickness * 3.0)],
                height_ratios=[edge, 1.0],
                hspace=0.04,
                wspace=0.04,
            )
            ax = fig.add_subplot(grid[1, 0])
            marginal_axes = (
                fig.add_subplot(grid[0, 0], sharex=ax),
                fig.add_subplot(grid[1, 1], sharey=ax),
                fig.add_subplot(grid[:, 2]),
            )
        elif rec.id == "broken_spectrum":
            if style.x_min is not None or style.x_max is not None:
                # The two panels' ranges are the split itself. Applying a manual
                # range to both would leave one panel showing an interval that
                # contains no data while still wearing the other band's label.
                raise ValueError(
                    "断轴图的横轴范围由断点位置决定，不能同时手动设定 x_min / x_max；"
                    "需要看单侧波段请改用连续轴的 x_min/x_max，或去掉断轴。"
                )
            # sharey is the whole point: the two sides must stay comparable in
            # height, or the figure would let a gap in the axis buy a fake slope.
            grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.06)
            ax = fig.add_subplot(grid[0])
            companion_ax = fig.add_subplot(grid[1], sharey=ax)
        elif rec.id in ("surface_3d", "surface_with_contour"):
            # The colour bar always takes its own column: created from a 3D parent
            # it lands on top of whichever panel happens to sit to the right.
            bars = max(0.05, float(style.colorbar_thickness) * 3.0)
            if rec.id == "surface_3d":
                grid = fig.add_gridspec(1, 2, width_ratios=[1.0, bars], wspace=0.06)
                ax = fig.add_subplot(grid[0], projection="3d")
                colour_axis = fig.add_subplot(grid[1])
            else:
                # Surface for shape, flat map for values: a perspective view
                # cannot be read against a scale, and the pair says so without
                # refusing the three-dimensional picture.
                grid = fig.add_gridspec(
                    1, 3, width_ratios=[1.3, 1.0, bars], wspace=0.02
                )
                ax = fig.add_subplot(grid[0], projection="3d")
                companion_ax = fig.add_subplot(grid[1])
                colour_axis = fig.add_subplot(grid[2])
        elif rec.id == "poincare_sphere":
            ax = fig.add_subplot(111, projection="3d")
        elif rec.id in POLAR_KINDS:
            if rec.id == "polar_and_cartesian":
                # Two readings of one sweep, side by side: the polar frame shows
                # the symmetry, the straight one shows what the rim hides, namely
                # how the trace behaves where the angle wraps.
                grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
                ax = fig.add_subplot(grid[0], projection="polar")
                companion_ax = fig.add_subplot(grid[1])
            else:
                ax = fig.add_subplot(111, projection="polar")
        elif fit_choice in _MODEL_IDS:
            # A named model always brings its residuals: a curve that looks right
            # is exactly the case where the reader needs the difference shown.
            ax, residual_ax = fig.subplots(2, 1, height_ratios=[3.2, 1.0], sharex=True)
        else:
            ax = fig.add_subplot(111)
            residual_ax = None
        _draw(
            ax, fig, df, rec.id, enc, opts, style, residual_ax, marginal_axes,
            companion_ax, colour_axis,
        )
        if residual_ax is not None:
            # One row of x labels for one shared axis: the strip underneath carries
            # them, as it does in every published residual panel.
            for label in ax.get_xticklabels():
                label.set_visible(False)
            residual_ax.set_xlabel(ax.get_xlabel() or "")
            ax.set_xlabel("")
        if opts.get("title"):
            # A renderer note (what the dB reference is, how many cells were
            # hidden) carries the meaning of the figure, so a user title goes
            # above it rather than replacing it.
            note = ax.get_title(loc="left")
            ax.set_title(
                opts["title"] if not note else f"{opts['title']}\n{note}",
                pad=14,
                loc="left",
            )
        if opts.get("xlabel"):
            ax.set_xlabel(opts["xlabel"])
        if opts.get("ylabel"):
            ax.set_ylabel(opts["ylabel"])
        for axis in ["x", "y"]:
            if opts.get(axis + "log"):
                if rec.id not in LOG_AXIS_TYPES:
                    raise ValueError("此图型不支持对数轴。")
                mapped = enc.get(axis, enc.get("value") if kind_for_log(rec.id, axis) else None)
                names = mapped if isinstance(mapped, list) else [mapped] if mapped else []
                values = (
                    np.concatenate(
                        [pd.to_numeric(df[c], errors="coerce").dropna().to_numpy() for c in names]
                    )
                    if names
                    else np.array([])
                )
                if values.size and np.any(values <= 0):
                    raise ValueError("对数轴要求全部数据大于 0。")
                getattr(ax, "set_" + axis + "scale")("log")
        # After the log scales exist: the tick formatter needs to know a log axis
        # is present so it does not print exponents as fixed decimals.
        style.configure_axes(ax, rec.id)
        for child in fig.axes:
            if child in (ax, residual_ax):
                continue
            # A twin shares the frame, so it takes the same tick and limit
            # treatment; gridding it would print a second set of lines. The
            # residual strip is skipped entirely: a manual y limit belongs to the
            # data axis, and applying it here would crop the residuals away.
            style.configure_axes(child, rec.id, grid=False)
        style.apply_margins(fig)
        fig.optiplot_encodings = enc
        style.bake_into(fig)
        if output:
            out = Path(output)
            if out.suffix.lower() not in EXPORT_FORMATS:
                raise ValueError(
                    "导出格式必须是 " + " / ".join(
                        f.upper() for f in (".png", ".svg", ".pdf", ".tiff")
                    )
                    + f"。当前是 {out.suffix!r}。"
                )
            fig.savefig(out, **style.savefig_kwargs())
        return fig


def _draw(ax, fig, df, kind, e, opts, style, residual_ax=None, marginal_axes=None,
         companion_ax=None, colour_axis=None):
    palette = style.colors
    numeric = list(df.select_dtypes(include="number").columns)
    if e.get("group") and kind in ["spectrum_lines", "scatter_fit", "errorbar", "polar"]:
        if str(opts.get("fit", e.get("fit", "none"))) in _MODEL_IDS:
            raise ValueError("分组曲线族不能整体拟合；请一次只画一条曲线，或先按参数分开。")
        group = e["group"]
        sub_enc = {k: v for k, v in e.items() if k != "group"}
        levels = list(df.dropna(subset=[group]).groupby(group, sort=False, observed=True))
        colours = _group_colours(style, len(levels))
        for i, (name, subset) in enumerate(levels):
            before_lines, before_cols = len(ax.lines), len(ax.collections)
            _draw(ax, fig, subset, kind, sub_enc, opts, style)
            color = colours[i]
            for line in ax.lines[before_lines:]:
                line.set_color(color)
                line.set_label("_nolegend_")
            for collection in ax.collections[before_cols:]:
                collection.set_color(color)
                collection.set_label("_nolegend_")
            for container in ax.containers:
                container.set_label("_nolegend_")
            ax.plot([], [], color=color, label=str(name))
        _legend(ax, style, title=group)
        return
    if kind == "spectrum_lines":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        if not 1 <= len(ys) <= 8:
            raise ValueError("曲线请选择 1–8 个响应列。")
        for i, yname in enumerate(ys):
            d = (
                df[[xname, yname]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            # Keep acquisition order and gaps: neither sort nor bridge missing measurements.
            ax.plot(
                d[xname],
                d[yname],
                color=palette[i % len(palette)],
                drawstyle="steps-mid" if style.step else "default",
                label=yname,
                **style.series_style(i),
            )
        choice = str(opts.get("fit", e.get("fit", "none")))
        if choice in _MODEL_IDS:
            if len(ys) != 1:
                raise ValueError("模型拟合一次只能作用于一条曲线，请先只选一个响应列。")
            _fit_overlay(
                ax,
                residual_ax,
                _finite(df, [xname, ys[0]]),
                xname,
                ys[0],
                choice,
                palette,
                style,
                x_factor=math.pi / 180.0
                if str(opts.get("angle_unit", "deg")) == "deg"
                else 1.0,
            )
        ax.set(xlabel=xname, ylabel=ys[0] if len(ys) == 1 else "Response")
        _legend(ax, style)
    elif kind == "broken_spectrum":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        low_end, high_end = e["gap"]
        if companion_ax is None:
            raise ValueError("断轴图需要左右两个面板。")
        panels = (ax, companion_ax)
        for i, yname in enumerate(ys):
            d = _finite(df, [xname, yname])
            colour = palette[i % len(palette)]
            for panel, keep in zip(
                panels, (d[xname] <= low_end, d[xname] >= high_end)
            ):
                part = d[keep]
                if len(part) < 2:
                    continue
                panel.plot(
                    part[xname],
                    part[yname],
                    color=colour,
                    label=yname if panel is ax else "_nolegend_",
                    drawstyle="steps-mid" if style.step else "default",
                    **style.series_style(i),
                )
        ax.set(xlabel=xname, ylabel=ys[0] if len(ys) == 1 else "Response")
        # Per-axis, not set_yticklabels([]): the two panels share a y axis, and
        # blanking the labels there blanks them on the panel that carries the scale.
        companion_ax.tick_params(axis="y", labelleft=False)
        companion_ax.set_xlabel(xname)
        # On the far side of the seam, lifted clear of the break mark.
        companion_ax.set_title(
            f"省略 {low_end:g} – {high_end:g}（该区间无测量点）",
            loc="left",
            pad=16,
            fontsize=style.resolved_font_size() * 0.85,
        )
        _break_marks(ax, companion_ax, style)
        _legend(ax, style)
        return
    elif kind == "peak_annotation":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        if not 1 <= len(ys) <= 4:
            raise ValueError("峰位标注请选择 1–4 个响应列，再多标注会互相压住。")
        unlabelled = 0
        for i, yname in enumerate(ys):
            d = (
                df[[xname, yname]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            colour = palette[i % len(palette)]
            # The drawn curve keeps its gaps; only the peak search runs on finite
            # rows, because bridging a missing stretch could place a resonance in
            # a region the instrument never sampled.
            ax.plot(
                d[xname],
                d[yname],
                color=colour,
                label=yname,
                drawstyle="steps-mid" if style.step else "default",
                **style.series_style(i),
            )
            found = _peak_and_fwhm(d[xname].to_numpy(), d[yname].to_numpy())
            if not found.found:
                unlabelled += 1
                continue
            ax.plot(
                [found.position],
                [found.value],
                marker="o" if found.is_max else "v",
                markersize=max(4.5, float(style.marker_size)),
                color=colour,
                markeredgecolor=colour,
                zorder=max(int(style.series_zorder) + 2, 4),
            )
            if np.isfinite(found.left) and np.isfinite(found.right):
                ax.plot(
                    [found.left, found.right],
                    [found.level, found.level],
                    color=colour,
                    linestyle=(0, (4, 3)),
                    linewidth=max(0.6, float(style.line_width) * 0.6),
                )
            ax.annotate(
                _peak_label(xname, found),
                xy=(found.position, found.value),
                xytext=(7, 7 if found.is_max else -24),
                textcoords="offset points",
                fontsize=style.resolved_font_size(),
                color=colour,
                # The label has to survive crossing another curve's flank, and
                # plain text over a line is unreadable in print.
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=style.axes_facecolor,
                    edgecolor="none",
                    alpha=0.8,
                ),
            )
        if unlabelled:
            ax.set_title(
                f"{unlabelled} 条曲线在扫描范围内没有内部极值，未标注峰位", pad=10, loc="left"
            )
        ax.set(xlabel=xname, ylabel=ys[0] if len(ys) == 1 else "Response")
        _legend(ax, style)
    elif kind == "stacked_curves":
        xname, group = e["x"], e["group"]
        yname = (e["y"] if isinstance(e["y"], list) else [e["y"]])[0]
        rows = (
            df[[xname, yname, group]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        levels = list(dict.fromkeys(rows[group].to_numpy()))
        if not 2 <= len(levels) <= 20:
            raise ValueError("堆叠曲线需要 2–20 个分组取值，再多就只是把图拉长。")
        curves = [rows[rows[group] == level] for level in levels]
        # One shared span, not each curve's own: the offsets then stay comparable,
        # so a taller bump really is a taller bump rather than a flatter sample.
        span = max((float(c[yname].max() - c[yname].min()) for c in curves), default=0.0)
        if not np.isfinite(span) or span <= 0:
            span = float(rows[yname].max() - rows[yname].min()) or 1.0
        colours = _group_colours(style, len(levels))
        for i, (level, curve, colour) in enumerate(zip(levels, curves, colours)):
            lift = i * float(style.stack_offset) * span
            ax.plot(
                curve[xname],
                curve[yname] + lift,
                color=colour,
                drawstyle="steps-mid" if style.step else "default",
                label=str(level),
                **style.series_style(0),
            )
            if style.stack_fill:
                ax.fill_between(
                    curve[xname],
                    lift,
                    curve[yname] + lift,
                    color=colour,
                    alpha=style.fill_alpha,
                    lw=0,
                )
        ax.set(xlabel=xname, ylabel=f"{yname}（按 {group} 逐条上移）")
        _legend(ax, style, title=group)
    elif kind in ["spectral_difference", "spectral_ratio"]:
        d, xname, a, b = _spectral_pair(df, e)
        if kind == "spectral_difference":
            xvals = d[xname].to_numpy()
            values = d[a].to_numpy() - d[b].to_numpy()
            label = f"{a} − {b}"
            reference = 0.0
        else:
            # A ratio at a vanishing denominator is an artefact of the division,
            # not a measurement, so those rows are dropped and counted.
            denominator = d[b].to_numpy()
            valid = np.abs(denominator) > 1e-12 * max(np.abs(denominator).max(), 1e-30)
            if not valid.any():
                raise ValueError("分母列全部接近零，比值无定义。")
            dropped = int((~valid).sum())
            xvals = d[xname].to_numpy()[valid]
            values = d[a].to_numpy()[valid] / denominator[valid]
            label = f"{a} / {b}"
            if dropped:
                ax.set_title(
                    f"{dropped} 行因分母接近零被屏蔽", fontsize=style.resolved_font_size() * 0.85
                )
            reference = 1.0
        ax.plot(
            xvals,
            values,
            color=palette[0 % len(palette)],
            label=label,
            **style.series_style(0),
        )
        ax.axhline(reference, color="#7A94AB", lw=0.9, ls="--", zorder=1)
        ax.set(
            xlabel=xname,
            ylabel="Difference" if kind.endswith("difference") else "Ratio",
        )
        _legend(ax, style)
    elif kind == "spectral_envelope":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        if len(ys) < 3:
            raise ValueError("包络带需要至少 3 个响应列；两列请直接画曲线。")
        d = df[[xname] + ys].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        d = d.dropna()
        if d.empty:
            raise ValueError("这些列没有共同的完整行。")
        block = d[ys].to_numpy()
        low, high = block.min(axis=1), block.max(axis=1)
        ax.fill_between(
            d[xname].to_numpy(),
            low,
            high,
            color=palette[0 % len(palette)],
            alpha=style.fill_alpha,
            label=f"min–max of {len(ys)} columns",
        )
        ax.plot(
            d[xname].to_numpy(),
            np.median(block, axis=1),
            color=palette[0 % len(palette)],
            label="median",
            **style.series_style(0),
        )
        ax.set(xlabel=xname, ylabel="Response (range, not SD)")
        _legend(ax, style)
    elif kind == "energy_axis":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        if not 1 <= len(ys) <= 8:
            raise ValueError("曲线请选择 1–8 个响应列。")
        positive = True
        for i, yname in enumerate(ys):
            d = (
                df[[xname, yname]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            if (d[xname].dropna() <= 0).any():
                positive = False
            ax.plot(
                d[xname],
                d[yname],
                color=palette[i % len(palette)],
                label=yname,
                **style.series_style(i),
            )
        ax.set(xlabel=xname, ylabel=ys[0] if len(ys) == 1 else "Response")
        if len(ys) > 1:
            _legend(ax, style)
        if not positive:
            raise ValueError("光子能量副轴要求波长全部为正。")
        # E[eV] = hc/λ with hc = 1239.841984 eV·nm. Matplotlib probes the
        # transform outside the data range, including at zero, so the converter
        # is made total rather than suppressing the resulting warning.
        def _nm_to_ev(lam):
            arr = np.asarray(lam, dtype=float)
            out = np.full(arr.shape, np.nan)
            np.divide(1239.841984, arr, out=out, where=arr != 0)
            return out.item() if out.ndim == 0 else out

        ax.secondary_xaxis("top", functions=(_nm_to_ev, _nm_to_ev)).set_xlabel(
            "photon energy / eV"
        )
    elif kind == "dual_axis":
        d, xname, a, b = _spectral_pair(df, e)
        right_column = e.get("right", b)
        left_column = a if right_column == b else b
        ax.plot(
            d[xname].to_numpy(),
            d[left_column].to_numpy(),
            color=palette[0 % len(palette)],
            label=left_column,
            **style.series_style(0),
        )
        second = ax.twinx()
        second.plot(
            d[xname].to_numpy(),
            d[right_column].to_numpy(),
            color=palette[1 % len(palette)],
            label=right_column,
            **style.series_style(1),
        )
        # Colour each axis label to its own series, otherwise a two-axis frame
        # invites reading either curve against whichever scale is nearer.
        ax.set_ylabel(left_column, color=palette[0 % len(palette)])
        second.set_ylabel(right_column, color=palette[1 % len(palette)])
        ax.set_xlabel(xname)
        ax.tick_params(axis="y", colors=palette[0 % len(palette)])
        second.tick_params(axis="y", colors=palette[1 % len(palette)])
        lines = ax.get_lines() + second.get_lines()
        _legend(ax, style, handles=lines)
    elif kind == "spectral_derivative":
        xname = e["x"]
        ys = e["y"] if isinstance(e["y"], list) else [e["y"]]
        if not 1 <= len(ys) <= 8:
            raise ValueError("导数图请选择 1–8 个响应列。")
        plotted = 0
        for i, yname in enumerate(ys):
            d = (
                df[[xname, yname]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values(xname)
            )
            if d[xname].nunique() < 3:
                continue
            slope = np.gradient(
                d[yname].to_numpy(), d[xname].to_numpy(), edge_order=2 if len(d) > 3 else 1
            )
            ax.plot(
                d[xname].to_numpy(),
                slope,
                color=palette[i % len(palette)],
                label=f"d {yname} / d {xname}",
                **style.series_style(i),
            )
            plotted += 1
        if not plotted:
            raise ValueError("没有足够的有序采样点可以求导。")
        ax.axhline(0.0, color="#7A94AB", lw=0.9, ls="--", zorder=1)
        ax.set(xlabel=xname, ylabel=f"d / d{xname}")
        _legend(ax, style)
    elif kind == "peak_evolution":
        param, wave, value = e["param"], e["wave"], e["value"]
        rows = df[[param, wave, value]].dropna()
        if rows.empty:
            raise ValueError("没有可用于提取峰位的完整三元组。")
        positions, widths, skipped = [], [], 0
        for level, group in rows.groupby(param, sort=True):
            found = _peak_and_fwhm(group[wave].to_numpy(), group[value].to_numpy())
            if not found.found:
                skipped += 1
                continue
            positions.append((float(level), float(found.position)))
            widths.append((float(level), float(found.width) if np.isfinite(found.width) else np.nan))
        if len(positions) < 2:
            raise ValueError(
                f"只有 {len(positions)} 个 {param} 取值存在内部极值，无法构成演化曲线；"
                "若峰位于扫描边界，请扩大扫描范围而不是画出来。"
            )
        ax.plot(
            [p[0] for p in positions],
            [p[1] for p in positions],
            color=palette[0 % len(palette)],
            label=f"peak {wave}",
            **style.series_style(0),
        )
        second = ax.twinx() if widths else None
        finite = [(w[0], w[1]) for w in widths if np.isfinite(w[1])]
        if second is not None and finite:
            second.plot(
                [p[0] for p in finite],
                [p[1] for p in finite],
                color=palette[1 % len(palette)],
                label="FWHM",
                **style.series_style(1),
            )
            second.set_ylabel("FWHM", color=palette[1 % len(palette)])
            second.tick_params(axis="y", colors=palette[1 % len(palette)])
        ax.set(xlabel=param, ylabel=f"peak {wave}")
        if skipped:
            ax.set_title(
                f"{skipped} 个 {param} 取值无内部极值，已跳过",
                fontsize=style.resolved_font_size() * 0.85,
            )
        handles = ax.get_lines() + (second.get_lines() if second is not None else [])
        _legend(ax, style, handles=handles)
    elif kind in ["scatter_fit", "density"]:
        xname, yname = e["x"], e["y"]
        choice = str(opts.get("fit", e.get("fit", "none")))
        d = _finite(df, [xname, yname])
        x, y = d[xname].to_numpy(), d[yname].to_numpy()
        if kind == "density":
            m = ax.hexbin(x, y, mincnt=1,
                             cmap=style.cmap if style.cmap != "auto" else "viridis",
                             bins="log", gridsize=style.hexbin_gridsize)
            _colorbar(fig, m, ax, "Count (log color scale)", style)
        else:
            ax.scatter(
                x,
                y,
                s=style.scatter_size,
                alpha=style.scatter_alpha,
                color=palette[0 % len(palette)],
                edgecolors=(palette[0 % len(palette)] if style.scatter_edge else "none"),
                linewidths=style.marker_edge_width if style.scatter_edge else 0,
                rasterized=len(d) > style.rasterize_above,
                zorder=style.series_zorder,
            )
            if choice == "linear":
                if len(d) < 3 or np.unique(x).size < 2:
                    raise ValueError("线性拟合至少需要 3 个点及两个不同的 X 值。")
                coef = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                ss = np.sum((y - y.mean()) ** 2)
                r2 = 1 - np.sum((y - np.polyval(coef, x)) ** 2) / ss if ss else float("nan")
                ax.plot(
                    xx,
                    np.polyval(coef, xx),
                    color=palette[1 % len(palette)],
                    label=f"OLS: y={coef[0]:.3g}x{coef[1]:+.3g}; R²={r2:.3f}",
                )
                _legend(ax, style)
            elif choice in _MODEL_IDS:
                _fit_overlay(ax, residual_ax, d, xname, yname, choice, palette, style)
                _legend(ax, style)
        ax.set(xlabel=xname, ylabel=yname)
    elif kind == "errorbar":
        xname, yname = e["x"], e["y"]
        yerr = e.get("yerr", e.get("error"))
        if yerr:
            d = _finite(df, [xname, yname, yerr])
            err = d[yerr].to_numpy()
            if np.any(err < 0):
                raise ValueError("误差值不能为负数。")
            ax.errorbar(
                d[xname],
                d[yname],
                yerr=err,
                fmt=style.marker if style.marker != "none" else "o",
                ms=style.marker_size,
                color=palette[0 % len(palette)],
                label=yerr,
                **style.errorbar_kwargs(),
            )
        else:
            d = _finite(df, [xname, yname])
            g = d.groupby(xname)[yname]
            stats = g.agg(["mean", "std", "count"])
            valid = stats["count"] >= 2
            if not valid.any():
                ax.scatter(
                    stats.index,
                    stats["mean"],
                    marker="x",
                    color=palette[0 % len(palette)],
                    label="n=1; uncertainty unavailable",
                )
                ax.set(xlabel=xname, ylabel=yname)
                return
            s = stats.loc[valid]
            err = (
                s["std"] / np.sqrt(s["count"])
                if opts.get("error_type", e.get("error_type", "sd")) == "sem"
                else s["std"]
            )
            label = (
                "Mean ± SEM"
                if opts.get("error_type", e.get("error_type", "sd")) == "sem"
                else "Mean ± SD"
            )
            ax.errorbar(
                s.index,
                s["mean"],
                yerr=err,
                fmt="o-" if style.show_lines else "o",
                ms=style.marker_size,
                lw=style.line_width,
                color=palette[0 % len(palette)],
                label=label,
            )
            if (~valid).any():
                ax.scatter(
                    stats.index[~valid],
                    stats.loc[~valid, "mean"],
                    marker="x",
                    color=palette[1 % len(palette)],
                    label="n=1; uncertainty unavailable",
                )
        ax.set(xlabel=xname, ylabel=yname)
        _legend(ax, style)
    elif kind in PAINTED_GRID_KINDS:
        ux, uy, z, xn, yn, zn = _grid_of(df, e, numeric)
        # The threshold is judged against the measured value, before any
        # normalisation: a user setting 0.05 means five per cent of what they
        # measured, not five per cent of some row's standard deviation.
        mask = _threshold_mask(z, style)
        caption = zn
        if kind == "heatmap_normalized":
            per = str(opts.get("normalize", e.get("normalize", "per_y")))
            if per not in NORMALIZATIONS:
                raise ValueError(
                    f"normalize 需是 {'/'.join(NORMALIZATIONS)} 之一，当前 {per!r}"
                )
            fixed, other = (yn, xn) if per == "per_y" else (xn, yn)
            z = _normalised_surface(z, per)
            caption = f"每个 {fixed} 处沿 {other} 的 z-score"
        if kind == "contour":
            cmap, norm = _surface_colormap(z, style)
            m = ax.contourf(
                ux, uy, z, levels=style.contour_levels, cmap=cmap, norm=norm
            )
            if style.contour_labels:
                _labelled_contours(ax, ux, uy, z, style)
            hidden = 0
        else:
            m, hidden = _paint_surface(ax, ux, uy, z, style, masked=mask)
            if kind == "heatmap_contours":
                _labelled_contours(ax, ux, uy, z, style)
        if hidden:
            # Say what the flat colour is, next to the colour, rather than
            # trusting a reader to guess that a grey square was measured.
            caption += f"\n已屏蔽 {hidden} 格（|{zn}| < {float(style.mask_below):g}）"
        if marginal_axes is None:
            _colorbar(fig, m, ax, caption, style)
        else:
            top, right, colour_axis = marginal_axes
            fig.colorbar(m, cax=colour_axis, label=caption)
            _marginal_profiles(top, right, ux, uy, z, xn, yn, style)
        ax.set(xlabel=xn, ylabel=yn)
        ax.grid(False)
        return
    elif kind == "mueller_matrix":
        cells = e["cells"]
        row = df[cells].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if row.empty:
            raise ValueError("Mueller 矩阵没有可用的数值。")
        matrix = row.to_numpy()[0].reshape(4, 4)
        limit = float(np.nanmax(np.abs(matrix))) or 1.0
        m = ax.imshow(
            matrix,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        base = ("S0", "S1", "S2", "S3")
        ax.set_xticks(range(4), base)
        ax.set_yticks(range(4), base)
        ax.set(xlabel="输出 Stokes 分量", ylabel="入射 Stokes 分量", title="Mueller 矩阵")
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=style.resolved_font_size() * 0.85,
                    color="white" if abs(value) > 0.6 * limit else "#17324D",
                )
        _colorbar(fig, m, ax, "元素值（与 M00 同量纲）", style)
        # A statement about the numbers, not a verdict on the sample: whether a
        # deviation from symmetric means the material is non-reciprocal is the
        # user's call, and the units of the two blocks decide it.
        off = matrix - matrix.T
        if np.isfinite(off).all():
            i, j = np.unravel_index(int(np.nanargmax(np.abs(off))), off.shape)
            if i != j and abs(off[i, j]) > 1e-9 * limit:
                ax.set_title(
                    f"Mueller 矩阵；与转置矩阵最大偏差 {abs(off[i, j]):.3g}"
                    f"（M{i}{j}/M{j}{i}），是否违反互易需结合样品类型判断",
                    loc="left",
                    pad=10,
                    fontsize=style.resolved_font_size() * 0.85,
                )
        ax.grid(False)
        return
    elif kind in ("surface_3d", "surface_with_contour"):
        ux, uy, z, xn, yn, zn = _grid_of(df, e, numeric)
        cmap, norm = _surface_colormap(z, style)
        mesh = np.ma.masked_invalid(z)
        levels = ax.plot_surface(
            *np.meshgrid(ux, uy),
            mesh,
            cmap=cmap,
            norm=norm,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            alpha=0.97,
        )
        ax.view_init(elev=float(style.view_elevation), azim=float(style.view_azimuth))
        ax.set(xlabel=xn, ylabel=yn)
        # The colour bar sits directly off the 3D box's back-right edge, which is
        # exactly where a z label would land; naming the quantity twice there only
        # produces two overprinted strings.
        if colour_axis is None:
            ax.set_zlabel(zn)
        # z's units differ from x and y, so there is no honest "true" box ratio:
        # the frame is normalised to a cube and only the declared exaggeration is
        # applied, instead of a span ratio that silently encodes unit choice.
        ax.set_box_aspect((1.0, 1.0, float(style.z_exaggeration)))
        note = ""
        if float(style.z_exaggeration) != 1.0:
            note = (
                f"纵向拉伸 ×{style.z_exaggeration:g}（只改形状，刻度仍是实测量；"
                "此角度下斜率与倾角不可读）"
            )
        if note:
            ax.set_title(note, loc="left", pad=8, fontsize=style.resolved_font_size() * 0.85)
        if colour_axis is not None:
            _colorbar(fig, levels, colour_axis, zn, style, dedicated=True)
        if companion_ax is not None:
            companion_ax.contourf(ux, uy, z, levels=style.contour_levels, cmap=cmap, norm=norm)
            if style.contour_labels:
                _labelled_contours(companion_ax, ux, uy, z, style)
            companion_ax.set(xlabel=xn, ylabel=yn)
            companion_ax.set_title(
                "俯视等高线（读数值看这里）",
                loc="left",
                pad=8,
                fontsize=style.resolved_font_size() * 0.85,
            )
            companion_ax.grid(False)
        return
    elif kind == "poincare_sphere":
        s0, s1, s2, s3 = (
            pd.to_numeric(df[c], errors="coerce") for c in e["stokes"]
        )
        live = np.isfinite(s0) & np.isfinite(s1) & np.isfinite(s2) & np.isfinite(s3) & (s0 > 0)
        if int(live.sum()) < 2:
            raise ValueError("Poincaré 球需要至少两个 S0 > 0 的完整 Stokes 测量。")
        u = (s1[live] / s0[live]).to_numpy()
        v = (s2[live] / s0[live]).to_numpy()
        w = (s3[live] / s0[live]).to_numpy()
        radius = np.sqrt(u**2 + v**2 + w**2)
        # The physical bound is on the radius, not the sum of components: using
        # the latter would flag a legitimate state at (0.6, 0.6, 0) as impossible.
        unphysical = int((radius > 1.0 + 1e-6).sum())
        # Points are drawn where they were measured; the sphere is only a frame.
        # Forcing them onto the surface would hide a partial polarisation, which
        # is the very thing the radius carries.
        circle = np.linspace(0.0, 2.0 * math.pi, 100)
        # Three great circles read as a frame without hiding what lies behind
        # them; a filled surface turns the sphere into a blob at most angles.
        for first, second in ((0, 1), (0, 2), (1, 2)):
            ring = np.zeros((circle.size, 3))
            ring[:, first] = np.cos(circle)
            ring[:, second] = np.sin(circle)
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#B9CBD8", lw=0.7, zorder=1)
        ax.plot(
            u, v, w,
            color=palette[0 % len(palette)],
            lw=style.line_width,
            zorder=3,
            label=f"轨迹（按 {e['label']} 顺序）",
        )
        ax.scatter(u, v, w, s=style.scatter_size * 0.5, color=palette[1 % len(palette)], zorder=4)
        ax.view_init(elev=float(style.view_elevation), azim=float(style.view_azimuth))
        ax.set(xlabel="S1/S0", ylabel="S2/S0", zlabel="S3/S0")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        note = f"|S|/S0 最大 {float(np.nanmax(radius)):.3g}"
        if unphysical:
            note += f"；{unphysical} 点落在球外（Stokes 参数不自洽），仍按实测画出"
        ax.set_title(note, loc="left", pad=6, fontsize=style.resolved_font_size() * 0.85)
        _legend(ax, style)
        return
    elif kind in POLAR_KINDS:
        theta, r = e["theta"], e["r"]
        d = _finite(df, [theta, r])
        unit = opts.get("angle_unit", e.get("angle_unit", "deg"))
        angles = np.deg2rad(d[theta].to_numpy()) if unit in ["deg", "degrees"] else d[theta].to_numpy()
        radius = d[r].to_numpy()
        if (radius < 0).any():
            raise ValueError("极坐标半径存在负值，请使用角度-响应折线图以保留符号。")
        ix = np.argsort(angles)
        angles, radius = angles[ix], radius[ix]
        # Kept in the unit the file used: converting to radians and back to draw
        # the straight-axis panel would give it values that are not the ones read.
        degrees = d[theta].to_numpy()[ix]
        if kind == "polar_db":
            live = radius > 0
            if not live.any():
                raise ValueError("分贝方向图要求至少一个正的半径值：0 与负数取对数无定义。")
            factor = float(style.db_factor)
            reference = float(np.nanmax(radius))
            # A zero reading is not -inf dB, it is below the plotted floor; those
            # points are dropped and counted rather than drawn at the rim.
            dropped = int((~live).sum())
            plotted = factor * np.log10(np.where(live, radius, np.nan) / reference)
            # The depth follows the data, capped by db_floor: a fixed -30 dB rim
            # would squash a pattern that only spans 5 dB into the outer ring and
            # throw away the whole radius.
            shallowest = float(np.nanmin(plotted))
            # Round outward, not inward: rounding in would clip the tail of the
            # pattern and report it as points that did not fit.
            depth = min(float(style.db_floor), max(5.0, 5.0 * math.ceil(-shallowest / 5.0)))
            clipped = int((plotted < -depth).sum())
            inside = live & (plotted >= -depth)
            ax.plot(
                angles[inside],
                plotted[inside],
                color=palette[0 % len(palette)],
                label=r,
                **style.series_style(0),
            )
            ax.set_rlim(-depth, 0.0)
            ax.set_rlabel_position(90)
            note = (
                f"0 dB = {reference:.4g} {r}，按 {factor:g}·log10；"
                f"径向画到 −{depth:g} dB"
            )
            unseen = dropped + clipped
            if unseen:
                note += f"；{unseen} 点为 0/负值或低于该范围，未画"
            ax.set_title(note, loc="left", pad=10, fontsize=style.resolved_font_size() * 0.85)
        else:
            ax.plot(angles, radius, color=palette[0 % len(palette)], label=r, **style.series_style(0))
        _legend(ax, style, loc="upper right", bbox_to_anchor=(1.35, 1.13))
        if kind == "polar_and_cartesian" and companion_ax is not None:
            companion_ax.plot(
                degrees, radius, color=palette[0 % len(palette)], label=r, **style.series_style(0)
            )
            companion_ax.set(xlabel=f"{theta} ({unit})", ylabel=r)
            _legend(companion_ax, style)
        return
    elif kind in ["distribution", "box"]:
        value = e.get("value", numeric[0] if numeric else None)
        if kind == "box" and e.get("group"):
            group = e["group"]
            groups = list(df.dropna(subset=[group]).groupby(group, sort=False, observed=True))[:20]
            arrays = []
            labels = []
            for name, g in groups:
                a = (
                    pd.to_numeric(g[value], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy()
                )
                if len(a):
                    arrays.append(a)
                    labels.append(str(name))
            if not arrays:
                raise ValueError("没有可绘制的分组数值。")
            bp = ax.boxplot(arrays, patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor="#CAE5F1", edgecolor=palette[0 % len(palette)])
            rng = np.random.default_rng(7)
            for i, a in enumerate(arrays):
                ax.scatter(
                    i + 1 + rng.uniform(-0.13, 0.13, len(a)),
                    a,
                    s=style.scatter_size * 0.5625,
                    alpha=style.scatter_alpha,
                    color=palette[i % len(palette)],
                )
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20 if len(labels) > 5 else 0)
            ax.set(xlabel=group, ylabel=value)
        else:
            a = _finite(df, [value])[value]
            ax.hist(
                a,
                bins=min(80, max(5, int(np.sqrt(len(a))))),
                color=palette[0 % len(palette)],
                alpha=style.series_alpha if style.series_alpha < 1 else 0.85,
                edgecolor="white",
                zorder=style.series_zorder,
            )
            ax.set(xlabel=value, ylabel="Count")
    elif kind == "correlation":
        cols = e.get("columns", numeric)[:12]
        corr = df[cols].corr()
        m = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
        ax.set_yticks(range(len(cols)), cols)
        _colorbar(fig, m, ax, "Pearson r; pairwise complete", style)
        if len(cols) <= 7:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(corr.iloc[i, j]) > 0.65 else "#17324D",
                    )
        ax.grid(False)
        return
    elif kind == "flow":
        source, target = e.get("source", "source"), e.get("target", "target")
        edges = df[[source, target]].dropna().astype(str).drop_duplicates()
        nodes = list(dict.fromkeys(edges.to_numpy().ravel()))
        if len(nodes) > 18:
            raise ValueError("流程预览支持最多 18 个节点；请拆分复杂流程。")
        indegree = {v: 0 for v in nodes}
        outgoing = {v: [] for v in nodes}
        for a, b in edges.itertuples(index=False, name=None):
            indegree[b] += 1
            outgoing[a].append(b)
        starts = [v for v in nodes if indegree[v] == 0]
        chain = (
            len(starts) == 1
            and len(edges) == len(nodes) - 1
            and all(indegree[v] <= 1 and len(outgoing[v]) <= 1 for v in nodes)
        )
        positions = {}
        if chain:
            ordered = []
            current = starts[0]
            while current not in ordered:
                ordered.append(current)
                if not outgoing[current]:
                    break
                current = outgoing[current][0]
            chain = len(ordered) == len(nodes)
        if chain:
            nrows = math.ceil(len(nodes) / 3)
            for i, v in enumerate(ordered):
                row, col = divmod(i, 3)
                col = col if row % 2 == 0 else 2 - col
                positions[v] = (
                    0.17 + 0.33 * col,
                    0.5 if nrows == 1 else 0.8 - row * 0.6 / (nrows - 1),
                )
        else:
            positions = {
                v: (
                    0.5 + 0.36 * math.cos(math.pi / 2 - 2 * math.pi * i / len(nodes)),
                    0.5 + 0.36 * math.sin(math.pi / 2 - 2 * math.pi * i / len(nodes)),
                )
                for i, v in enumerate(nodes)
            }
        boxes = {}
        import textwrap

        for v, (x, y) in positions.items():
            boxes[v] = FancyBboxPatch(
                (x - 0.105, y - 0.05),
                0.21,
                0.10,
                boxstyle="round,pad=.012",
                facecolor="#E8F4FA",
                edgecolor=palette[0 % len(palette)],
                zorder=2,
            )
            ax.add_patch(boxes[v])
            ax.text(
                x,
                y,
                "\n".join(textwrap.wrap(v, 18)),
                ha="center",
                va="center",
                fontsize=9,
                zorder=3,
            )
        for a, b in edges.itertuples(index=False, name=None):
            if a == b:
                x, y = positions[a]
                ax.annotate(
                    "",
                    xy=(x + 0.04, y + 0.063),
                    xytext=(x - 0.04, y + 0.063),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "connectionstyle": "arc3,rad=-2",
                        "color": "#7A94AB",
                    },
                )
            else:
                ax.add_patch(
                    FancyArrowPatch(
                        positions[a],
                        positions[b],
                        patchA=boxes[a],
                        patchB=boxes[b],
                        arrowstyle="-|>",
                        mutation_scale=14,
                        connectionstyle="arc3,rad=0" if chain else "arc3,rad=.08",
                        shrinkA=4,
                        shrinkB=4,
                        color="#7A94AB",
                        zorder=1,
                    )
                )
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        return
    elif kind == "table":
        shown = df[e.get("columns", list(df.columns))[:8]].head(14).copy()
        for c in shown:
            shown[c] = shown[c].map(
                lambda v: (
                    ""
                    if pd.isna(v)
                    else f"{v:.4g}" if isinstance(v, (int, float, np.number)) else str(v)[:32]
                )
            )
        tab = ax.table(
            cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center"
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.6)
        for (r, c), cell in tab.get_celld().items():
            cell.set_linewidth(0.4)
            cell.set_edgecolor("#D2DFE9")
            if r == 0:
                cell.set_facecolor("#17324D")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF5F9")
        ax.axis("off")
        if len(df) > 14 or len(df.columns) > 8:
            ax.set_title(
                "Preview: first 14 rows / 8 columns; full data in export package", fontsize=9
            )
        return
    else:
        raise ValueError(f"未实现图型：{kind}")
    ax.grid(False)
