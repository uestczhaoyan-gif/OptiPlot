"""Deterministic publication renderers. No fitting, smoothing or interpolation by default."""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:  # normal import, as part of the optiplot package
    from .style import EXPORT_FORMATS, Style
except ImportError:  # standalone copy inside an exported reproducible bundle
    from style import EXPORT_FORMATS, Style

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
# The authoritative list of drawable figure types. A new branch in _draw is not
# a supported figure type until it appears here; catalog/styles.json may only
# reference these ids.
FIGURE_TYPES = (
    "spectrum_lines",
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
    "contour",
    "matrix_heatmap",
    "polar",
    "distribution",
    "box",
    "correlation",
    "flow",
    "table",
)


# A difference crosses zero by construction, so a log axis there is meaningless;
# the others are ratios or positive-valued spectra.
LOG_AXIS_TYPES = (
    "spectrum_lines",
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


def _colorbar(fig, m, ax, label, style):
    """Artist-level styling: one of the two routing points rcParams cannot reach."""
    cb = fig.colorbar(
        m, ax=ax, label=label, fraction=style.colorbar_thickness, pad=style.colorbar_pad
    )
    cb.outline.set_linewidth(0.7)
    cb.ax.yaxis.label.set_fontsize(style.resolved_font_size() * style.colorbar_scale)
    return cb


def _peak_and_fwhm(wave, value):
    """Locate the interior extremum of a curve and its full width at half maximum.

    Returns (position, width, is_maximum) with NaNs when the curve has no
    interior turning point. The half-maximum is measured against the higher of
    the two curve endpoints rather than zero, because a reflectance dip sitting
    on a pedestal of 0.9 is not 90% deep.
    """
    wave = np.asarray(wave, dtype=float)
    value = np.asarray(value, dtype=float)
    order = np.argsort(wave)
    wave, value = wave[order], value[order]
    if wave.size < 5:
        return np.nan, np.nan, None
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
        return np.nan, np.nan, None
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
    return _refine(wave, value, idx, is_max), width, is_max


def _refine(wave, value, idx, is_max):
    """Sub-sample extremum position from the parabola through the peak and its
    two neighbours.

    The nearest sample can be a whole grid step away from the true resonance, so
    an angle-resolved track would otherwise advance in flat stair-steps and a
    peak 8 nm from its neighbour would read as unmoved.
    """
    if idx < 1 or idx >= wave.size - 1:
        return wave[idx]
    x = wave[idx - 1 : idx + 2]
    y = value[idx - 1 : idx + 2]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or len(set(x)) < 3:
        return wave[idx]
    a, b, _ = np.polyfit(x - x[1], y, 2)
    if a == 0 or (a > 0) != (not is_max):
        return wave[idx]
    offset = -b / (2 * a)
    return float(x[1] + offset) if x[0] <= x[1] + offset <= x[2] else wave[idx]


def kind_for_log(kind, axis):
    return kind == "distribution" and axis == "x"


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
    with matplotlib.rc_context(style.rc_params()):
        fig = Figure(
            figsize=style.size_inches(), dpi=style.preview_dpi, **style.figure_kwargs()
        )
        ax = fig.add_subplot(111, projection="polar" if rec.id == "polar" else None)
        _draw(ax, fig, df, rec.id, enc, opts, style)
        if opts.get("title"):
            ax.set_title(opts["title"], pad=14, loc="left")
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


def _draw(ax, fig, df, kind, e, opts, style):
    palette = style.colors
    numeric = list(df.select_dtypes(include="number").columns)
    if e.get("group") and kind in ["spectrum_lines", "scatter_fit", "errorbar", "polar"]:
        group = e["group"]
        sub_enc = {k: v for k, v in e.items() if k != "group"}
        for i, (name, subset) in enumerate(
            df.dropna(subset=[group]).groupby(group, sort=False, observed=True)
        ):
            before_lines, before_cols = len(ax.lines), len(ax.collections)
            _draw(ax, fig, subset, kind, sub_enc, opts, style)
            color = palette[i % len(palette)]
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
        ax.set(xlabel=xname, ylabel=ys[0] if len(ys) == 1 else "Response")
        if len(ys) > 1:
            _legend(ax, style)
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
            peak, width, _ = _peak_and_fwhm(group[wave].to_numpy(), group[value].to_numpy())
            if not np.isfinite(peak):
                skipped += 1
                continue
            positions.append((float(level), float(peak)))
            widths.append((float(level), float(width) if np.isfinite(width) else np.nan))
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
            if opts.get("fit", e.get("fit", "none")) == "linear":
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
    elif kind in ["heatmap", "contour", "matrix_heatmap"]:
        if kind == "matrix_heatmap":
            z = df[e.get("columns", numeric)].to_numpy(dtype=float)
            ux = np.arange(z.shape[1])
            uy = np.arange(z.shape[0])
            xn, yn, zn = "Column index", "Row index", "Value"
        else:
            xn, yn, zn = e["x"], e["y"], e["z"]
            d = _finite(df, [xn, yn, zn])
            if d.duplicated([xn, yn]).any():
                raise ValueError("二维坐标存在重复观测；请先明确汇总方式。")
            table = d.pivot(index=yn, columns=xn, values=zn).sort_index().sort_index(axis=1)
            ux = table.columns.to_numpy()
            uy = table.index.to_numpy()
            z = table.to_numpy()
            if min(z.shape) < 2:
                raise ValueError("二维图至少需要每个方向有两个坐标。")
        cmap = style.cmap
        signed = np.nanmin(z) < 0 < np.nanmax(z)
        norm = None
        if cmap == "auto":
            cmap = "RdBu_r" if signed else "viridis"
        if cmap == "RdBu_r" and signed:
            limit = float(np.nanmax(np.abs(z)))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        if kind == "contour":
            m = ax.contourf(ux, uy, z, levels=style.contour_levels, cmap=cmap, norm=norm)
        else:
            m = ax.pcolormesh(
                ux, uy, np.ma.masked_invalid(z), cmap=cmap, norm=norm,
                shading=style.image_shading,
            )
        _colorbar(fig, m, ax, zn, style)
        ax.set(xlabel=xn, ylabel=yn)
        ax.grid(False)
        return
    elif kind == "polar":
        theta, r = e["theta"], e["r"]
        d = _finite(df, [theta, r])
        unit = opts.get("angle_unit", e.get("angle_unit", "deg"))
        angles = d[theta].to_numpy()
        angles = np.deg2rad(angles) if unit in ["deg", "degrees"] else angles
        if (d[r] < 0).any():
            raise ValueError("极坐标半径存在负值，请使用角度-响应折线图以保留符号。")
        ix = np.argsort(angles)
        ax.plot(
            angles[ix],
            d[r].to_numpy()[ix],
            color=palette[0 % len(palette)],
            label=r,
            **style.series_style(0),
        )
        _legend(ax, style, loc="upper right", bbox_to_anchor=(1.35, 1.13))
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
