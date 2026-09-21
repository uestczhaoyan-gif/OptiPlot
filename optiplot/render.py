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


def _finite(df, names):
    cols = list(dict.fromkeys(names))
    out = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("所选变量没有完整且有限的数值配对。")
    return out


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
    """Artist-level styling: the one routing point rcParams cannot reach."""
    cb = fig.colorbar(m, ax=ax, label=label, fraction=style.colorbar_thickness,
                            pad=style.colorbar_pad)
    cb.outline.set_linewidth(0.7)
    cb.ax.yaxis.label.set_fontsize(style.resolved_font_size() * style.colorbar_scale)
    return cb


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
                if rec.id not in [
                    "spectrum_lines",
                    "scatter_fit",
                    "density",
                    "errorbar",
                    "distribution",
                ]:
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
