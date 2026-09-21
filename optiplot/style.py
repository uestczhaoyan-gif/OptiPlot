"""Style parameters and the plumbing that routes them to Matplotlib.

A style option is only useful once it reaches the right place, and Matplotlib
has four different places: rcParams (inherited by everything drawn inside the
context), the Figure constructor, per-artist keyword arguments read at draw
time, and savefig. This module is the single owner of that routing, so a
renderer branch never has to invent its own `options.get(...)` convention.

Add a parameter by: declaring it here, returning it from the right one of
rc_params/figure_kwargs/savefig_kwargs (or reading it in _draw), and adding a
case to tests/test_style.py.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict, replace
import difflib

from matplotlib import font_manager

# Curated starting points. Any installed font family name is also accepted.
FAMILY_PRESETS = {
    "sans": ["Arial", "Helvetica", "DejaVu Sans"],
    "serif": ["Times New Roman", "Cambria", "DejaVu Serif"],
    "mono": ["Consolas", "Courier New", "DejaVu Sans Mono"],
    "system": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"],
}
CJK_PRESETS = {
    "none": [],
    "yahei": ["Microsoft YaHei"],
    "heiti": ["SimHei"],
    "songti": ["SimSun"],
    "kaiti": ["KaiTi"],
    "songfang": ["FangSong"],
    "noto": ["Noto Sans SC"],
}
WEIGHTS = ("normal", "bold", "light")
TICK_DIRECTIONS = ("in", "out", "inout")
GRID_STATES = ("off", "major", "minor", "both")
# Types whose axes are a real numeric frame and can be re-styled as a line plot
# can. Everything else draws an image, a diagram or a polar frame, where grid
# lines, tick locators and a numeric xlim are either meaningless or wrong - and
# for a heatmap, cropping the axis would silently crop measured data.
AXIS_STYLED = frozenset(
    {"spectrum_lines", "scatter_fit", "density", "errorbar", "distribution", "box"}
)
# Journal column widths. `preview` is the on-screen default.
SIZES = {"preview": (7.2, 4.7), "single": (3.5, 2.65), "double": (7.2, 4.6), "slide": (10, 5.625)}
# A single-column figure has to carry smaller text at the same physical size,
# so the base font tracks the canvas preset unless the user pins it explicitly.
PRESET_FONT = {"single": 9.0, "preview": 11.0, "double": 11.0, "slide": 11.0}
# Options that are not presentation: they change what is drawn or what text
# appears, never how it looks. Kept separate so a typo in a style name cannot
# be quietly absorbed as one of these.
DATA_KEYS = frozenset(
    {
        "encodings",
        "fit",
        "angle_unit",
        "error_type",
        "title",
        "xlabel",
        "ylabel",
        "xlog",
        "ylog",
        "dpi",
        "cmap",
    }
)
# Multipliers, not absolute points: raising the base size should move every
# text element with it, otherwise a "smaller font" preset needs six numbers.
SCALE_DEFAULTS = {
    "title_scale": 1.10,
    "label_scale": 1.05,
    "tick_scale": 0.92,
    "legend_scale": 0.92,
    "colorbar_scale": 0.92,
}


def installed_families() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def _resolve(candidates: list[str], installed: set[str]) -> list[str]:
    """Keep only families that exist, always ending in a font that does."""
    hits = [c for c in candidates if c in installed]
    return hits or ["DejaVu Sans"]


@dataclass(frozen=True)
class Style:
    """Figure-agnostic presentation parameters.

    Groups: A typography (implemented), and B canvas / C axes / E legend /
    F colour / H export / I presets, added by later milestones. Data-semantics
    options (fit, angle_unit, error_type) and text content (title, labels) are
    deliberately not here - they are not presentation.
    """

    # A typography
    family: str = "sans"
    cjk: str = "yahei"
    font_size: float | None = None
    title_scale: float = SCALE_DEFAULTS["title_scale"]
    label_scale: float = SCALE_DEFAULTS["label_scale"]
    tick_scale: float = SCALE_DEFAULTS["tick_scale"]
    legend_scale: float = SCALE_DEFAULTS["legend_scale"]
    colorbar_scale: float = SCALE_DEFAULTS["colorbar_scale"]
    title_weight: str = "normal"
    label_weight: str = "normal"
    label_pad: float = 6.0
    tick_pad: float = 4.0
    # B canvas (started; margins and aspect follow)
    size: str = "preview"
    # C axes
    spines: str = "left+bottom"
    spine_width: float = 0.7
    spine_color: str = "#333333"
    tick_direction: str = "out"
    tick_length: float = 3.5
    tick_minor_length: float = 2.0
    tick_width: float = 0.7
    tick_minor: bool = False
    tick_rotation: float = 0.0
    tick_max: int | None = None
    tick_precision: int | None = None
    sci_power: int | None = None
    use_offset: bool = True
    axis_margin: float = 0.05
    grid: str = "off"
    grid_style: str = ":"
    grid_width: float = 0.6
    grid_color: str = "#9E9E9E"
    grid_alpha: float = 0.7
    grid_under_data: bool = True
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    x_reverse: bool = False
    y_reverse: bool = False
    y_zero_centered: bool = False

    # ------------------------------------------------------------------
    def validate(self) -> "Style":
        installed = installed_families()
        known = set(FAMILY_PRESETS) | installed
        if self.family not in known:
            near = difflib.get_close_matches(self.family, sorted(known), n=3)
            raise ValueError(
                f"字体族 {self.family!r} 未安装。可用：{', '.join(sorted(FAMILY_PRESETS))}"
                f"，或本机已安装字体名。{'最接近的：' + ', '.join(near) if near else ''}"
            )
        if self.cjk not in CJK_PRESETS and self.cjk not in installed:
            raise ValueError(f"中文字体 {self.cjk!r} 未安装。可选：{', '.join(sorted(CJK_PRESETS))}")
        if self.font_size is not None and not 4.0 <= float(self.font_size) <= 30.0:
            raise ValueError(f"基准字号需在 4–30 pt 之间，当前 {self.font_size}")
        for name in SCALE_DEFAULTS:
            value = getattr(self, name)
            if not 0.4 <= value <= 3.0:
                raise ValueError(f"{name} 需在 0.4–3.0 倍之间，当前 {value}")
        for name in ("title_weight", "label_weight"):
            if getattr(self, name) not in WEIGHTS:
                raise ValueError(f"{name} 需是 {'/'.join(WEIGHTS)} 之一")
        for name in ("label_pad", "tick_pad"):
            value = getattr(self, name)
            if not 0.0 <= value <= 40.0:
                raise ValueError(f"{name} 需在 0–40 pt 之间，当前 {value}")
        if self.size not in SIZES:
            raise ValueError(f"尺寸预设需是 {'/'.join(SIZES)} 之一，当前 {self.size!r}")
        if self.tick_direction not in TICK_DIRECTIONS:
            raise ValueError(f"tick_direction 需是 {'/'.join(TICK_DIRECTIONS)} 之一")
        if self.grid not in GRID_STATES:
            raise ValueError(f"grid 需是 {'/'.join(GRID_STATES)} 之一，当前 {self.grid!r}")
        for name in ("spine_width", "tick_width", "grid_width"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 5.0:
                raise ValueError(f"{name} 需在 0–5 pt 之间，当前 {value}")
        for name in ("tick_length", "tick_minor_length"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 25.0:
                raise ValueError(f"{name} 需在 0–25 pt 之间，当前 {value}")
        if not -90.0 <= self.tick_rotation <= 90.0:
            raise ValueError(f"tick_rotation 需在 -90 到 90 度之间，当前 {self.tick_rotation}")
        if self.tick_max is not None and not 2 <= int(self.tick_max) <= 40:
            raise ValueError(f"tick_max 需在 2–40 之间，当前 {self.tick_max}")
        if self.tick_precision is not None and not 0 <= int(self.tick_precision) <= 10:
            raise ValueError(f"tick_precision 需在 0–10 位之间，当前 {self.tick_precision}")
        if self.sci_power is not None and not -8 <= int(self.sci_power) <= 8:
            raise ValueError(f"sci_power 需在 -8 到 8 之间，当前 {self.sci_power}")
        if not 0.0 <= self.axis_margin <= 0.5:
            raise ValueError(f"axis_margin 需在 0–0.5 之间，当前 {self.axis_margin}")
        if not 0.0 <= float(self.grid_alpha) <= 1.0:
            raise ValueError(f"grid_alpha 需在 0–1 之间，当前 {self.grid_alpha}")
        for axis, low, high in (("x", self.x_min, self.x_max), ("y", self.y_min, self.y_max)):
            if low is not None and high is not None and float(low) >= float(high):
                raise ValueError(f"{axis} 轴下限必须小于上限（{low} ≥ {high}）")
        if self.y_zero_centered and self.y_min is not None and self.y_max is not None:
            raise ValueError("y_zero_centered 与手动 y 范围冲突，请只设一个")
        self.spines_visible()
        return self

    def size_inches(self) -> tuple[float, float]:
        return SIZES[self.size]

    def resolved_font_size(self) -> float:
        """Explicit pt wins; otherwise the canvas preset picks it."""
        return float(self.font_size) if self.font_size else PRESET_FONT[self.size]

    def font_stack(self) -> list[str]:
        """CJK font first, then the Latin family.

        Matplotlib does not fall back per glyph across `font.sans-serif`, and a
        CJK font such as Microsoft YaHei also carries Latin glyphs. So the CJK
        font has to lead or Chinese renders as tofu - and while it leads,
        `family` cannot affect Latin text either.

        That is why ``cjk="none"`` exists: an English-only figure, which is what
        most journals want, drops the CJK font and gets full control over
        `family`. Choosing a Chinese font and expecting a Latin family to show
        through it is not possible here, so the two are made explicitly
        exclusive rather than both half-working.
        """
        installed = installed_families()
        cjk = CJK_PRESETS.get(self.cjk, [self.cjk])
        latin = FAMILY_PRESETS.get(self.family, [self.family])
        ordered = list(dict.fromkeys([*cjk, *latin, "DejaVu Sans"]))
        return _resolve(ordered, installed)

    def spines_visible(self) -> frozenset[str]:
        """"all", "none", or a "+"-joined set such as "left+bottom+top"."""
        spec = self.spines.strip().lower()
        if spec in ("all", "both"):
            return frozenset({"left", "right", "top", "bottom"})
        if spec in ("none", ""):
            return frozenset()
        wanted = {part.strip() for part in spec.split("+")}
        unknown = wanted - {"left", "right", "top", "bottom"}
        if unknown:
            raise ValueError(f"未知脊线名：{', '.join(sorted(unknown))}")
        return frozenset(wanted)

    def rc_params(self) -> dict:
        """Everything Matplotlib inherits while the figure is drawn."""
        base = self.resolved_font_size()
        spine_set = self.spines_visible()
        direction = {"inout": "inout"}.get(self.tick_direction, self.tick_direction)
        return {
            "font.family": "sans-serif",
            "font.sans-serif": self.font_stack(),
            "axes.unicode_minus": False,
            "font.size": base,
            "axes.titlesize": base * self.title_scale,
            "axes.labelsize": base * self.label_scale,
            "xtick.labelsize": base * self.tick_scale,
            "ytick.labelsize": base * self.tick_scale,
            "legend.fontsize": base * self.legend_scale,
            "axes.titleweight": self.title_weight,
            "axes.labelweight": self.label_weight,
            "axes.labelpad": self.label_pad,
            "xtick.major.pad": self.tick_pad,
            "ytick.major.pad": self.tick_pad,
            "axes.spines.top": "top" in spine_set,
            "axes.spines.right": "right" in spine_set,
            "axes.spines.left": "left" in spine_set,
            "axes.spines.bottom": "bottom" in spine_set,
            "axes.linewidth": self.spine_width,
            "axes.edgecolor": self.spine_color,
            "xtick.direction": direction,
            "ytick.direction": direction,
            "xtick.major.size": self.tick_length,
            "ytick.major.size": self.tick_length,
            "xtick.minor.size": self.tick_minor_length,
            "ytick.minor.size": self.tick_minor_length,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "xtick.minor.visible": self.tick_minor,
            "ytick.minor.visible": self.tick_minor,
            "axes.axisbelow": self.grid_under_data,
            "savefig.facecolor": "white",
            # Vector text stays editable in Illustrator unless the caller asks
            # for paths; both matter for journal submission.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }

    def figure_kwargs(self) -> dict:
        return {"facecolor": "white", "layout": "constrained"}

    def savefig_kwargs(self, dpi: int = 300) -> dict:
        return {"dpi": int(dpi), "facecolor": "white"}

    # ------------------------------------------------------------------
    def bake_into(self, fig) -> "Style":
        """Pin the resolved font stack onto every Text artist in the figure.

        `font.family` is re-read from rcParams at *every* draw, but render()
        returns a figure that the caller draws later - by then the rc context
        has closed and the text silently falls back to the global font. Font
        sizes do not have this problem because they are resolved when the
        artist is created. So a real renderer is attached, one draw is forced to
        materialise the tick labels, and only then is the family written onto
        each artist. After that the figure carries its own typography wherever
        it is drawn - GUI canvas, savefig, or a reloaded reproducible bundle.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.text import Text

        FigureCanvasAgg(fig)
        fig.canvas.draw()
        stack = self.font_stack()
        for text in fig.findobj(Text):
            if text.get_text():
                text.set_fontfamily(stack)
        return self

    def configure_axes(self, ax, kind: str) -> None:
        """Per-Axes settings that rcParams cannot express: locators, formatters,
        limits and grid state.

        Only AXIS_STYLED types are touched. A heatmap's axis range is measured
        data, so cropping it would hide samples while looking like a cosmetic
        choice, and grid lines over a pcolormesh are noise.
        """
        if kind not in AXIS_STYLED:
            return
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        ax.tick_params(axis="both", which="both", labelrotation=self.tick_rotation)

        if self.grid != "off":
            major = self.grid in ("major", "both")
            minor = self.grid in ("minor", "both")
            common = {
                "linestyle": self.grid_style,
                "linewidth": self.grid_width,
                "color": self.grid_color,
                "alpha": self.grid_alpha,
            }
            ax.grid(major, which="major", **common)
            if minor:
                ax.grid(True, which="minor", **common)

        if self.tick_max is not None:
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(nbins=int(self.tick_max), prune="upper"))
        if self.tick_precision is not None:
            # A fixed-decimal formatter would print 1e3 as "1000.000" on a log
            # axis, where the tick values are exponents - leave those alone.
            for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
                if scale == "log":
                    continue
                axis.set_major_formatter(
                    FuncFormatter(lambda v, _: f"{v:.{int(self.tick_precision)}f}")
                )
        if self.sci_power is not None:
            power = int(self.sci_power)
            ax.ticklabel_format(style="sci", scilimits=(power, power), useOffset=self.use_offset)
        elif not self.use_offset:
            ax.ticklabel_format(useOffset=False)

        ax.margins(self.axis_margin)
        if self.x_min is not None or self.x_max is not None:
            ax.set_xlim(left=self.x_min, right=self.x_max)
        if self.y_min is not None or self.y_max is not None:
            ax.set_ylim(bottom=self.y_min, top=self.y_max)
        elif self.y_zero_centered:
            top, bottom = ax.get_ylim()
            reach = max(abs(top), abs(bottom))
            ax.set_ylim(-reach, reach)
        if self.x_reverse:
            ax.invert_xaxis()
        if self.y_reverse:
            ax.invert_yaxis()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Style":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            near = difflib.get_close_matches(unknown[0], sorted(known), n=3)
            raise ValueError(
                f"未知样式参数：{', '.join(unknown)}。"
                f"可用：{', '.join(sorted(known))}。"
                f"{'最接近的是：' + ', '.join(near) if near else ''}"
            )
        return cls(**values).validate()

    def replace(self, **values) -> "Style":
        return replace(self, **values).validate()

    @classmethod
    def from_options(cls, options: dict) -> tuple["Style", dict]:
        """Split a caller's options into (style, data-and-content).

        Strict on purpose: a misspelled style name would otherwise land in the
        data half and be silently ignored, which is the worst possible failure
        for a knob the user believes they turned.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(options) - known - DATA_KEYS)
        if unknown:
            near = difflib.get_close_matches(unknown[0], sorted(known), n=3)
            raise ValueError(
                f"未知参数：{', '.join(unknown)}。"
                f"样式参数：{', '.join(sorted(known))}。"
                f"数据参数：{', '.join(sorted(DATA_KEYS))}。"
                f"{'最接近的样式参数是：' + ', '.join(near) if near else ''}"
            )
        return cls.from_dict({k: v for k, v in options.items() if k in known}), {
            k: v for k, v in options.items() if k in DATA_KEYS
        }


STYLE_FIELDS = tuple(f.name for f in fields(Style))
