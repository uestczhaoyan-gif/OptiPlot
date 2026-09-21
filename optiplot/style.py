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
import json
from pathlib import Path

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


# Colour-blind-safe categorical palettes. Okabe-Ito is the default and the one
# the renderers were built around; the Paul Tol sets give more slots.
PALETTES = {
    "okabe": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"],
    "tol_bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    "tol_vibrant": ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB"],
    "tol_muted": [
        "#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
        "#882255", "#44AA99", "#999933", "#AA4499",
    ],
}
LEGEND_LOCS = (
    "best", "none", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center",
)
# `legend.loc` is a validated rcParam and accepts only the named positions -
# "auto" works as an ax.legend(loc=) argument but not here. Matplotlib's own
# automatic placement is called "best", so that is what we default to, and
# "none" is intercepted by _legend before it can reach the rcParams dict.
LEGEND_PARAM_DEFAULT = "best"
LEGEND_TITLE_SIZES = ("small", "medium", "large", "x-large")
# Cycled through when vary_line_style is on, so overlapping series stay
# separable in a greyscale print.
LINE_STYLES = ("-", "--", "-.", ":")
MARKER_FILLS = ("full", "none", "left", "right", "bottom", "top")
# pcolormesh takes `shading`, not imshow's `interpolation`; the two value sets
# do not overlap, so the field is named for the call that actually uses it.
# "flat" is a legal pcolormesh value but needs coordinates one larger than the
# data in every dimension, which a measured grid never is - so it is not offered.
SHADINGS = ("auto", "nearest", "gouraud")
# Matplotlib marker codes worth exposing for optics plots; the full set is on
# the docs page but these are the ones that survive small print sizes.
MARKERS = (
    "none", "o", "s", "^", "v", "<", ">", "D", "d", "p", "h", "+", "x", ".", ",",
    "*", "1", "2", "3", "4", "|", "_",
)
# Named cross-group combinations. A preset is worth having only when it moves
# several groups together coherently - canvas plus base size plus line weight
# plus legend - otherwise it is just `size` under another name.
STYLE_PRESETS = {
    "default": {},
    "journal": {
        "size": "double", "font_size": 9.0, "line_width": 1.2, "spines": "all",
        "tick_length": 3.0, "tick_minor": True, "legend_frame": False,
        "marker_size": 4.0, "scatter_size": 12.0, "dpi": 600, "pad_inches": 0.02,
        "tight_bbox": True,
    },
    "slide": {
        "size": "slide", "font_size": 17.0, "line_width": 3.0, "grid": "major",
        "grid_alpha": 0.35, "legend_columns": 2, "marker_size": 8.0,
        "scatter_size": 40.0, "tick_length": 6.0, "dpi": 150,
    },
    "poster": {
        "size": "slide", "font_size": 24.0, "line_width": 4.0, "spines": "all",
        "legend_frame": True, "legend_frame_alpha": 1.0, "marker_size": 11.0,
        "scatter_size": 70.0, "tick_length": 8.0, "label_pad": 14.0, "dpi": 300,
    },
}


def _validate_colour(value: str, name: str) -> None:
    """Face colours accept anything Matplotlib understands - named colours,
    grey levels, "none" - so the strict #RRGGBB rule for palette entries does
    not apply here."""
    from matplotlib.colors import is_color_like

    if str(value).lower() not in ("none", "auto", "inherit") and not is_color_like(value):
        raise ValueError(f"{name} 不是合法颜色：{value!r}")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Strict #RRGGBB. A 3-digit shorthand or a bare hex run is rejected with a
    clear message rather than being guessed at."""
    text = value.strip()
    if not text.startswith("#") or len(text) != 7:
        raise ValueError(f"颜色需写成 #RRGGBB 七位形式，当前 {value!r}")
    digits = text[1:]
    if any(ch not in "0123456789abcdefABCDEF" for ch in digits):
        raise ValueError(f"颜色含非十六进制字符：{value!r}")
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))


LAYOUTS = ("constrained", "tight", "none")
EXPORT_FORMATS = (".png", ".svg", ".pdf", ".tif", ".tiff")
MM_PER_INCH = 25.4


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
    width_mm: float | None = None
    height_mm: float | None = None
    aspect_lock: float | None = None
    layout: str = "constrained"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    margin_left: float | None = None
    margin_right: float | None = None
    margin_top: float | None = None
    margin_bottom: float | None = None
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
    # E legend
    legend: str = "best"
    legend_columns: int = 1
    legend_frame: bool = False
    legend_frame_alpha: float = 0.0
    legend_facecolor: str = "none"
    legend_edgecolor: str = "#CCCCCC"
    legend_handle_length: float = 1.6
    legend_handle_textpad: float = 0.5
    legend_label_spacing: float = 0.5
    legend_column_spacing: float = 1.0
    legend_border_padding: float = 0.2
    legend_title_size: str = "large"
    # F colour
    palette: str = "okabe"
    custom_colors: str = ""
    cmap: str = "auto"
    # D data series
    line_width: float = 1.65
    vary_line_style: bool = True
    marker: str = "none"
    marker_size: float = 4.5
    marker_fill: str = "full"
    marker_edge_width: float = 0.8
    marker_every: int | None = None
    step: bool = False
    show_lines: bool = True
    show_points: bool = False
    series_alpha: float = 1.0
    series_zorder: int = 2
    error_cap_size: float = 3.0
    error_line_width: float | None = None
    scatter_size: float = 16.0
    scatter_alpha: float = 0.65
    scatter_edge: bool = False
    rasterize_above: int = 5000
    contour_levels: int = 14
    hexbin_gridsize: int = 45
    image_shading: str = "nearest"
    colorbar_thickness: float = 0.046
    colorbar_pad: float = 0.03
    fill_difference: bool = False
    fill_alpha: float = 0.15
    # H export
    dpi: int = 300
    preview_dpi: int = 110
    svg_text_as_paths: bool = False
    pdf_embed_type42: bool = True
    transparent: bool = False
    tight_bbox: bool = False
    pad_inches: float = 0.1

    @property
    def colors(self) -> list[str]:
        if self.palette == "custom":
            return self.parsed_custom_colors()
        return list(PALETTES[self.palette])

    def parsed_custom_colors(self) -> list[str]:
        values = [part.strip() for part in self.custom_colors.split(",") if part.strip()]
        if not values:
            raise ValueError('palette="custom" 需要 custom_colors 给出至少一个 #RRGGBB')
        for value in values:
            _hex_to_rgb(value)  # validates, raises on a malformed entry
        return values

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
        if self.legend not in LEGEND_LOCS:
            raise ValueError(f"legend 需是 {'/'.join(LEGEND_LOCS)} 之一，当前 {self.legend!r}")
        if not 1 <= int(self.legend_columns) <= 6:
            raise ValueError(f"legend_columns 需在 1–6 之间，当前 {self.legend_columns}")
        if self.legend_title_size not in LEGEND_TITLE_SIZES:
            raise ValueError(f"legend_title_size 需是 {'/'.join(LEGEND_TITLE_SIZES)} 之一")
        for name in (
            "legend_handle_length", "legend_handle_textpad", "legend_label_spacing",
            "legend_column_spacing", "legend_border_padding",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 5.0:
                raise ValueError(f"{name} 需在 0–5 之间，当前 {value}")
        if not 0.0 <= float(self.legend_frame_alpha) <= 1.0:
            raise ValueError(f"legend_frame_alpha 需在 0–1 之间，当前 {self.legend_frame_alpha}")
        if self.palette not in PALETTES and self.palette != "custom":
            raise ValueError(
                f"palette 需是 {'/'.join(PALETTES)} 或 custom，当前 {self.palette!r}"
            )
        if self.palette == "custom":
            self.parsed_custom_colors()
        if self.custom_colors and self.palette != "custom":
            raise ValueError("custom_colors 只有在 palette=\"custom\" 时才会被使用")
        if self.legend_frame:
            _hex_to_rgb(self.legend_edgecolor)
        if not 0.1 <= float(self.line_width) <= 8.0:
            raise ValueError(f"line_width 需在 0.1–8 pt 之间，当前 {self.line_width}")
        if self.marker not in MARKERS:
            raise ValueError(f"marker 需是 {'/'.join(MARKERS)} 之一，当前 {self.marker!r}")
        if self.marker_fill not in MARKER_FILLS:
            raise ValueError(f"marker_fill 需是 {'/'.join(MARKER_FILLS)} 之一")
        for name in ("marker_size", "scatter_size"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 200.0:
                raise ValueError(f"{name} 需在 0–200 之间，当前 {value}")
        if not 0.0 <= float(self.marker_edge_width) <= 5.0:
            raise ValueError(f"marker_edge_width 需在 0–5 之间，当前 {self.marker_edge_width}")
        if self.marker_every is not None and int(self.marker_every) < 1:
            raise ValueError(f"marker_every 需 ≥1（每几个点画一个标记），当前 {self.marker_every}")
        for name in ("series_alpha", "scatter_alpha", "fill_alpha"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} 需在 0–1 之间，当前 {getattr(self, name)}")
        if not 0.0 <= float(self.error_cap_size) <= 20.0:
            raise ValueError(f"error_cap_size 需在 0–20 之间，当前 {self.error_cap_size}")
        if self.error_line_width is not None and not 0.1 <= float(self.error_line_width) <= 8.0:
            raise ValueError(f"error_line_width 需在 0.1–8 之间，当前 {self.error_line_width}")
        if not 1 <= int(self.contour_levels) <= 100:
            raise ValueError(f"contour_levels 需在 1–100 之间，当前 {self.contour_levels}")
        if not 3 <= int(self.hexbin_gridsize) <= 200:
            raise ValueError(f"hexbin_gridsize 需在 3–200 之间，当前 {self.hexbin_gridsize}")
        if self.image_shading not in SHADINGS:
            raise ValueError(f"image_shading 需是 {'/'.join(SHADINGS)} 之一")
        if not 0.005 <= float(self.colorbar_thickness) <= 0.3:
            raise ValueError(f"colorbar_thickness 需在 0.005–0.3 之间，当前 {self.colorbar_thickness}")
        if not 0.0 <= float(self.colorbar_pad) <= 0.5:
            raise ValueError(f"colorbar_pad 需在 0–0.5 之间，当前 {self.colorbar_pad}")
        if int(self.rasterize_above) < 1:
            raise ValueError(f"rasterize_above 需 ≥1，当前 {self.rasterize_above}")
        if not -100 <= int(self.series_zorder) <= 100:
            raise ValueError(f"series_zorder 需在 -100–100 之间，当前 {self.series_zorder}")
        if not self.show_lines and not self.show_points:
            raise ValueError("show_lines 与 show_points 不能同时关闭，否则系列不可见")
        if not 40 <= int(self.dpi) <= 3000:
            raise ValueError(f"dpi 需在 40–3000 之间，当前 {self.dpi}")
        if not 40 <= int(self.preview_dpi) <= 600:
            raise ValueError(f"preview_dpi 需在 40–600 之间，当前 {self.preview_dpi}")
        if int(self.preview_dpi) > int(self.dpi):
            raise ValueError(
                f"preview_dpi ({self.preview_dpi}) 不应高于导出 dpi ({self.dpi})，"
                "预览不会比成品更清晰"
            )
        if not 0.0 <= float(self.pad_inches) <= 2.0:
            raise ValueError(f"pad_inches 需在 0–2 之间，当前 {self.pad_inches}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout 需是 {'/'.join(LAYOUTS)} 之一，当前 {self.layout!r}")
        for name in ("width_mm", "height_mm"):
            value = getattr(self, name)
            if value is not None and not 10.0 <= float(value) <= 2000.0:
                raise ValueError(f"{name} 需在 10–2000 mm 之间，当前 {value}")
        if self.aspect_lock is not None and not 0.1 <= float(self.aspect_lock) <= 10.0:
            raise ValueError(f"aspect_lock 需在 0.1–10 之间（宽/高），当前 {self.aspect_lock}")
        if self.aspect_lock is not None and self.width_mm is not None and self.height_mm is not None:
            raise ValueError(
                "aspect_lock 与同时给定的 width_mm + height_mm 冲突："
                "两个维度已经锁定了比例"
            )
        for name in ("figure_facecolor", "axes_facecolor"):
            _validate_colour(getattr(self, name), name)
        if self.has_custom_margins():
            # constrained and tight layouts compute the subplot box themselves and
            # discard anything set here, so honouring it would be a silent no-op.
            if self.layout != "none":
                raise ValueError(
                    "自定义留白需要 layout=\"none\"：constrained 与 tight 会自行计算"
                    "子图位置并丢弃 margin_* 的设定"
                )
            for side in ("left", "right", "top", "bottom"):
                value = getattr(self, f"margin_{side}")
                if value is not None and not 0.0 <= float(value) <= 0.9:
                    raise ValueError(f"margin_{side} 需在 0–0.9 之间（占画布比例），当前 {value}")
            if self.tight_bbox:
                raise ValueError(
                    "tight_bbox 与自定义留白冲突：紧裁会重新贴合内容，抹掉 margin_*"
                )
        return self

    def size_inches(self) -> tuple[float, float]:
        """Millimetres win over the named preset, because that is the unit a
        journal's author guide is written in."""
        if self.width_mm is None and self.height_mm is None:
            return SIZES[self.size]
        base_w, base_h = SIZES[self.size]
        if self.width_mm is not None and self.height_mm is not None:
            return float(self.width_mm) / MM_PER_INCH, float(self.height_mm) / MM_PER_INCH
        if self.width_mm is not None:
            width = float(self.width_mm) / MM_PER_INCH
            return width, width / (self.aspect_lock or base_w / base_h)
        height = float(self.height_mm) / MM_PER_INCH
        return height * (self.aspect_lock or base_w / base_h), height

    def has_custom_margins(self) -> bool:
        return any(
            getattr(self, f"margin_{side}") is not None
            for side in ("left", "right", "top", "bottom")
        )

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
            "axes.facecolor": self.axes_facecolor,
            "figure.facecolor": self.figure_facecolor,
            "legend.loc": LEGEND_PARAM_DEFAULT if self.legend == "none" else self.legend,
            # There is no legend.ncols rcParam; column count is passed to
            # ax.legend() per call. See Style.legend_kwargs().
            "legend.frameon": self.legend_frame,
            "legend.framealpha": self.legend_frame_alpha,
            "legend.facecolor": self.legend_facecolor,
            "legend.edgecolor": self.legend_edgecolor,
            "legend.handlelength": self.legend_handle_length,
            "legend.handletextpad": self.legend_handle_textpad,
            "legend.labelspacing": self.legend_label_spacing,
            "legend.columnspacing": self.legend_column_spacing,
            "legend.borderpad": self.legend_border_padding,
            "legend.title_fontsize": self.legend_title_size,
            "savefig.facecolor": "none" if self.transparent else self.figure_facecolor,
            # Vector text stays editable in Illustrator unless the caller asks
            # for paths; both matter for journal submission.
            "svg.fonttype": "path" if self.svg_text_as_paths else "none",
            "pdf.fonttype": 42 if self.pdf_embed_type42 else 3,
        }

    def figure_kwargs(self) -> dict:
        kwargs = {"facecolor": self.figure_facecolor}
        # "tight" is applied at savefig, not at construction; only constrained is
        # a Figure(layout=...) value.
        if self.layout == "constrained":
            kwargs["layout"] = "constrained"
        return kwargs

    def savefig_kwargs(self, dpi: int | None = None) -> dict:
        """`dpi` is only an override for callers that have no Style of their
        own; a figure rendered from a Style exports at that Style's dpi."""
        return {
            "dpi": int(dpi or self.dpi),
            "facecolor": "none" if self.transparent else "white",
            "bbox_inches": "tight" if self.tight_bbox else None,
            "pad_inches": self.pad_inches,
        }

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

    def legend_kwargs(self) -> dict:
        """Artist-level legend arguments - the ones with no rcParam."""
        return {"ncols": int(self.legend_columns)}

    def series_style(self, index: int = 0) -> dict:
        """Per-series line and marker arguments for ax.plot().

        Colour is deliberately not here - the renderer takes that from
        `palette`, so cycling stays in one place.
        """
        kwargs = {
            "linewidth": self.line_width if self.show_lines else 0.0,
            "zorder": self.series_zorder,
            "alpha": self.series_alpha,
        }
        if self.vary_line_style:
            kwargs["linestyle"] = LINE_STYLES[index % len(LINE_STYLES)]
        if self.marker != "none" or self.show_points:
            marker = self.marker if self.marker != "none" else "o"
            kwargs.update(
                marker=marker,
                markersize=self.marker_size,
                # fillstyle, not markerfacecolor: "left"/"right"/"top"/"bottom"
                # are fill styles and are not valid colours.
                fillstyle=self.marker_fill,
                markeredgecolor="auto",
                markeredgewidth=self.marker_edge_width,
            )
            if self.marker_every is not None:
                kwargs["markevery"] = int(self.marker_every)
        if not self.show_lines:
            kwargs["linestyle"] = "None"
        return kwargs

    def errorbar_kwargs(self) -> dict:
        """Appearance only. Which axis carries the error stays the renderer's
        call, since x- versus y-error is a data question, not a style one."""
        return {
            "capsize": self.error_cap_size,
            "elinewidth": self.error_line_width or self.line_width,
            "zorder": self.series_zorder,
        }

    def apply_margins(self, fig) -> None:
        """Explicit subplot box, honoured only under layout="none".

        Kept separate from configure_axes because the canvas box belongs to the
        figure and applies to every figure type, whereas axis styling is
        restricted to the numeric-frame types.
        """
        if self.layout != "none":
            return
        left = self.margin_left if self.margin_left is not None else 0.125
        right = (
            1.0 - self.margin_right
            if self.margin_right is not None
            else 1.0 - 0.1
        )
        top = 1.0 - self.margin_top if self.margin_top is not None else 0.88
        bottom = self.margin_bottom if self.margin_bottom is not None else 0.11
        for ax in fig.axes:
            ax.set_position([left, bottom, max(right - left, 0.05), max(top - bottom, 0.05)])

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

    @classmethod
    def from_preset(cls, name: str = "default", **overrides) -> "Style":
        """A named preset, with any field overridden on top.

        Unknown names are rejected rather than falling back to default, so a
        typo in a shared preset file is visible instead of quietly producing a
        differently-styled figure.
        """
        if name not in STYLE_PRESETS:
            near = difflib.get_close_matches(name, sorted(STYLE_PRESETS), n=3)
            raise ValueError(
                f"未知样式预设 {name!r}。可用：{', '.join(sorted(STYLE_PRESETS))}。"
                f"{'最接近的是：' + ', '.join(near) if near else ''}"
            )
        values = dict(STYLE_PRESETS[name])
        values.update(overrides)
        return cls.from_dict(values)

    def save_preset(self, path) -> Path:
        """Write the whole style out so a group can share one house look."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        target.write_text(body + "\n", encoding="utf-8")
        return target

    @classmethod
    def load_preset(cls, path) -> "Style":
        source = Path(path)
        if not source.exists():
            raise ValueError(f"找不到样式预设文件：{source}")
        try:
            values = json.loads(source.read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            raise ValueError(f"样式预设不是合法 JSON：{source}（{exc}）") from None
        if not isinstance(values, dict):
            raise ValueError(f"样式预设应是一个 JSON 对象，{source} 里是 {type(values).__name__}")
        return cls.from_dict(values)

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
