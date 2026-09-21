"""Style plumbing: routing, validation, and proof that each knob bites.

A parameter that can be set but changes nothing is worse than no parameter,
because the user believes they turned it. So most of this file reads back the
resolved artist properties from a real figure rather than trusting the dict.
"""

from pathlib import Path
import io
import json
import warnings
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from optiplot import analyze_dataframe, analyze_file, recommend
from optiplot.render import render
from optiplot.style import Style, SIZES, PRESET_FONT, DATA_KEYS, MARKERS, MARKER_FILLS, SHADINGS, installed_families

ROOT = Path(__file__).resolve().parents[1]


def profile():
    rng = np.random.default_rng(1)
    x = np.linspace(1200, 1800, 80)
    return analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": x,
                "device_A": np.exp(-((x - 1500) ** 2) / 2e4),
                "device_B": np.exp(-((x - 1550) ** 2) / 2e4),
            }
        )
    )


def drawn(options=None, style=None):
    p = profile()
    fig = render(p, recommend(p)[0], options=options, style=style)
    return fig, fig.axes[0]


# ── routing ─────────────────────────────────────────────────────────


def test_defaults_reproduce_the_pre_plumbing_renderer():
    """The old renderer hard-coded these; moving them into Style must not
    silently restyle every existing figure."""
    rc = Style().rc_params()
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert rc["axes.linewidth"] == 0.7
    assert rc["svg.fonttype"] == "none" and rc["pdf.fonttype"] == 42
    assert rc["axes.unicode_minus"] is False


def test_base_font_still_follows_the_canvas_preset():
    assert Style(size="single").resolved_font_size() == PRESET_FONT["single"]
    assert Style(size="double").resolved_font_size() == PRESET_FONT["double"]
    # an explicit size wins over the preset
    assert Style(size="single", font_size=14).resolved_font_size() == 14.0


def test_size_preset_reaches_the_figure():
    for name, inches in SIZES.items():
        _, ax = drawn({"size": name})
        got = ax.figure.get_size_inches()
        assert np.allclose(got, inches), f"{name} -> {got}"


def test_stack_always_ends_with_an_installed_font():
    installed = installed_families()
    for style in (Style(), Style(family="mono"), Style(family="serif", cjk="none")):
        stack = style.font_stack()
        assert stack, f"{style.family}/{style.cjk} resolved to nothing"
        assert all(name in installed for name in stack), f"uninstalled font in {stack}"


def test_cjk_none_hands_the_latin_family_full_control():
    """The two knobs are mutually exclusive by design, so this is the case where
    `family` actually reaches Latin glyphs."""
    assert Style(family="serif", cjk="none").font_stack()[0] == "Times New Roman"
    assert Style(family="mono", cjk="none").font_stack()[0] == "Consolas"


def test_a_cjk_font_leads_and_therefore_owns_latin_text_too():
    """Documents the Matplotlib constraint rather than fighting it: with a CJK
    font selected, `family` cannot show through, because Matplotlib does not
    fall back per glyph and YaHei carries Latin shapes."""
    stack = Style(family="serif", cjk="yahei").font_stack()
    assert stack[0] == "Microsoft YaHei"
    assert "Times New Roman" in stack  # still the fallback for uncovered glyphs


def test_chinese_text_actually_renders_with_the_default_style():
    """Tofu is the failure mode here, and it is silent. Matplotlib warns rather
    than raises, so assert the warning never fires on a Chinese label."""
    ax = drawn({"title": "光束强度二维分布", "xlabel": "横向位置 / μm"})[1]
    fig = ax.figure
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        FigureCanvasAgg(fig)
        fig.canvas.draw()


def test_family_knob_changes_latin_glyphs_when_cjk_is_off():
    """Prove the two figures really differ in pixels, not just in metadata."""
    import hashlib

    def digest(style):
        p = profile()
        fig = render(p, recommend(p)[0], style=style)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        fig.clear()
        return hashlib.sha256(buf.getvalue()).hexdigest()

    assert digest(Style(family="serif", cjk="none")) != digest(
        Style(family="mono", cjk="none")
    )


# ── each typography knob actually bites ─────────────────────────────


def test_font_size_scales_every_text_element():
    _, small = drawn({"font_size": 8})
    _, large = drawn({"font_size": 16})
    for getter in (
        lambda a: a.xaxis.label.get_fontsize(),
        lambda a: a.get_xticklabels()[0].get_fontsize(),
    ):
        assert getter(large) > getter(small) * 1.5


def test_scales_are_independent():
    """Raising only the tick scale must not drag the axis label with it."""
    _, base = drawn({"font_size": 10})
    _, ticks = drawn({"font_size": 10, "tick_scale": 2.0})
    assert ticks.get_xticklabels()[0].get_fontsize() > base.get_xticklabels()[0].get_fontsize()
    assert ticks.xaxis.label.get_fontsize() == pytest.approx(base.xaxis.label.get_fontsize())


def test_title_and_label_weights_reach_the_artists():
    _, ax = drawn({"title": "T", "title_weight": "bold", "label_weight": "bold"})
    assert ax.title.get_weight() == "bold"
    assert ax.xaxis.label.get_weight() == "bold"


def test_label_pad_widens_the_gap_between_axis_and_label():
    """Constrained layout keeps the label pinned near the figure edge and moves
    the axes instead, so absolute label position proves nothing. What has to
    grow is the gap from the axes bottom edge to the top of the label."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    gaps = []
    for pad in (0.0, 30.0):
        ax = drawn({"label_pad": pad})[1]
        fig = ax.figure
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        gaps.append(
            ax.get_window_extent(r).y0 - ax.xaxis.label.get_window_extent(r).y1
        )
    # 30 pt at the figure's 110 dpi is ~45.8 px
    assert gaps[1] - gaps[0] == pytest.approx(30 * 110 / 72, abs=6)


def test_default_stack_still_carries_a_cjk_font():
    """The pre-plumbing renderer always put an installed CJK font first. Losing
    it would silently turn every Chinese axis label into tofu boxes."""
    from optiplot.style import CJK_PRESETS

    installed_cjk = {
        name for names in CJK_PRESETS.values() for name in names if name in installed_families()
    }
    if not installed_cjk:
        pytest.skip("no CJK font installed (CI runners may not ship one)")
    stack = Style().font_stack()
    assert installed_cjk & set(stack), f"no CJK font in {stack}"


def test_cjk_family_is_resolvable():
    stack = Style(cjk="songti").font_stack()
    assert "SimSun" in stack


def test_colourbar_scale_is_applied_at_artist_level():
    """rcParams cannot reach a colorbar label; this proves the 4th route works."""
    grid = pd.DataFrame(
        {
            "x_um": np.tile(np.arange(20, dtype=float), 20),
            "y_um": np.repeat(np.arange(20, dtype=float), 20),
            "intensity_au": np.random.default_rng(2).normal(size=400),
        }
    )
    p = analyze_dataframe(grid)
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    small = render(p, rec, style=Style(colorbar_scale=0.6)).axes[-1].yaxis.label.get_fontsize()
    big = render(p, rec, style=Style(colorbar_scale=1.8)).axes[-1].yaxis.label.get_fontsize()
    assert big > small * 2


# ── validation at the boundary ──────────────────────────────────────


# ── C axes: spines, ticks, grid, limits ────────────────────────────


def spine(ax, name):
    return {"top": ax.spines["top"], "right": ax.spines["right"],
            "left": ax.spines["left"], "bottom": ax.spines["bottom"]}[name]


def test_default_hides_top_and_right_spines():
    ax = drawn()[1]
    assert not spine(ax, "top").get_visible()
    assert not spine(ax, "right").get_visible()
    assert spine(ax, "left").get_visible() and spine(ax, "bottom").get_visible()


def test_spines_all_shows_four():
    ax = drawn({"spines": "all"})[1]
    assert all(spine(ax, s).get_visible() for s in ("top", "right", "left", "bottom"))


def test_spine_width_and_colour_reach_the_artists():
    ax = drawn({"spines": "all", "spine_width": 2.5, "spine_color": "#FF0000"})[1]
    assert spine(ax, "top").get_linewidth() == 2.5
    assert spine(ax, "top").get_edgecolor()[:3] == (1.0, 0.0, 0.0)


def test_tick_direction_reaches_the_context():
    """Matplotlib exposes no public reader for a tick's direction, so the
    end-to-end proof is the pixel test below."""
    for direction in ("in", "out", "inout"):
        assert Style(tick_direction=direction).rc_params()["xtick.direction"] == direction


def test_axis_knobs_change_the_rendered_pixels():
    """The rcParam dict is not the contract - the picture is. `font.family` once
    passed every metadata assertion while producing byte-identical figures, so
    each knob that Matplotlib reads lazily has to be proven here."""
    import hashlib

    def digest(**kw):
        fig = render(profile(), recommend(profile())[0], style=Style(**kw))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        fig.clear()
        return hashlib.sha256(buf.getvalue()).hexdigest()

    base = digest()
    for kw in (
        {"tick_direction": "in"},
        {"tick_length": 12.0},
        {"grid": "major"},
        {"spines": "all"},
        {"tick_minor": True},
        {"tick_max": 3},
        {"tick_precision": 1},
        {"sci_power": 3},
        {"y_zero_centered": True},
        {"x_reverse": True},
    ):
        assert digest(**kw) != base, f"{kw} changed nothing"


def test_minor_ticks_can_be_switched_on():
    assert len(drawn({"tick_minor": False})[1].xaxis.get_minor_ticks()) == 0
    assert len(drawn({"tick_minor": True})[1].xaxis.get_minor_ticks()) > 0


def test_tick_rotation_reaches_the_labels():
    ax = drawn({"tick_rotation": 45})[1]
    assert {round(t.get_rotation()) for t in ax.get_xticklabels()} == {45}


def test_tick_max_limits_the_number_of_ticks():
    many = len(drawn({"tick_max": 3})[1].get_xticks())
    assert many <= 4  # MaxNLocator prunes the upper edge


def test_tick_precision_formats_the_labels():
    ax = drawn({"tick_precision": 2})[1]
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels and all(len(v.split(".")[1]) == 2 for v in labels if "." in v)


def test_sci_power_moves_the_exponent_into_the_offset_text():
    """Large wavelengths would otherwise print as 1200, 1400… Matplotlib puts the
    exponent in the axis offset text, not in each label."""
    plain = drawn()[1].xaxis.get_offset_text().get_text()
    sci = drawn({"sci_power": 3})[1].xaxis.get_offset_text().get_text()
    assert plain == ""
    assert sci.replace("−", "-") in {"$10^{3}$", "1e3", "$\\mathdefault{10^{3}}$"}


def test_grid_off_by_default_and_switchable():
    def visible(option):
        ax = drawn(option)[1]
        return any(line.get_visible() for line in ax.xaxis.get_gridlines())

    assert visible({"grid": "off"}) is False
    assert visible({"grid": "major"}) is True
    assert visible({"grid": "both"}) is True


def test_grid_colour_and_style_reach_the_lines():
    ax = drawn({"grid": "major", "grid_style": "-.", "grid_color": "#00FF00",
                "grid_width": 1.5, "grid_alpha": 0.25})[1]
    line = next(l for l in ax.xaxis.get_gridlines() if l.get_visible())
    assert line.get_linestyle() == "-."
    assert line.get_linewidth() == 1.5
    assert line.get_alpha() == 0.25


def test_grid_under_data_maps_to_axisbelow():
    assert drawn({"grid_under_data": True})[1].get_axisbelow() is True
    assert drawn({"grid_under_data": False})[1].get_axisbelow() is False


def test_axis_margin_widens_the_autoscaled_range():
    tight = drawn({"axis_margin": 0.0})[1].get_xlim()
    loose = drawn({"axis_margin": 0.3})[1].get_xlim()
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_explicit_limits_win():
    ax = drawn({"x_min": 1300, "x_max": 1600})[1]
    assert ax.get_xlim() == (1300.0, 1600.0)


def test_y_zero_centered_makes_the_range_symmetric():
    ax = drawn({"y_zero_centered": True})[1]
    bottom, top = ax.get_ylim()
    assert bottom == pytest.approx(-top)


def test_reversed_axes():
    ax = drawn({"x_reverse": True})[1]
    assert ax.get_xlim()[0] > ax.get_xlim()[1]


def test_image_and_diagram_types_keep_their_data_and_stay_ungridded():
    """Two things are genuinely wrong on an image-type figure: grid lines over a
    pcolormesh are noise, and cropping a heatmap axis hides measured samples
    while looking like a cosmetic choice. Spines and tick direction stay
    available - four-sided frames on a beam map are common and harmless."""
    grid = pd.DataFrame(
        {
            "x_um": np.tile(np.arange(20, dtype=float), 20),
            "y_um": np.repeat(np.arange(20, dtype=float), 20),
            "intensity_au": np.random.default_rng(6).normal(size=400),
        }
    )
    p = analyze_dataframe(grid)
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    ax = render(p, rec, style=Style(grid="both", x_min=5.0, x_max=10.0)).axes[0]
    assert not any(l.get_visible() for l in ax.xaxis.get_gridlines())
    assert ax.get_xlim() != (5.0, 10.0), "heatmap axis range must not be cropped"
    # ...while the same style still frames it normally
    assert not spine(ax, "top").get_visible()
    framed = render(p, rec, style=Style(spines="all")).axes[0]
    assert spine(framed, "top").get_visible()


# ── E legend ───────────────────────────────────────────────────────


def test_legend_none_suppresses_the_legend_entirely():
    assert drawn()[1].get_legend() is not None
    assert drawn({"legend": "none"})[1].get_legend() is None


def test_legend_columns_reach_the_legend():
    """Legend has set_ncols but no getter, so the observable proof is the pixel
    digest test at the end of this file."""
    assert Style(legend_columns=3).legend_kwargs() == {"ncols": 3}


def test_legend_frame_and_handle_length_are_applied():
    plain = drawn()[1]
    framed = drawn({"legend_frame": True, "legend_frame_alpha": 1.0})[1]
    assert not plain.get_legend().get_frame().get_visible()
    assert framed.get_legend().get_frame().get_visible()
    assert Style(legend_handle_length=3.0).rc_params()["legend.handlelength"] == 3.0


def test_legend_font_size_follows_the_base_size():
    """The legend used to be pinned at fontsize="small" regardless of canvas."""
    small = Style(font_size=8).rc_params()["legend.fontsize"]
    large = Style(font_size=16).rc_params()["legend.fontsize"]
    assert large > small * 1.8


# ── F colour ───────────────────────────────────────────────────────


def test_default_palette_is_still_okabe_itto():
    assert Style().colors[0] == "#0072B2"
    assert Style().colors[1] == "#D55E00"


def test_palette_switches_the_series_colours():
    first = drawn({"palette": "okabe"})[1].lines[0].get_color()
    other = drawn({"palette": "tol_vibrant"})[1].lines[0].get_color()
    assert first != other


def test_custom_palette_reaches_the_artists_and_cycles():
    """sample_spectrum has three response columns and gets three colours, so the
    fourth artist proves the palette cycles instead of running off the end."""
    p = analyze_file(ROOT / "examples" / "sample_spectrum.csv")
    rec = next(r for r in recommend(p) if r.id == "spectrum_lines")
    ax = render(p, rec, style=Style(palette="custom",
                                    custom_colors="#123456,#ABCDEF")).axes[0]
    assert ax.lines[0].get_color() == "#123456"
    assert ax.lines[1].get_color() == "#ABCDEF"
    assert ax.lines[2].get_color() == "#123456"


def test_custom_colours_without_the_custom_palette_is_an_error():
    """Otherwise the user types six hex codes and nothing happens."""
    with pytest.raises(ValueError, match="custom"):
        Style(custom_colors="#123456,#ABCDEF").validate()


@pytest.mark.parametrize("bad", ["#123", "red", "#GGHHII", "123456", "#12345", ""])
def test_malformed_custom_colours_are_rejected(bad):
    with pytest.raises(ValueError):
        Style(palette="custom", custom_colors=bad).validate()


def test_shipped_palettes_have_more_slots_than_the_default_line_limit():
    """Every series beyond the palette length silently reuses a colour, so a
    short palette is a real limit on how many curves stay distinguishable."""
    from optiplot.style import PALETTES

    for name, colors in PALETTES.items():
        assert len(colors) >= 6, f"{name} offers only {len(colors)} colours"


def test_cmap_now_comes_from_style_not_the_data_options():
    grid = pd.DataFrame(
        {
            "x_um": np.tile(np.arange(16, dtype=float), 16),
            "y_um": np.repeat(np.arange(16, dtype=float), 16),
            "intensity_au": np.random.default_rng(7).normal(size=256),
        }
    )
    p = analyze_dataframe(grid)
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    viridis = render(p, rec, style=Style(cmap="viridis")).axes[0].collections[0].get_cmap().name
    magma = render(p, rec, style=Style(cmap="magma")).axes[0].collections[0].get_cmap().name
    assert (viridis, magma) == ("viridis", "magma")


def test_cmap_auto_still_picks_a_diverging_map_for_signed_data():
    """The zero-centred default is a scientific safeguard, not cosmetics."""
    grid = pd.DataFrame(
        {
            "x_um": np.tile(np.arange(16, dtype=float), 16),
            "y_um": np.repeat(np.arange(16, dtype=float), 16),
        }
    )
    grid["delta"] = np.linspace(-1, 1, 256).reshape(16, 16).ravel()
    p = analyze_dataframe(grid)
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    assert render(p, rec, style=Style()).axes[0].collections[0].get_cmap().name == "RdBu_r"


def test_legend_and_palette_knobs_change_the_rendered_pixels():
    import hashlib

    def digest(**kw):
        fig = render(profile(), recommend(profile())[0], style=Style(**kw))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        fig.clear()
        return hashlib.sha256(buf.getvalue()).hexdigest()

    base = digest()
    for kw in (
        {"legend": "none"},
        {"legend_columns": 2},
        {"legend_frame": True, "legend_frame_alpha": 1.0},
        {"palette": "tol_muted"},
        {"palette": "custom", "custom_colors": "#111111,#222222,#333333"},
    ):
        assert digest(**kw) != base, f"{kw} changed nothing"


# ── D data series ─────────────────────────────────────────────────


def test_line_width_reaches_the_line():
    assert drawn({"line_width": 4.0})[1].lines[0].get_linewidth() == 4.0


def test_line_style_cycles_only_when_asked():
    ax = drawn({"vary_line_style": True})[1]
    styles = {l.get_linestyle() for l in ax.lines}
    assert len(styles) > 1
    flat = drawn({"vary_line_style": False})[1]
    assert {l.get_linestyle() for l in flat.lines} == {"-"}


def test_markers_can_be_added_and_hollowed():
    solid = drawn({"marker": "o", "marker_size": 9.0})[1].lines[0]
    assert solid.get_marker() == "o" and solid.get_markersize() == 9.0
    hollow = drawn({"marker": "o", "marker_fill": "none"})[1].lines[0]
    assert hollow.get_fillstyle() == "none"


def test_marker_every_thins_the_markers():
    line = drawn({"marker": "o", "marker_every": 5})[1].lines[0]
    assert line.get_markevery() == 5


def test_step_switches_to_a_stepped_line():
    assert drawn({"step": True})[1].lines[0].get_drawstyle() == "steps-mid"
    assert drawn({"step": False})[1].lines[0].get_drawstyle() == "default"


def test_points_only_drops_the_connecting_line():
    """The default must not invent a trend between unconnected measurements."""
    line = drawn({"show_lines": False, "show_points": True})[1].lines[0]
    assert line.get_linestyle() == "None"
    assert line.get_linewidth() == 0.0


def test_series_alpha_and_zorder_reach_the_line():
    line = drawn({"series_alpha": 0.4, "series_zorder": 7})[1].lines[0]
    assert line.get_alpha() == 0.4
    assert line.get_zorder() == 7


def test_scatter_knobs_reach_the_collection():
    p = analyze_file(ROOT / "examples" / "sample_dense_scatter.csv")
    rec = next(r for r in recommend(p) if r.id == "scatter_fit")
    coll = render(p, rec, style=Style(scatter_size=60.0, scatter_alpha=0.2,
                                      scatter_edge=True)).axes[0].collections[0]
    assert coll.get_sizes()[0] == 60.0
    assert coll.get_alpha() == 0.2
    assert (coll.get_edgecolors() != 0).any()


def test_rasterize_threshold_is_respected():
    p = analyze_file(ROOT / "examples" / "sample_dense_scatter.csv")
    rec = next(r for r in recommend(p) if r.id == "scatter_fit")
    assert render(p, rec, style=Style(rasterize_above=10)).axes[0].collections[0].get_rasterized()
    assert not render(
        p, rec, style=Style(rasterize_above=10_000_000)
    ).axes[0].collections[0].get_rasterized()


def test_errorbar_cap_and_line_width_reach_the_caps():
    p = analyze_file(ROOT / "examples" / "sample_explicit_error.csv")
    rec = next(r for r in recommend(p) if r.id == "errorbar")
    ax = render(p, rec, style=Style(error_cap_size=9.0, error_line_width=2.5)).axes[0]
    widths = []
    for part in ax.containers[0]:
        for item in part if isinstance(part, (list, tuple)) else [part]:
            value = item.get_linewidth()
            widths.extend(float(v) for v in (value if hasattr(value, "__len__") else [value]))
    assert 2.5 in widths, f"error_line_width never reached an artist: {widths}"


def test_contour_level_count_reaches_the_artist():
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "contour")
    few = render(p, rec, style=Style(contour_levels=3)).axes[0].collections[0]
    many = render(p, rec, style=Style(contour_levels=40)).axes[0].collections[0]
    assert len(few.get_array()) < len(many.get_array())


def test_hexbin_gridsize_changes_the_bin_count():
    p = analyze_file(ROOT / "examples" / "sample_dense_scatter.csv")
    rec = next(r for r in recommend(p) if r.id == "density")
    coarse = render(p, rec, style=Style(hexbin_gridsize=8)).axes[0].collections[0]
    fine = render(p, rec, style=Style(hexbin_gridsize=80)).axes[0].collections[0]
    assert len(coarse.get_offsets()) < len(fine.get_offsets())


def test_image_shading_reaches_the_mesh():
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    # QuadMesh has no get_shading(), and matplotlib silently substitutes "auto"
    # for an unrecognised value, so assert on the mesh geometry the mode produced
    # rather than on the string.
    def rows(shading):
        return render(p, rec, style=Style(image_shading=shading)).axes[0]             .collections[0].get_coordinates()[0].shape[0]

    assert rows("nearest") > rows("gouraud")


def test_colorbar_geometry_reaches_the_layout():
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    thin = render(p, rec, style=Style(colorbar_thickness=0.02)).axes[-1].get_position().width
    fat = render(p, rec, style=Style(colorbar_thickness=0.2)).axes[-1].get_position().width
    # constrained layout redistributes, so the ratio compresses; monotonic is the real contract
    assert fat > thin * 1.5


def test_every_offered_choice_is_actually_accepted_by_matplotlib():
    """Matplotlib silently substitutes an unrecognised shading value and only
    emits a warning, so a choice list can advertise an option that does nothing.
    Assert no such substitution warning fires for any value we offer."""
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    for shading in SHADINGS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render(p, rec, style=Style(image_shading=shading))
        noisy = [str(w.message) for w in caught if "not in list of valid values" in str(w.message)]
        assert not noisy, f"image_shading={shading!r} was rejected: {noisy}"


def test_every_offered_marker_and_fillstyle_survives_a_real_draw():
    """Same class of failure as shading, and this guard has to actually draw:
    a bad fillstyle only surfaces when the marker is rendered, so constructing
    the figure and checking the attribute would have let it through."""
    p = profile()
    rec = recommend(p)[0]
    for marker in MARKERS:
        if marker == "none":
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax = render(p, rec, style=Style(marker=marker)).axes[0]
            FigureCanvasAgg(ax.figure)
            ax.figure.canvas.draw()
        assert ax.lines[0].get_marker() == marker
        assert not [w for w in caught if w.category is UserWarning], (
            f"marker={marker!r} warned: {[str(w.message) for w in caught]}"
        )
    for fill in MARKER_FILLS:
        ax = render(p, rec, style=Style(marker="s", marker_fill=fill)).axes[0]
        FigureCanvasAgg(ax.figure)
        ax.figure.canvas.draw()
        assert ax.lines[0].get_fillstyle() == fill, f"fillstyle={fill!r} not applied"


@pytest.mark.parametrize(
    "bad",
    [
        {"line_width": 0},
        {"line_width": 50},
        {"marker": "hexagon"},
        {"marker_fill": "striped"},
        {"marker_size": -1},
        {"marker_every": 0},
        {"series_alpha": 2},
        {"scatter_alpha": -0.5},
        {"fill_alpha": 9},
        {"error_cap_size": 99},
        {"contour_levels": 0},
        {"hexbin_gridsize": 2},
        {"image_shading": "bilinear"},
        {"colorbar_thickness": 0},
        {"colorbar_pad": 5},
        {"rasterize_above": 0},
        {"series_zorder": 9999},
        {"show_lines": False, "show_points": False},
    ],
)
def test_bad_series_values_are_rejected(bad):
    with pytest.raises(ValueError):
        Style(**bad).validate()


def test_series_knobs_change_the_rendered_pixels():
    import hashlib

    def digest(**kw):
        fig = render(profile(), recommend(profile())[0], style=Style(**kw))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        fig.clear()
        return hashlib.sha256(buf.getvalue()).hexdigest()

    base = digest()
    for kw in (
        {"line_width": 5.0},
        {"vary_line_style": False},
        {"marker": "s", "marker_size": 8.0},
        {"marker": "o", "marker_fill": "none"},
        {"marker": "o", "marker_every": 6},
        {"step": True},
        {"show_lines": False, "show_points": True},
        {"series_alpha": 0.3},
    ):
        assert digest(**kw) != base, f"{kw} changed nothing"


@pytest.mark.parametrize(
    "bad",
    [
        {"spines": "diagonal"},
        {"tick_direction": "sideways"},
        {"grid": "sometimes"},
        {"grid_alpha": 4},
        {"axis_margin": 2.0},
        {"tick_rotation": 180},
        {"tick_max": 1},
        {"tick_precision": 99},
        {"sci_power": 40},
        {"spine_width": 90},
        {"x_min": 5, "x_max": 5},
        {"y_min": 1, "y_max": 0},
        {"y_zero_centered": True, "y_min": 0, "y_max": 1},
        {"legend": "wherever"},
        {"legend_columns": 0},
        {"legend_title_size": "gigantic"},
        {"legend_handle_length": 99},
        {"palette": "rainbow"},
        {"palette": "custom"},
    ],
)
def test_bad_style_values_are_rejected(bad):
    with pytest.raises(ValueError):
        Style(**bad).validate()


@pytest.mark.parametrize(
    "bad",
    [
        {"family": "Comic Sans Nonexistent"},
        {"cjk": "WuAi"},
        {"font_size": 300},
        {"font_size": 1},
        {"title_scale": 0.01},
        {"label_pad": -5},
        {"size": "a4"},
        {"title_weight": "heavy"},
    ],
)
def test_bad_typography_values_are_rejected(bad):
    with pytest.raises(ValueError):
        Style(**bad).validate()


def test_unknown_style_key_is_not_silently_swallowed():
    """A typo must not land in the data half and be ignored."""
    with pytest.raises(ValueError, match="font_sze"):
        Style.from_options({"font_sze": 12})


def test_data_keys_stay_out_of_the_style():
    style, opts = Style.from_options({"size": "single", "fit": "linear", "title": "T"})
    assert style.size == "single"
    assert opts == {"fit": "linear", "title": "T"}
    assert not set(opts) & {f.name for f in Style.__dataclass_fields__.values()}


def test_every_documented_data_key_is_recognised():
    style, opts = Style.from_options({k: None for k in DATA_KEYS})
    assert set(opts) == set(DATA_KEYS)


# ── presets and reproducibility ─────────────────────────────────────


def test_style_round_trips_through_json():
    """Group I presets and the reproducible ZIP both serialise a Style."""
    s = Style(family="serif", font_size=12, tick_scale=1.3, label_pad=11)
    back = Style.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back == s


def test_style_is_immutable_once_validated():
    s = Style()
    with pytest.raises(Exception):
        s.font_size = 99  # frozen dataclass


def test_style_object_and_options_dict_agree():
    a = drawn({"font_size": 13})[1].xaxis.label.get_fontsize()
    b = drawn(style=Style(font_size=13))[1].xaxis.label.get_fontsize()
    assert a == pytest.approx(b)
