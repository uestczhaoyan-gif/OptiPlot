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
from optiplot import analyze_dataframe, recommend
from optiplot.render import render
from optiplot.style import Style, SIZES, PRESET_FONT, DATA_KEYS, installed_families

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
def test_bad_values_are_rejected(bad):
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
