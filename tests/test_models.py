"""The fit library: parameter recovery, refusals, and the honesty checks.

Every model here is tested against data whose parameters are known, because a
fitting library that has never been shown to recover the truth is a way of
attaching false authority to a curve.
"""

import numpy as np
import pandas as pd
import pytest
from optiplot.models import (
    MODELS,
    MODEL_BY_ID,
    FitModel,
    fit_model,
    model_for,
)

NOISE = np.random.default_rng(20260922).normal(0.0, 0.002, 200)


def model(identifier):
    return MODEL_BY_ID[identifier]


def fit(identifier, x, y, x_factor=1.0):
    return fit_model(model(identifier), np.asarray(x, float), np.asarray(y, float), x_factor)


# ── recovery ──────────────────────────────────────────────────────
def test_single_exponential_recovers_its_three_parameters():
    x = np.linspace(0.0, 8.0, 80)
    r = fit("single_exponential", x, 1.0 + 2.5 * np.exp(-x / 1.7) + NOISE[: 80])
    assert r.params["A"] == pytest.approx(2.5, abs=0.05)
    assert r.params["tau"] == pytest.approx(1.7, abs=0.05)
    assert r.params["C"] == pytest.approx(1.0, abs=0.05)
    assert r.r2 > 0.999
    assert r.residual.size == 80


def test_double_exponential_separates_two_time_constants():
    x = np.linspace(0.0, 8.0, 80)
    y = 1.0 + 1.8 * np.exp(-x / 0.6) + 0.9 * np.exp(-x / 3.2)
    r = fit("double_exponential", x, y + NOISE[:80])
    assert r.params["tau1"] == pytest.approx(0.6, rel=0.05)
    assert r.params["tau2"] == pytest.approx(3.2, rel=0.05)
    assert not any("单指数已能描述" in n for n in r.notes)


def test_a_single_decay_fitted_as_two_components_still_describes_the_curve():
    """Deciding whether the second component is needed is model selection, which
    this library refuses to do for the user; what it guarantees is that the curve
    is still right when they over-parameterise it."""
    x = np.linspace(0.0, 8.0, 80)
    y = 1.0 + 1.4 * np.exp(-x / 1.5)
    r = fit("double_exponential", x, y + NOISE[:80])
    assert np.max(np.abs(r.curve(x) - y)) < 0.02
    assert "两个时间常数接近" in model("double_exponential").caveat


def test_power_law_recovers_the_exponent():
    x = np.linspace(0.5, 8.5, 80)
    r = fit("power_law", x, 0.3 + 2.0 * x**1.8 + NOISE[:80])
    assert r.params["k"] == pytest.approx(1.8, abs=0.05)
    assert r.params["A"] == pytest.approx(2.0, abs=0.1)


def test_peak_shapes_recover_centre_and_width():
    x = np.linspace(0.0, 8.0, 161)
    gaussian = 0.1 + 2.0 * np.exp(-((x - 4.0) ** 2) / (2 * 0.8**2))
    lorentzian = 0.1 + 2.0 * 0.6**2 / ((x - 4.0) ** 2 + 0.6**2)
    g = fit("gaussian", x, gaussian + NOISE[:161])
    l = fit("lorentzian", x, lorentzian + NOISE[:161])
    assert g.params["mu"] == pytest.approx(4.0, abs=0.05)
    assert g.params["sigma"] == pytest.approx(0.8, abs=0.05)
    assert l.params["mu"] == pytest.approx(4.0, abs=0.05)
    assert l.params["gamma"] == pytest.approx(0.6, abs=0.05)


def test_malus_finds_the_order_without_being_told_it():
    """The starting order cannot be guessed from a curve showing two humps: it is
    cos²(θ) over 360° just as well as cos²(9θ) over 40°."""
    for order in (1.0, 3.0, 7.0):
        theta = np.linspace(0.0, 360.0, 181)
        y = 0.05 + np.cos(np.deg2rad(order * theta)) ** 2
        r = fit("malus", theta, y + NOISE[:181], x_factor=np.pi / 180)
        assert r.params["k"] == pytest.approx(order, rel=0.02), (order, r.params)


def test_malus_follows_the_angle_unit_it_is_given():
    theta = np.linspace(0.0, 2 * np.pi, 181)
    y = 0.05 + np.cos(2 * theta) ** 2
    r = fit("malus", theta, y + NOISE[:181])
    assert r.params["k"] == pytest.approx(2.0, rel=0.02)


def test_dispersion_recovers_the_pole_only_when_data_approach_it():
    """Far from the oscillator the curve is right and the parameters are not,
    which is the failure this model's caveat exists to say out loud."""
    near = np.linspace(0.55, 0.9, 60)
    far = np.linspace(1.0, 1.6, 60)
    truth = lambda lam: 1.45 + 0.15 * 0.25 / (lam**2 - 0.25)
    close = fit("drude_lorentz", near, truth(near) + NOISE[:60] * 0.25)
    distant = fit("drude_lorentz", far, truth(far) + NOISE[:60] * 0.25)
    for r, lam in ((close, near), (distant, far)):
        assert np.max(np.abs(r.curve(lam) - truth(lam))) < 0.002
        assert not (lam.min() <= r.params["lambda0"] <= lam.max()), "the pole was fitted through"
    assert close.params["lambda0"] == pytest.approx(0.5, rel=0.02)
    assert "远离极点" in model("drude_lorentz").caveat


def test_dispersion_reports_which_side_the_pole_was_placed_on():
    lam = np.linspace(1.0, 1.6, 40)
    r = fit("drude_lorentz", lam, 1.45 + 0.05 / (lam**2 - 0.25))
    assert any("短波侧" in n for n in r.notes), r.notes


# ── refusals ───────────────────────────────────────────────────────
def test_too_few_points_is_refused_not_fitted():
    x = np.linspace(0.0, 4.0, 4)
    with pytest.raises(ValueError, match="至少"):
        fit("double_exponential", x, np.exp(-x))


def test_repeated_x_values_cannot_identify_parameters():
    x = np.repeat([1.0, 2.0], 30)
    y = np.tile([0.5, 0.2], 30)
    with pytest.raises(ValueError, match="不可辨识"):
        fit("gaussian", x, y)


def test_power_law_refuses_a_non_positive_axis():
    x = np.linspace(0.0, 5.0, 40)
    with pytest.raises(ValueError, match="全部为正"):
        fit("power_law", x, np.sqrt(x))


def test_dispersion_on_pole_crossing_data_reports_a_bad_fit_not_a_fake_pole():
    """The model cannot know a pole lay inside the scan; it can only keep the pole
    out of the fitted range and say the curve did not come out."""
    lam = np.linspace(0.3, 0.8, 40)
    y = 1.4 + 0.05 / (lam**2 - 0.25)
    r = fit("drude_lorentz", lam, y[np.isfinite(y)])
    assert not (0.3 <= r.params["lambda0"] <= 0.8)
    assert any("偏低" in n for n in r.notes) or r.r2 < 0.8, r.notes


def test_an_unknown_model_is_an_error_not_a_silent_none():
    with pytest.raises(ValueError, match="未知拟合模型"):
        model_for("triple_exponential")


def test_a_flat_response_has_no_r_squared():
    x = np.linspace(0.0, 8.0, 40)
    r = fit("single_exponential", x, np.full(40, 0.7))
    assert not np.isfinite(r.r2)
    assert any("R² 无定义" in n for n in r.notes)


# ── the fit itself ─────────────────────────────────────────────────
def test_the_wrong_shape_fits_far_worse_than_the_right_one():
    """The point of publishing residuals: R² alone can look fine on a model with
    the wrong wings, and RMSE against the same peak is where it shows."""
    x = np.linspace(0.0, 8.0, 161)
    gaussian = 0.1 + 2.0 * np.exp(-((x - 4.0) ** 2) / (2 * 0.8**2))
    right = fit("gaussian", x, gaussian)
    wrong = fit("lorentzian", x, gaussian)
    assert wrong.rmse > 10 * right.rmse


def test_curve_uses_the_fitted_parameters():
    x = np.linspace(0.0, 8.0, 40)
    r = fit("single_exponential", x, 1.0 + 2.5 * np.exp(-x / 1.7))
    p = r.params
    assert r.curve(x) == pytest.approx(p["C"] + p["A"] * np.exp(-x / p["tau"]), abs=1e-9)


def test_label_names_parameters_and_reports_goodness():
    x = np.linspace(0.0, 8.0, 40)
    r = fit("single_exponential", x, 1.0 + 2.5 * np.exp(-x / 1.7))
    assert r.label().startswith("单指数衰减: A=")
    assert "R²=" in r.label()


def test_every_model_is_fittable_and_carries_a_caveat():
    """A model without a caveat claims unconditional applicability, which no
    physical line shape has."""
    x = np.linspace(1.0, 6.0, 40)
    y = 0.2 + np.sin(x) / x
    assert len({m.id for m in MODELS}) == len(MODELS)
    for m in MODELS:
        assert m.caveat, m.id
        assert m.min_points >= len(m.parameters) + 1
        assert len(m.guess(x, y)) == len(m.parameters), m.id
        starts = m.starts(x, y) if m.starts else [m.guess(x, y)]
        assert all(len(s) == len(m.parameters) for s in starts), m.id


def test_model_ids_are_stable_enough_to_store_in_a_recipe():
    assert model_for("malus") is model("malus")
    assert [m.id for m in MODELS] == [
        "single_exponential",
        "double_exponential",
        "power_law",
        "gaussian",
        "lorentzian",
        "drude_lorentz",
        "malus",
    ]


def test_a_parameter_beyond_the_scan_is_called_out_as_a_constant():
    """A straight decline fitted as a decay puts the time constant four orders
    away from anything this scan could measure, and the figure must say so rather
    than print tau=3.5e4 as a lifetime."""
    x = np.linspace(0.0, 8.0, 60)
    r = fit("single_exponential", x, 1.0 - 0.001 * x)
    assert r.params["tau"] > 40 * _span_of(x)
    assert any("只起常数作用" in n for n in r.notes), (r.label(), r.notes)
    # a genuine second component inside the scan is not flagged
    honest = fit("double_exponential", x, 0.05 + 1.8 * np.exp(-x / 0.7) + 0.6 * np.exp(-x / 4.5))
    assert not any("只起常数作用" in n for n in honest.notes), honest.notes


def _span_of(x):
    return float(np.max(x) - np.min(x))


# ── the figure ─────────────────────────────────────────────────────
import subprocess
import sys
import zipfile

from optiplot import analyze_dataframe, recommend
from optiplot.export import export_bundle
from optiplot.render import FIT_CHOICES, render


def decay_profile():
    x = np.linspace(0.0, 8.0, 60)
    y = 0.05 + 2.4 * np.exp(-x / 1.9) + NOISE[:60]
    return analyze_dataframe(pd.DataFrame({"decay_time_ns": x, "photoluminescence": y}))


def pick(profile, identifier):
    return next(r for r in recommend(profile) if r.id == identifier)


def test_a_named_model_brings_its_residual_panel():
    p = decay_profile()
    fig = render(p, pick(p, "scatter_fit"), options={"fit": "single_exponential"})
    assert len(fig.axes) == 2
    main, strip = fig.axes
    assert len(main.lines) == 1, "the fit curve is missing"
    assert "R²=" in main.lines[0].get_label()
    assert len(strip.get_xticks()) and strip.get_ylabel() == "残差"
    # the residuals are drawn as one stem per measurement, at the measured x
    segments = strip.collections[0].get_segments()
    assert len(segments) == 60
    assert all(float(s[0][1]) == 0.0 for s in segments), "stems must start at zero"


def test_the_shared_axis_is_labelled_once():
    p = decay_profile()
    fig = render(p, pick(p, "spectrum_lines"), options={"fit": "single_exponential"})
    main, strip = fig.axes
    assert main.get_xlabel() == ""
    assert not any(l.get_visible() for l in main.get_xticklabels())
    assert strip.get_xlabel() == "decay_time_ns"


def test_a_linear_fit_still_gets_no_panel():
    p = decay_profile()
    fig = render(p, pick(p, "scatter_fit"), options={"fit": "linear"})
    assert len(fig.axes) == 1


def test_a_misspelled_model_is_refused_not_dropped():
    p = decay_profile()
    with pytest.raises(ValueError, match="未知拟合选项"):
        render(p, pick(p, "scatter_fit"), options={"fit": "typo_exponential"})


def test_a_curve_family_refuses_one_fit_over_the_top_of_it():
    """Fitting a single curve through fifteen angle-resolved spectra would report
    one line shape for a set that is defined by how it varies."""
    theta = np.repeat(np.linspace(0.0, 70.0, 8), 40)
    lam = np.tile(np.linspace(1300.0, 1700.0, 40), 8)
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "theta_deg": theta,
                "wavelength_nm": lam,
                "reflectance": 1.0 - 0.5 * np.exp(-((lam - 1500.0 - 4 * theta) ** 2) / 2000.0),
            }
        )
    )
    with pytest.raises(ValueError, match="不能整体拟合"):
        render(p, pick(p, "spectrum_lines"), options={"fit": "gaussian"})


def test_a_model_fit_survives_the_reproducible_bundle(tmp_path):
    """The bundle copies render.py and whatever it imports by bare name; a model
    fit is the case that needs models.py alongside it."""
    p = decay_profile()
    bundle = export_bundle(
        p, pick(p, "scatter_fit"), tmp_path / "f.zip", {"fit": "single_exponential"}
    )
    out = tmp_path / "standalone"
    out.mkdir()
    with zipfile.ZipFile(bundle) as z:
        assert "models.py" in z.namelist()
        z.extractall(out)
    subprocess.run(
        [sys.executable, str(out / "render_plot.py")],
        cwd=out,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "reproduced.png").stat().st_size > 1000


def test_every_offered_fit_choice_either_draws_or_refuses_loudly():
    """The list the interface offers must not contain an entry that quietly draws
    nothing. A model that cannot apply to this data says why instead."""
    assert FIT_CHOICES[0] == "none"
    p = decay_profile()
    for choice in FIT_CHOICES[1:]:
        try:
            fig = render(p, pick(p, "scatter_fit"), options={"fit": choice})
        except ValueError as exc:
            assert str(exc), choice
            continue
        assert len(fig.axes) == (1 if choice == "linear" else 2), choice
        assert "R²=" in fig.axes[0].lines[0].get_label(), choice
