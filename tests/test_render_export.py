"""End-to-end plot, export and data-integrity regression checks."""

from pathlib import Path
import json
import re
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from optiplot import analyze_file, analyze_dataframe, recommend, Recommendation
from optiplot.render import render, FIGURE_TYPES
from optiplot.style import Style
from optiplot.export import export_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.csv")), ids=lambda p: p.stem)
def test_every_recommendation_renders(path, tmp_path):
    profile = analyze_file(path)
    for r in recommend(profile):
        out = tmp_path / (r.id + ".png")
        fig = render(profile, r, out)
        assert out.stat().st_size > 1000
        fig.clear()


def test_no_automatic_fit_and_keeps_missing_gaps():
    p = analyze_dataframe(pd.DataFrame({"time_s": [0, 1, 2, 3, 4], "signal": [1, 2, np.nan, 4, 5]}))
    scatter = next(r for r in recommend(p) if r.id == "scatter_fit")
    fig = render(p, scatter)
    assert len(fig.axes[0].lines) == 0
    line = next(r for r in recommend(p) if r.id == "spectrum_lines")
    fig = render(p, line)
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[2])


def test_grouped_replicates_are_not_pooled():
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "power_mW": [1, 1, 2, 2] * 2,
                "device": ["A"] * 4 + ["B"] * 4,
                "signal": [1, 3, 3, 5, 10, 12, 12, 14],
            }
        )
    )
    rec = next(r for r in recommend(p) if r.id == "errorbar")
    fig = render(p, rec)
    values = [line.get_ydata() for line in fig.axes[0].lines if len(line.get_ydata()) == 2]
    assert any(np.allclose(a, [2, 4]) for a in values)
    assert any(np.allclose(a, [11, 13]) for a in values)


def test_portable_export_runs_without_project(tmp_path):
    p = analyze_file(ROOT / "examples/sample_spectrum.csv")
    rec = recommend(p)[0]
    bundle = export_bundle(p, rec, tmp_path / "figure.zip", {"size": "single"})
    out = tmp_path / "standalone"
    out.mkdir()
    with zipfile.ZipFile(bundle) as z:
        z.extractall(out)
    subprocess.run(
        [sys.executable, str(out / "render_plot.py")],
        cwd=out,
        check=True,
        capture_output=True,
        text=True,
    )
    for ext in ["png", "svg", "pdf"]:
        assert (out / f"reproduced.{ext}").stat().st_size > 1000
    assert "<text" in (out / "figure.svg").read_text(encoding="utf-8")
    assert (
        json.loads((out / "recipe.json").read_text(encoding="utf-8"))["recommendation"]["id"]
        == rec.id
    )


def test_negative_uncertainty_is_rejected():
    p = analyze_dataframe(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 4], "sd": [0.1, -0.1, 0.2]}))
    r = Recommendation("errorbar", "error", "high", "test", {"x": "x", "y": "y", "error": "sd"})
    with pytest.raises(ValueError, match="负"):
        render(p, r)


def test_angle_wavelength_grid_is_identified():
    w, a = np.meshgrid([500, 600, 700], [0, 45, 90, 135])
    p = analyze_dataframe(
        pd.DataFrame({"wavelength_nm": w.ravel(), "angle_deg": a.ravel(), "signal": np.arange(12)})
    )
    assert p.grid_like
    assert "heatmap" in {r.id for r in recommend(p)}


def test_matrix_heatmap_from_npy(tmp_path):
    path = tmp_path / "field.npy"
    np.save(path, np.arange(20).reshape(4, 5))
    p = analyze_file(path)
    r = next(r for r in recommend(p) if r.id == "matrix_heatmap")
    assert render(p, r).axes[0].collections[0].get_array().shape == (4, 5)


def test_group_with_singletons_has_no_invented_error():
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "power_mW": [1, 1, 2, 2, 1, 2],
                "device": ["A"] * 4 + ["B"] * 2,
                "signal": [1, 3, 3, 5, 11, 13],
            }
        )
    )
    r = next(r for r in recommend(p) if r.id == "errorbar")
    fig = render(p, r)
    assert len(fig.axes[0].collections) >= 2


def test_workflow_arrows_preserve_all_edges():
    p = analyze_dataframe(pd.DataFrame({"source": ["A", "B", "B"], "target": ["B", "C", "B"]}))
    r = next(r for r in recommend(p) if r.id == "flow")
    fig = render(p, r)
    # Three boxes, two arrows, plus a self-loop annotation.
    assert len(fig.axes[0].patches) == 5
    assert len(fig.axes[0].texts) == 4


def test_log_axes_reject_nonpositive_scatter():
    p = analyze_dataframe(pd.DataFrame({"x": [-1, 1, 2, 3], "y": [1, 2, 3, 4]}))
    r = next(r for r in recommend(p) if r.id == "scatter_fit")
    with pytest.raises(ValueError, match="大于 0"):
        render(p, r, options={"xlog": True})


def test_signed_heatmap_has_zero_centered_color():
    x, y = np.meshgrid([0, 1, 2], [0, 1, 2])
    p = analyze_dataframe(pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": np.arange(9) - 4}))
    r = next(r for r in recommend(p) if r.id == "heatmap")
    fig = render(p, r)
    assert fig.axes[0].collections[0].norm(0) == 0.5


def test_angle_grid_prefers_grid_not_collapsed_polar():
    w, a = np.meshgrid([500, 600, 700], [0, 45, 90, 135])
    p = analyze_dataframe(
        pd.DataFrame(
            {"wavelength_nm": w.ravel(), "angle_deg": a.ravel(), "signal": np.arange(12) + 1}
        )
    )
    assert recommend(p)[0].id == "heatmap"
    assert "polar" not in {r.id for r in recommend(p)}


def test_styles_reference_real_figure_types():
    styles = json.loads((ROOT / "catalog/styles.json").read_text(encoding="utf-8"))
    assert len({s["id"] for s in styles}) == len(styles)
    assert styles, "the figure-type library must not be empty"
    for s in styles:
        assert s["label"].strip() and s["data_schema"].strip() and s["recipe"].strip()
        assert s["patterns"], f"{s['id']} names no figure type"
        unknown = set(s["patterns"]) - set(FIGURE_TYPES)
        assert not unknown, f"{s['id']} references unimplemented types {unknown}"


def test_styles_carry_no_per_figure_provenance():
    """The product draws figures; it does not cite papers under each one."""
    styles = json.loads((ROOT / "catalog/styles.json").read_text(encoding="utf-8"))
    allowed = {"id", "label", "patterns", "data_schema", "recipe"}
    for s in styles:
        assert set(s) == allowed, f"{s['id']} carries {set(s) - allowed}"
    text = (ROOT / "catalog/styles.json").read_text(encoding="utf-8").lower()
    for leak in ("doi.org", "evidence", "source_url", "paper_id", "visual_review"):
        assert leak not in text, f"{leak} reappeared in styles.json"


def test_papers_remain_valid_bibliography():
    """papers.json is project-level evidence that the type set came from real
    top-journal reading. It is deliberately not joined to styles.json."""
    papers = json.loads((ROOT / "catalog/papers.json").read_text(encoding="utf-8-sig"))
    assert len({p["id"] for p in papers}) == len(papers)
    assert all(p["title"] and p["venue"] and p["year"] for p in papers)


# ── N5 group A: derived comparisons on a shared spectral grid ──────


def spectrum_profile():
    return analyze_file(ROOT / "examples" / "sample_spectrum.csv")


def rec_of(profile, kind):
    found = next((r for r in recommend(profile) if r.id == kind), None)
    assert found is not None, f"{kind} not offered"
    return found


def test_difference_subtracts_row_by_row():
    p = spectrum_profile()
    r = rec_of(p, "spectral_difference")
    ax = render(p, r).axes[0]
    a, b = r.encodings["y"]
    both = p.data[[a, b]].dropna()
    drawn_line = ax.lines[0].get_ydata()
    assert np.allclose(drawn_line, (both[a] - both[b]).to_numpy()[: len(drawn_line)])
    # axhline is the zero reference; it is the only line in axes coordinates
    references = [l for l in ax.lines if l.get_transform() is ax.get_yaxis_transform()]
    assert len(references) == 1 and references[0].get_ydata()[0] == 0.0


def test_ratio_masks_near_zero_denominator_and_says_so():
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": np.linspace(400.0, 900.0, 8),
                "T_sample": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "T_ref": [1.0, 1.0, 0.0, 1e-18, 1.0, 2.0, 2.0, 2.0],
            }
        )
    )
    r = rec_of(p, "spectral_ratio")
    ax = render(p, r).axes[0]
    assert "屏蔽" in ax.get_title()
    assert len(ax.lines[0].get_xdata()) == 6  # two rows dropped, not interpolated


def test_ratio_of_an_all_zero_denominator_refuses():
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": [400.0, 500.0, 600.0, 700.0],
                "T_a": [1.0, 2.0, 3.0, 4.0],
                "T_b": [0.0, 0.0, 0.0, 0.0],
            }
        )
    )
    r = Recommendation("spectral_ratio", "r", "medium", "t", {"x": "wavelength_nm", "y": ["T_a", "T_b"]})
    with pytest.raises(ValueError, match="分母"):
        render(p, r)


def test_difference_and_ratio_need_exactly_two_columns():
    p = spectrum_profile()
    r = rec_of(p, "spectral_difference")
    from dataclasses import replace as dc_replace

    with pytest.raises(ValueError, match="两个响应列"):
        render(p, dc_replace(r, encodings={**r.encodings, "y": ["device_A"]}))


def test_envelope_band_spans_min_to_max_and_labels_it_as_a_range():
    p = spectrum_profile()
    r = rec_of(p, "spectral_envelope")
    ax = render(p, r).axes[0]
    cols = r.encodings["y"]
    both = p.data[[p.encodings_x if False else "wavelength_nm"] + cols].dropna()
    band = ax.collections[0]
    verts = band.get_paths()[0].vertices
    assert verts[:, 1].min() == pytest.approx(both[cols].to_numpy().min())
    assert verts[:, 1].max() == pytest.approx(both[cols].to_numpy().max())
    assert "not SD" in ax.get_ylabel()


def test_energy_axis_adds_a_conjugate_top_axis():
    p = spectrum_profile()
    r = rec_of(p, "energy_axis")
    ax = render(p, r).axes[0]
    # secondary_xaxis does not appear in fig.axes; it is a child of the parent
    top = [c for c in ax.get_children() if type(c).__name__ == "SecondaryAxis"]
    assert top, "no secondary axis was added"
    assert "eV" in top[0].get_xlabel()
    # The secondary's own limits read low-to-high; the inversion shows up in
    # which end of the main axis each value corresponds to. 1170 nm is the left
    # edge and must map to the *higher* energy.
    left_nm, right_nm = ax.get_xlim()
    energies = sorted(top[0].get_xlim())
    assert energies[0] == pytest.approx(1239.841984 / right_nm, rel=1e-6)
    assert energies[1] == pytest.approx(1239.841984 / left_nm, rel=1e-6)


def test_energy_axis_is_not_offered_for_a_non_wavelength_axis():
    p = analyze_dataframe(
        pd.DataFrame({"frequency_thz": [1.0, 2.0, 3.0, 4.0], "a_signal": [1, 2, 3, 4],
                      "b_signal": [2, 3, 4, 5]})
    )
    assert "energy_axis" not in {r.id for r in recommend(p)}


def test_incomparable_columns_get_no_difference_or_ratio():
    """Subtracting a position from an intensity is arithmetic, not measurement."""
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    ids = {r.id for r in recommend(p)}
    assert "spectral_difference" not in ids
    assert "spectral_ratio" not in ids
    assert "spectral_envelope" not in ids


def test_series_labels_are_not_mistaken_for_units():
    """device_A / device_B share a stem and no unit, so they must pair."""
    p = spectrum_profile()
    assert "spectral_difference" in {r.id for r in recommend(p)}


def test_derived_comparisons_are_medium_not_high():
    """They rest on the user having chosen two columns that really are the same
    quantity, which the data cannot prove."""
    p = spectrum_profile()
    for r in recommend(p):
        if r.id in {"spectral_difference", "spectral_ratio", "spectral_envelope", "energy_axis"}:
            assert r.tier == "medium", r.id


def test_difference_refuses_a_log_axis():
    p = spectrum_profile()
    r = rec_of(p, "spectral_difference")
    with pytest.raises(ValueError, match="对数轴"):
        render(p, r, options={"ylog": True})


def test_every_new_figure_type_is_registered_and_renderable():
    p = spectrum_profile()
    kinds = {r.id for r in recommend(p)} & {
        "spectral_difference",
        "spectral_ratio",
        "spectral_envelope",
        "energy_axis",
    }
    assert kinds, "no derived-comparison type was offered at all"
    for kind in kinds:
        assert kind in FIGURE_TYPES, f"{kind} renders but is not registered"
        render(p, rec_of(p, kind))


# Types deliberately left without recipes, with the reason. These stay available
# to draw; they simply do not absorb the recipe-writing budget while the library
# is being widened. Removing one from here means someone wrote recipes for it.
SPARSE_BY_DESIGN = {
    "box",  # grouped box plots barely appear in this subfield's figure captions
    "density",
    "matrix_heatmap",
    "distribution",
}


def test_styles_cover_figure_types_except_documented_gaps():
    """A type with no drawing recipes is invisible in the figure library, so a
    gap has to be a decision someone wrote down, not an oversight."""
    styles = json.loads((ROOT / "catalog/styles.json").read_text(encoding="utf-8"))
    covered = {p for s in styles for p in s["patterns"]}
    uncovered = set(FIGURE_TYPES) - covered
    assert uncovered <= SPARSE_BY_DESIGN, f"undocumented recipe gaps: {sorted(uncovered)}"
    # every documented gap must still be a real gap
    assert not (SPARSE_BY_DESIGN - uncovered), (
        "a type in SPARSE_BY_DESIGN now has recipes; drop it from the allowlist"
    )


def test_every_figure_type_is_offered_somewhere():
    """A registered renderer nobody can reach is dead code, not a capability."""
    reachable = set()
    # matrix_heatmap only arises from a coordinate-free array, so the CSV sweep
    # alone would report it as dead when it is not.
    sources = sorted((ROOT / "examples").glob("*.csv")) + [ROOT / "examples" / "sample_matrix.npy"]
    for path in sources:
        reachable |= {r.id for r in recommend(analyze_file(path))}
    unreachable = set(FIGURE_TYPES) - reachable
    assert not unreachable, f"registered but never recommended: {sorted(unreachable)}"


# ── N5 group A: dual axis, derivative, peak tracking ───────────────


def test_derivative_recovers_a_known_slope():
    """d/dx of x^2 is 2x; a wrong axis spacing shows up immediately."""
    x = np.linspace(1.0, 5.0, 41)
    p = analyze_dataframe(
        pd.DataFrame({"wavelength_nm": x, "T_signal": x**2, "R_signal": x**2 * 1.001})
    )
    r = next(x for x in recommend(p) if x.id == "spectral_derivative")
    line = render(p, r).axes[0].lines[0]
    assert np.allclose(line.get_ydata(), 2 * x, atol=1e-8)


def test_derivative_is_not_offered_when_the_axis_repeats():
    """np.gradient divides by the coordinate spacing; repeated drive points make
    that zero, so the option must not appear at all."""
    p = analyze_file(ROOT / "examples" / "sample_replicates.csv")
    assert p.n_rows > 8
    ids = {r.id for r in recommend(p)}
    if "power_mW" in p.axis_columns and not p.repeated_x:
        assert "spectral_derivative" in ids
    repeated = analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": np.repeat(np.linspace(1400.0, 1700.0, 20), 2),
                "A_signal": np.arange(40, dtype=float),
                "B_signal": np.arange(40, dtype=float) + 1,
            }
        )
    )
    assert "spectral_derivative" not in {r.id for r in recommend(repeated)}


def test_peak_extraction_recovers_a_gaussian_centre_and_width():
    from optiplot.render import _peak_and_fwhm

    sigma = 25.0
    wave = np.linspace(1400.0, 1800.0, 201)
    value = np.exp(-((wave - 1580.0) ** 2) / (2 * sigma**2))
    found = _peak_and_fwhm(wave, value)
    assert found.is_max and found.position == pytest.approx(1580.0, abs=1.0)
    assert found.width == pytest.approx(2.3548 * sigma, rel=0.02)
    assert found.value == pytest.approx(1.0, abs=0.01)
    # the crossings travel with the number so a caller can draw the width
    assert found.left < found.position < found.right
    assert found.right - found.left == pytest.approx(found.width, rel=1e-9)


def test_a_dip_is_not_confused_with_the_noise_floor_of_a_flat_curve():
    """A reflectance dip on a flat pedestal has an interior maximum too: the
    tallest noise spike. Only the deeper turning point is the resonance."""
    from optiplot.render import _peak_and_fwhm

    wave = np.linspace(1300.0, 1800.0, 41)
    value = 1.0 - 0.62 * np.exp(-((wave - 1600.0) ** 2) / (2 * 55.0**2))
    value += np.sin(wave * 12.9898) * 0.004  # deterministic pseudo-noise
    found = _peak_and_fwhm(wave, value)
    assert found.is_max is False
    assert found.position == pytest.approx(1600.0, abs=6.0)
    assert found.width == pytest.approx(2.3548 * 55.0, rel=0.05)


def test_peak_on_the_scan_edge_is_reported_as_no_peak():
    """A monotonic rise to the last wavelength is a truncated scan, and calling
    that endpoint a resonance would report the instrument limit as physics."""
    from optiplot.render import _peak_and_fwhm

    wave = np.linspace(1400.0, 1800.0, 60)
    found = _peak_and_fwhm(wave, wave.copy())
    assert not found.found
    assert np.isnan(found.position) and np.isnan(found.width) and found.is_max is None


def test_peak_evolution_tracks_the_blue_shift():
    p = analyze_file(ROOT / "examples" / "sample_angle_resolved.csv")
    r = next(x for x in recommend(p) if x.id == "peak_evolution")
    assert r.encodings["param"] == "theta_deg"
    ax = render(p, r).axes[0]
    xs = ax.lines[0].get_xdata()
    ys = ax.lines[0].get_ydata()
    assert len(xs) >= 3 and ys[0] > ys[-1], "resonance should move to shorter wavelength"
    assert np.all(np.diff(ys) < 0), "peak track should be monotonic for a linear shift"
    # This example is an absorption dip, so reporting the tallest noise spike
    # instead of the turning point would leave a monotone-looking but wrong track.
    assert ys[0] == pytest.approx(1620.0, abs=12.5)
    second = [a for a in render(p, r).axes if a is not ax]
    assert second and second[0].lines, "FWHM trace missing from the twin axis"


def test_peak_evolution_counts_skipped_settings():
    from optiplot.render import _peak_and_fwhm

    p = analyze_file(ROOT / "examples" / "sample_angle_resolved.csv")
    r = next(x for x in recommend(p) if x.id == "peak_evolution")
    rows = p.data[[r.encodings["param"], r.encodings["wave"], r.encodings["value"]]].dropna()
    tracked = sum(
        np.isfinite(_peak_and_fwhm(g[r.encodings["wave"]], g[r.encodings["value"]])[0])
        for _, g in rows.groupby(r.encodings["param"])
    )
    ax = render(p, r).axes[0]
    assert len(ax.lines[0].get_xdata()) == tracked


def test_stacked_curves_lift_each_curve_clear_of_the_last():
    from optiplot.style import Style

    p = analyze_file(ROOT / "examples" / "sample_angle_resolved.csv")
    r = next(x for x in recommend(p) if x.id == "stacked_curves")
    ax = render(p, r).axes[0]
    spans = sorted(
        (float(np.min(l.get_ydata())), float(np.max(l.get_ydata())))
        for l in ax.lines
        if len(l.get_ydata())
    )
    assert len(spans) == 15
    for ceiling, next_floor in zip(spans, spans[1:]):
        assert ceiling[1] < next_floor[0], "stacked curves still overlap"
    assert len(ax.collections) == 15, "fill band missing"
    assert len(render(p, r, style=Style(stack_fill=False)).axes[0].collections) == 0
    wide = render(p, r, style=Style(stack_offset=3.0)).axes[0]
    assert wide.get_ylim()[1] > ax.get_ylim()[1], "stack_offset changed nothing"


def test_peak_annotation_labels_the_turning_point_it_found():
    p = analyze_file(ROOT / "examples" / "sample_spectrum.csv")
    r = next(x for x in recommend(p) if x.id == "peak_annotation")
    ax = render(p, r).axes[0]
    curves = [l for l in ax.lines if len(l.get_xdata()) > 5]
    spans = [l for l in ax.lines if len(l.get_xdata()) == 2]
    markers = [l for l in ax.lines if len(l.get_xdata()) == 1]
    assert len(curves) == 3 and len(markers) == 3, "a curve was marked or missed"
    assert len(spans) == 3, "each peak's half-maximum span should be drawn"
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 3 and all("FWHM" in t for t in texts)
    ceilings = [float(np.nanmax(l.get_ydata())) for l in curves]
    for marker in markers:
        # the marker has to sit on some curve's turning point, not near one
        assert min(abs(marker.get_ydata()[0] - top) for top in ceilings) < 0.05
        assert np.isfinite(marker.get_xdata()[0])


def test_a_curve_without_an_interior_peak_gets_no_annotation():
    """A monotonic rise to the last wavelength is a truncated scan; labelling its
    endpoint would report the instrument's range as a resonance."""
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": np.linspace(1200.0, 1600.0, 40),
                "absorbance": np.linspace(0.02, 0.9, 40),
            }
        )
    )
    assert "peak_annotation" not in {r.id for r in recommend(p)}
    assert "spectrum_lines" in {r.id for r in recommend(p)}


def test_a_grid_with_holes_is_a_heatmap_but_not_a_contour():
    """Blank cells are honest on a colour map and dishonest under contourf, which
    would interpolate a value across the gap and print it as measured."""
    p = analyze_file(ROOT / "examples" / "sample_angle_resolved.csv")
    recs = {r.id: r for r in recommend(p)}
    assert "contour" not in recs
    heat = recs["heatmap"]
    holes = int(p.data.isna().any(axis=1).sum())
    assert heat.tier == "medium" and str(holes) in heat.reason, (heat.tier, heat.reason)
    ax = render(p, heat).axes[0]
    assert ax.collections, "pcolormesh never drawn"


def test_a_scatter_cloud_is_not_mistaken_for_a_grid_with_holes():
    p = analyze_file(ROOT / "examples" / "sample_dense_scatter.csv")
    assert "heatmap" not in {r.id for r in recommend(p)}


def test_marginal_profiles_are_the_column_and_row_means():
    """The panels claim to be means; a profile that is actually a max or a slice
    would read as the same figure and say something entirely different."""
    x, y = np.meshgrid(np.linspace(-3.0, 3.0, 9), np.linspace(-2.0, 2.0, 7))
    z = np.exp(-(x**2 / 6.0 + y**2 / 2.0))
    p = analyze_dataframe(
        pd.DataFrame({"x_um": x.ravel(), "y_um": y.ravel(), "field_au": z.ravel()})
    )
    rec = next(r for r in recommend(p) if r.id == "heatmap_marginals")
    fig = render(p, rec)
    assert len(fig.axes) == 4, "map, two profiles and the colour bar"
    ax, top, right = fig.axes[0], fig.axes[1], fig.axes[2]
    assert top.lines[0].get_ydata() == pytest.approx(z.mean(axis=0), abs=1e-9)
    assert right.lines[0].get_xdata() == pytest.approx(z.mean(axis=1), abs=1e-9)
    assert top.lines[0].get_xdata() == pytest.approx(np.linspace(-3.0, 3.0, 9))
    # each caption sits on the axis carrying that profile's own values
    assert "y_um" in top.get_ylabel() and top.get_xlabel() == ""
    assert "x_um" in right.get_xlabel()


    # each caption sits on the axis carrying the profile's own values
    assert "y_um" in top.get_ylabel() and top.get_xlabel() == ""
    assert "x_um" in right.get_xlabel()


def test_marginal_panels_do_not_clip_the_map_they_share_axes_with():
    """A limit set on a shared child propagates back to the parent, and
    pcolormesh extends half a cell past the outermost coordinate, so the naive
    "match the panels to the data range" silently cut the border cells off."""
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, 21), np.linspace(-3.0, 3.0, 15))
    z = np.exp(-(x**2 / 4.0 + y**2) / 2.0)
    p = analyze_dataframe(
        pd.DataFrame({"x_um": x.ravel(), "y_um": y.ravel(), "field_au": z.ravel()})
    )
    plain = render(p, next(r for r in recommend(p) if r.id == "heatmap")).axes[0]
    with_panels = render(
        p, next(r for r in recommend(p) if r.id == "heatmap_marginals")
    ).axes[0]
    assert with_panels.get_xlim() == pytest.approx(plain.get_xlim())
    assert with_panels.get_ylim() == pytest.approx(plain.get_ylim())
    # the cells at the border must actually be inside the drawn frame
    assert with_panels.get_xlim()[0] < x.min() and with_panels.get_xlim()[1] > x.max()


def tilt_grid(hole=False):
    """A beam whose vertical extent grows with y, so the two normalisations give
    visibly different answers and a test can tell them apart. `hole` adds one
    genuinely unmeasured cell, which makes the grid partial."""
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, 11), np.linspace(-3.0, 3.0, 7))
    z = np.exp(-(x**2 / 4.0 + y**2) / 2.0) * (1.0 + 0.4 * y)
    frame = pd.DataFrame({"x_um": x.ravel(), "y_um": y.ravel(), "field_au": z.ravel()})
    if hole:
        frame.loc[0, "field_au"] = np.nan
    return analyze_dataframe(frame), z


def test_a_surface_and_its_contours_are_both_drawn():
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "heatmap_contours")
    ax = render(p, rec).axes[0]
    assert len(ax.collections) == 2, "expected a pcolormesh and a contour set"
    assert ax.collections[1].get_paths(), "contour lines missing"


# ── polar: dB scale and the multi-turn pair ────────────────────────
def antenna_profile(minimum=1e-6):
    a = np.linspace(0.0, 360.0, 721)
    power = np.abs(np.sinc(np.deg2rad(a) * 6)) ** 2 + minimum
    return analyze_dataframe(pd.DataFrame({"theta_deg": a, "far_field_W": power}))


def test_the_dB_reference_sits_at_zero_and_halves_at_minus_three():
    """0 dB is the peak by construction, and half the power must read -3.01 dB.
    A factor applied to the wrong kind of quantity shows up exactly here."""
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "theta_deg": [0.0, 90.0, 180.0, 270.0],
                "far_field_W": [1.0, 0.5, 0.25, 0.125],
            }
        )
    )
    rec = next(r for r in recommend(p) if r.id == "polar_db")
    ydata = render(p, rec, style=Style(db_floor=30.0)).axes[0].lines[0].get_ydata()
    assert ydata == pytest.approx([0.0, -3.0103, -6.0206, -9.0309], abs=1e-3)


def test_plotted_decibels_are_measured_values_transformed_not_a_fitted_curve():
    p = antenna_profile()
    rec = next(r for r in recommend(p) if r.id == "polar_db")
    ydata = render(p, rec).axes[0].lines[0].get_ydata()
    raw = p.data["far_field_W"].to_numpy()
    expected = 10.0 * np.log10(raw / raw.max())
    assert float(np.nanmax(ydata)) == pytest.approx(0.0, abs=1e-9)
    assert set(np.round(ydata, 9)) <= set(np.round(expected, 9))


def test_the_field_factor_doubles_the_decibel_span():
    """10·log10 for a power quantity, 20·log10 for a field amplitude; the column
    name cannot tell them apart, so the figure has to say which it used."""
    p = analyze_dataframe(
        pd.DataFrame(
            {
                "theta_deg": [0.0, 90.0, 180.0, 270.0],
                "far_field_W": [1.0, 0.5, 0.25, 0.125],
            }
        )
    )
    rec = next(r for r in recommend(p) if r.id == "polar_db")
    power = render(p, rec, style=Style(db_factor=10.0)).axes[0].lines[0].get_ydata()
    field = render(p, rec, style=Style(db_factor=20.0)).axes[0].lines[0].get_ydata()
    assert power == pytest.approx([0.0, -3.0103, -6.0206, -9.0309], abs=1e-3)
    assert field == pytest.approx(2.0 * power, abs=1e-6)
    assert "20·log10" in render(p, rec, style=Style(db_factor=20.0)).axes[0].get_title(loc="left")


def test_the_radial_depth_follows_the_data_and_says_what_it_clipped():
    shallow = analyze_dataframe(
        pd.DataFrame({"theta_deg": np.arange(0.0, 360.0, 5.0), "resp_W": 0.6 + 0.4 * np.cos(np.deg2rad(np.arange(0.0, 360.0, 5.0)))})
    )
    rec = next(r for r in recommend(shallow) if r.id == "polar_db")
    title = render(shallow, rec).axes[0].get_title(loc="left")
    assert "未画" not in title, "a 40 % modulation should fit inside the radius"
    assert "−10 dB" in title or "−5 dB" in title, title

    deep = antenna_profile()
    drec = next(r for r in recommend(deep) if r.id == "polar_db")
    dtitle = render(deep, drec, style=Style(db_floor=30.0)).axes[0].get_title(loc="left")
    assert "−30 dB" in dtitle and "未画" in dtitle, dtitle


def test_a_pattern_reaching_zero_is_counted_not_plotted_at_the_rim():
    a = np.arange(0.0, 360.0, 3.0)
    power = np.where(np.isclose(np.mod(a, 90.0), 0.0), 0.0, 1.0)
    p = analyze_dataframe(pd.DataFrame({"theta_deg": a, "far_field_W": power}))
    rec = next(r for r in recommend(p) if r.id == "polar_db")
    title = render(p, rec).axes[0].get_title(loc="left")
    assert "0/负值" in title and "4" in title, title
    assert not np.isnan(render(p, rec).axes[0].lines[0].get_ydata()).any()


def test_a_multi_turn_scan_gets_the_unwrapped_pair_and_a_single_turn_does_not():
    multi = analyze_file(ROOT / "examples" / "sample_multi_turn.csv")
    assert "polar_and_cartesian" in {r.id for r in recommend(multi)}
    single = analyze_file(ROOT / "examples" / "sample_polarization.csv")
    assert "polar_and_cartesian" not in {r.id for r in recommend(single)}


def test_both_panels_of_the_pair_draw_the_same_numbers():
    p = analyze_file(ROOT / "examples" / "sample_multi_turn.csv")
    rec = next(r for r in recommend(p) if r.id == "polar_and_cartesian")
    fig = render(p, rec)
    polar, cartesian = fig.axes
    assert np.array_equal(polar.lines[0].get_ydata(), cartesian.lines[0].get_ydata())
    assert cartesian.lines[0].get_xdata() == pytest.approx(p.data["rotator_angle_deg"].to_numpy())


def test_a_user_title_goes_above_the_scale_note_rather_than_over_it():
    """The note carries the dB reference and how many points were dropped; a
    title is decoration and must not delete it."""
    p = analyze_file(ROOT / "examples" / "sample_multi_turn.csv")
    rec = next(r for r in recommend(p) if r.id == "polar_db")
    bare = render(p, rec).axes[0].get_title(loc="left")
    titled = render(p, rec, options={"title": "旋转台扫描"}).axes[0].get_title(loc="left")
    assert titled.startswith("旋转台扫描") and bare in titled


# ── broken axis ────────────────────────────────────────────────────
def test_a_broken_axis_needs_a_real_hole_not_just_few_points():
    """Breaking an axis is only justified by a gap far wider than the sampling
    interval; a sparse but continuous sweep must stay on one continuous axis."""
    continuous = analyze_dataframe(
        pd.DataFrame(
            {
                "wavelength_nm": np.arange(400.0, 1500.0, 5.0),
                "response_A": np.sin(np.arange(400.0, 1500.0, 5.0) / 90.0),
            }
        )
    )
    assert "broken_spectrum" not in {r.id for r in recommend(continuous)}
    split = analyze_file(ROOT / "examples" / "sample_split_band.csv")
    assert "broken_spectrum" in {r.id for r in recommend(split)}


def test_the_break_is_named_and_the_two_panels_share_the_vertical_scale():
    split = analyze_file(ROOT / "examples" / "sample_split_band.csv")
    rec = next(r for r in recommend(split) if r.id == "broken_spectrum")
    assert rec.encodings["gap"] == [700.0, 1200.0]
    fig = render(split, rec)
    left, right = fig.axes
    assert left.get_ylim() == pytest.approx(right.get_ylim()), "peak heights must stay comparable"
    note = right.get_title(loc="left")
    assert "700" in note and "1200" in note and "无测量点" in note
    # the panel that carries the scale keeps its numbers; the other one does not
    assert any(t.get_text() for t in left.get_yticklabels())
    assert not any(t.get_visible() for t in right.get_yticklabels())


def test_the_facing_spines_are_removed_and_marked_as_broken():
    split = analyze_file(ROOT / "examples" / "sample_split_band.csv")
    rec = next(r for r in recommend(split) if r.id == "broken_spectrum")
    left, right = render(split, rec).axes
    assert not left.spines["right"].get_visible()
    assert not right.spines["left"].get_visible()
    # two slashes per panel edge, drawn in axes coordinates
    assert len(left.lines) >= 2 and len(right.lines) >= 2


def test_a_manual_x_range_is_refused_rather_than_applied_to_both_panels():
    """Each panel's range is the split itself; a shared manual range would leave
    one panel empty while still labelled as the other band."""
    from optiplot.style import Style

    split = analyze_file(ROOT / "examples" / "sample_split_band.csv")
    rec = next(r for r in recommend(split) if r.id == "broken_spectrum")
    with pytest.raises(ValueError, match="断轴图的横轴范围"):
        render(split, rec, style=Style(x_min=500.0, x_max=600.0))
    # a vertical range is fine: both panels already share it
    fig = render(split, rec, style=Style(y_min=0.0, y_max=1.2))
    assert fig.axes[0].get_ylim() == pytest.approx((0.0, 1.2))
    assert fig.axes[1].get_ylim() == pytest.approx((0.0, 1.2))


def test_nothing_is_drawn_inside_the_omitted_interval():
    """The point of the figure is that no measurement exists there; a line
    crossing the seam would invent one."""
    split = analyze_file(ROOT / "examples" / "sample_split_band.csv")
    rec = next(r for r in recommend(split) if r.id == "broken_spectrum")
    low_end, high_end = rec.encodings["gap"]
    for panel in render(split, rec).axes:
        for line in panel.get_lines():
            xs = np.asarray(line.get_xdata(), dtype=float)
            if xs.size:
                assert np.all((xs <= low_end) | (xs >= high_end))
    assert not any(
        low_end < x < high_end
        for panel in render(split, rec).axes
        for line in panel.get_lines()
        for x in np.asarray(line.get_xdata(), dtype=float)
    )


# ── Mueller matrix and Poincaré sphere ─────────────────────────────
def mueller_frame():
    # asymmetric on purpose, so the transpose deviation has something to report
    return pd.DataFrame([{f"m{i}{j}": 0.5 * (i - j) for i in range(4) for j in range(4)}])


def test_a_four_by_four_block_is_not_claimed_to_be_a_mueller_matrix():
    """Four rows of four numbers is an ordinary table; only names m00…m33 make
    the claim, and the recommender must not make it on the reader's behalf."""
    plain = analyze_dataframe(
        pd.DataFrame({f"c{i}": [1.0, 2.0, 3.0, 4.0] for i in range(4)})
    )
    assert "mueller_matrix" not in {r.id for r in recommend(plain)}
    named = analyze_dataframe(mueller_frame())
    assert "mueller_matrix" in {r.id for r in recommend(named)}


def test_the_mueller_grid_is_labelled_and_centred_on_zero():
    p = analyze_dataframe(mueller_frame())
    rec = next(r for r in recommend(p) if r.id == "mueller_matrix")
    ax = render(p, rec).axes[0]
    assert len(ax.texts) == 16, "every element should carry its own value"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["S0", "S1", "S2", "S3"]
    norm = ax.images[0].norm
    assert norm.vcenter == 0.0, "a signed matrix needs a diverging scale"
    assert norm.vmin == pytest.approx(-norm.vmax)


def test_the_mueller_transpose_deviation_is_reported_as_a_number_only():
    p = analyze_dataframe(mueller_frame())
    rec = next(r for r in recommend(p) if r.id == "mueller_matrix")
    title = render(p, rec).axes[0].get_title(loc="left")
    assert "与转置矩阵最大偏差" in title and "结合样品类型判断" in title, title


def test_stokes_columns_are_normalised_by_s0_not_projected_onto_the_sphere():
    """The radius is the degree of polarisation; pushing every point onto the
    surface would assert full polarisation the measurement does not show."""
    p = analyze_file(ROOT / "examples" / "sample_stokes.csv")
    rec = next(r for r in recommend(p) if r.id == "poincare_sphere")
    ax = render(p, rec).axes[0]
    trace = ax.lines[-1].get_data_3d()
    radius = np.sqrt(sum(component**2 for component in trace))
    assert radius.max() < 1.0, "this sample is partly polarised throughout"
    assert float(np.nanmax(radius)) == pytest.approx(0.95, abs=0.01)
    assert "0.95" in ax.get_title(loc="left")


def test_unusable_stokes_rows_are_dropped_before_the_trace():
    frame = pd.DataFrame(
        {
            "S0": [1.0, 2.0, 0.0, np.nan, 1.0],
            "S1": [0.5, 1.0, 0.1, 0.2, 0.3],
            "S2": [0.0, 0.0, 0.0, 0.0, 0.1],
            "S3": [0.0, 0.1, 0.0, 0.0, 0.1],
        }
    )
    p = analyze_dataframe(frame)
    rec = next(r for r in recommend(p) if r.id == "poincare_sphere")
    ax = render(p, rec).axes[0]
    assert ax.lines[-1].get_data_3d()[0].size == 3, "S0<=0 and the NaN row must not trace"


def test_an_inconsistent_stokes_set_is_counted_not_rescaled():
    frame = pd.DataFrame(
        {
            "S0": [1.0, 1.0, 1.0],
            "S1": [0.9, 0.8, 0.2],
            "S2": [0.5, 0.4, 0.1],
            "S3": [0.4, 0.3, 0.1],
        }
    )
    p = analyze_dataframe(frame)
    rec = next(r for r in recommend(p) if r.id == "poincare_sphere")
    ax = render(p, rec).axes[0]
    assert "球外" in ax.get_title(loc="left") and "不自洽" in ax.get_title(loc="left")
    trace = ax.lines[-1].get_data_3d()
    assert float(np.sqrt(trace[0][0] ** 2 + trace[1][0] ** 2 + trace[2][0] ** 2)) > 1.0


def test_the_sphere_view_angle_is_a_real_control():
    from optiplot.style import Style

    p = analyze_file(ROOT / "examples" / "sample_stokes.csv")
    rec = next(r for r in recommend(p) if r.id == "poincare_sphere")
    default = render(p, rec).axes[0]
    turned = render(p, rec, style=Style(view_elevation=70.0, view_azimuth=10.0)).axes[0]
    assert (default.elev, default.azim) != (turned.elev, turned.azim)
    assert (turned.elev, turned.azim) == (70.0, 10.0)


def test_normalisation_makes_each_line_zero_mean_unit_spread():
    p, _ = tilt_grid()
    rec = next(r for r in recommend(p) if r.id == "heatmap_normalized")
    mesh = render(p, rec, style=Style(missing_fill="none")).axes[0].collections[0]
    grid = np.asarray(mesh.get_array()).reshape(7, 11)
    rows = grid[~np.isnan(grid).any(axis=1)]  # the row holding the unmeasured cell
    assert rows.shape[0] >= 5
    assert rows.mean(axis=1) == pytest.approx(np.zeros(rows.shape[0]), abs=1e-9)
    assert rows.std(axis=1) == pytest.approx(np.ones(rows.shape[0]), abs=1e-9)


def test_the_two_normalisations_are_not_the_same_figure():
    p, _ = tilt_grid()
    rec = next(r for r in recommend(p) if r.id == "heatmap_normalized")
    per_y = render(p, rec, options={"normalize": "per_y"}).axes[-1].yaxis.label.get_text()
    per_x = render(p, rec, options={"normalize": "per_x"}).axes[-1].yaxis.label.get_text()
    assert "每个 y_um 处沿 x_um" in per_y
    assert "每个 x_um 处沿 y_um" in per_x
    assert per_y != per_x, "the colour bar must name the convention actually used"


def test_a_flat_line_normalises_to_nothing_not_to_zero():
    from optiplot.render import _normalised_surface

    flat = np.ones((4, 5))
    assert np.isnan(_normalised_surface(flat, "per_y")).all()
    mixed = np.vstack([np.linspace(1.0, 2.0, 5), np.full(5, 7.0)])
    out = _normalised_surface(mixed, "per_y")
    assert np.isnan(out[1]).all() and np.isfinite(out[0]).all()


def test_an_unknown_normalisation_is_refused():
    p, _ = tilt_grid()
    rec = next(r for r in recommend(p) if r.id == "heatmap_normalized")
    with pytest.raises(ValueError, match="normalize 需是"):
        render(p, rec, options={"normalize": "per_z"})


def test_a_threshold_hides_cells_in_a_colour_that_is_not_the_missing_colour():
    """A grey square can mean "measured, small" or "never measured" and the two
    say opposite things about the sample."""
    p, _ = tilt_grid(hole=True)
    rec = next(r for r in recommend(p) if r.id == "heatmap")
    fig = render(
        p, rec, style=Style(mask_below=0.3, missing_fill="dimgrey", masked_fill="white")
    )
    mesh, veil = fig.axes[0].collections[0], fig.axes[0].collections[1]
    missing = to_hex(mesh.get_cmap()(np.ma.masked))
    hidden = to_hex(veil.get_cmap()(0.5))
    assert missing == "#696969" and hidden == "#ffffff"
    # the unmeasured cell is not counted among the hidden ones
    grid = np.array(p.data["field_au"], dtype=float).reshape(7, 11)
    expect = int(np.nansum(np.abs(grid) < 0.3))
    stated = int(re.search(r"已屏蔽 (\d+) 格", fig.axes[-1].yaxis.label.get_text())[1])
    assert stated == expect, (stated, expect)
    assert 0 < stated < grid.size, "some but not all cells hidden"


def test_the_threshold_is_judged_on_measured_values_not_normalised_ones():
    p, z = tilt_grid()
    rec = next(r for r in recommend(p) if r.id == "heatmap_normalized")
    fig = render(p, rec, options={"normalize": "per_y"}, style=Style(mask_below=0.3))
    label = fig.axes[-1].yaxis.label.get_text()
    raw_hidden = int((np.abs(z) < 0.3).sum())
    assert str(raw_hidden) in label, (label, raw_hidden)


def test_dual_axis_gives_each_series_its_own_scale():
    p = analyze_file(ROOT / "examples" / "sample_liv_sweep.csv")
    r = next(x for x in recommend(p) if x.id == "dual_axis")
    ax = render(p, r).axes[0]
    second = [a for a in ax.figure.axes if a is not ax]
    assert second, "twinx axis missing"
    assert ax.get_ylabel() != second[0].get_ylabel()
    assert ax.get_ylim() != second[0].get_ylim()
    # the two axis labels must be tinted to their own series
    assert ax.yaxis.label.get_color() != second[0].yaxis.label.get_color()


def test_dual_axis_is_not_offered_for_same_unit_columns():
    """device_A and device_B belong on one axis; splitting them across two is how
    a dual-axis plot invents agreement."""
    p = analyze_file(ROOT / "examples" / "sample_spectrum.csv")
    assert "dual_axis" not in {r.id for r in recommend(p)}


def test_dual_axis_is_not_offered_across_frame_coordinates():
    """x_um and intensity_au differ in unit, but the transverse coordinate of a
    beam map is not a second response -- it is the axis the map is read on."""
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    assert "dual_axis" not in {r.id for r in recommend(p)}


def test_a_family_wider_than_the_palette_gets_distinct_colours():
    """Fifteen angles of one coating are neither a single curve nor replicates, so
    handing two of them the same colour would misread as either."""
    p = analyze_file(ROOT / "examples" / "sample_angle_resolved.csv")
    r = next(x for x in recommend(p) if x.id == "spectrum_lines")
    assert r.encodings["group"] == "theta_deg"
    fig = render(p, r)
    colours = {to_hex(line.get_color()) for line in fig.axes[0].lines}
    colours.discard("#7A94AB")  # the zero guide line, not a series
    assert len(colours) == 15, sorted(colours)
    # and a small group count still gets the curated palette, untouched
    devices = analyze_file(ROOT / "examples" / "sample_devices.csv")
    box = next(x for x in recommend(devices) if x.id == "box")
    assert box.encodings["group"] == "device"
    from optiplot.render import _group_colours
    from optiplot.style import Style

    assert _group_colours(Style(), 3) == Style().colors[:3]


def test_new_types_are_medium_tier():
    for name, sample in [
        ("dual_axis", "sample_liv_sweep"),
        ("spectral_derivative", "sample_spectrum"),
        ("peak_evolution", "sample_angle_resolved"),
    ]:
        p = analyze_file(ROOT / "examples" / f"{sample}.csv")
        found = next((x for x in recommend(p) if x.id == name), None)
        assert found is not None, f"{name} not offered on {sample}"
        assert found.tier == "medium", name
        assert "请" in found.reason or "不" in found.reason, f"{name} states no caveat"
