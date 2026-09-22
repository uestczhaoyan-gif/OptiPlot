"""End-to-end plot, export and data-integrity regression checks."""

from pathlib import Path
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from optiplot import analyze_file, analyze_dataframe, recommend, Recommendation
from optiplot.render import render, FIGURE_TYPES
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
    # the shared axes must still line up after the colour bar takes its column
    assert ax.get_ylim() == pytest.approx(right.get_ylim())
    assert ax.get_xlim() == pytest.approx(top.get_xlim())


def test_a_surface_and_its_contours_are_both_drawn():
    p = analyze_file(ROOT / "examples" / "sample_beam_map.csv")
    rec = next(r for r in recommend(p) if r.id == "heatmap_contours")
    ax = render(p, rec).axes[0]
    assert len(ax.collections) == 2, "expected a pcolormesh and a contour set"
    assert ax.collections[1].get_paths(), "contour lines missing"


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
