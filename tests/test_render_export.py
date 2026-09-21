"""End-to-end plot, export and data-integrity regression checks."""

from pathlib import Path
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest
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
