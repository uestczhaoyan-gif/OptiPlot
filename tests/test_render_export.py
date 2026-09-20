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
from optiplot.render import render
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
    r = Recommendation("errorbar", "error", 1, "test", {"x": "x", "y": "y", "error": "sd"})
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


def test_catalog_links_and_counts():
    papers = json.loads((ROOT / "catalog/papers.json").read_text(encoding="utf-8-sig"))
    cases = json.loads((ROOT / "catalog/cases.json").read_text(encoding="utf-8-sig"))
    assert len({p["id"] for p in papers}) == len(papers)
    assert len({c["id"] for c in cases}) == len(cases)
    ids = {p["id"] for p in papers}
    assert all(
        c["paper_id"] in ids and c["source_url"].startswith("https://") and c["recipe"]
        for c in cases
    )
    for p in papers:
        assert p["case_count"] == sum(c["paper_id"] == p["id"] for c in cases)
