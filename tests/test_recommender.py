"""Behavioral regressions for conservative scientific recommendations."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from optiplot import analyze_dataframe, analyze_file, recommend
from optiplot.core import TIER_LABELS, TIER_ORDER


class RecommenderTestCase(unittest.TestCase):
    def suggestions(self, data):
        profile = analyze_dataframe(pd.DataFrame(data))
        return profile, {rec.id: rec for rec in recommend(profile)}

    def test_spectrum_axis_is_independent_of_numeric_column_order(self):
        p, recs = self.suggestions(
            {"signal": [1, 4, 2, 3], "wavelength_nm": [400, 500, 600, 700], "other": [2, 3, 1, 2]}
        )
        line = recs["spectrum_lines"]
        self.assertEqual(line.encodings["x"], "wavelength_nm")
        self.assertEqual(line.encodings["y"], ["signal", "other"])
        self.assertNotIn(line.encodings["x"], line.encodings["y"])

    def test_random_columns_do_not_get_semantic_axes_or_lines(self):
        p, recs = self.suggestions(
            {"efficiency": [3, 1, 4, 2], "pixel": [8, 2, 5, 3], "intensity": [9, 4, 7, 5]}
        )
        self.assertEqual(p.axis_columns, [])
        self.assertFalse(p.grid_like)
        self.assertNotIn("spectrum_lines", recs)
        self.assertIn("scatter_fit", recs)

    def test_default_scatter_never_requests_fitting(self):
        _, recs = self.suggestions({"x": [1, 2, 3, 4], "signal": [4, 1, 3, 2]})
        self.assertNotIn("fit", recs["scatter_fit"].encodings)
        self.assertIn("默认不拟合", recs["scatter_fit"].reason)

    def test_infinities_and_missing_values_are_reported_without_mutating_input(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "signal": [1, np.inf, np.nan, 4]})
        p = analyze_dataframe(df)
        self.assertTrue(np.isinf(df.loc[1, "signal"]))
        self.assertTrue(pd.isna(p.data.loc[1, "signal"]))
        self.assertEqual(p.missing_counts["signal"], 2)
        self.assertTrue(any("无穷" in note for note in p.notes))
        self.assertNotIn("spectrum_lines", {r.id for r in recommend(p)})

    def test_empty_or_entirely_missing_data_is_rejected(self):
        for df in [pd.DataFrame(), pd.DataFrame({"signal": [np.nan, np.inf, -np.inf]})]:
            with self.subTest(df=df), self.assertRaises(ValueError):
                analyze_dataframe(df)

    def test_string_only_data_has_explicit_table_fallback(self):
        p, recs = self.suggestions({"label": ["reference", "sample"], "status": ["ok", "ok"]})
        self.assertEqual(set(recs), {"table"})
        self.assertTrue(any("未发现有效数值列" in note for note in p.notes))

    def test_single_row_does_not_suggest_statistics(self):
        p, recs = self.suggestions({"time_s": [1], "signal": [3.2], "signal_sd": [0.1]})
        self.assertEqual(set(recs), {"table"})
        self.assertTrue(any("只有一行" in note for note in p.notes))

    def test_mixed_numeric_and_categorical_column_is_preserved(self):
        p, recs = self.suggestions(
            {"condition": ["control", "1", "2", "control"], "signal": [1, 2, 3, 4]}
        )
        self.assertIn("condition", p.categorical_columns)
        self.assertEqual(p.data.loc[0, "condition"], "control")
        self.assertNotIn("scatter_fit", recs)

    def test_identifiers_constants_and_error_columns_are_not_y_curves(self):
        p, recs = self.suggestions(
            {
                "sample_id": [1, 2, 3, 4],
                "baseline": [0, 0, 0, 0],
                "time_s": [1, 2, 3, 4],
                "signal": [5, 2, 4, 3],
                "signal_sd": [0.1, 0.2, 0.3, 0.2],
            }
        )
        self.assertEqual(recs["spectrum_lines"].encodings["y"], ["signal"])
        self.assertEqual(recs["scatter_fit"].encodings["x"], "time_s")
        self.assertIn("sample_id", p.id_columns)
        self.assertIn("baseline", p.constant_columns)

    def test_full_grid_uses_correct_named_coordinates_after_reordering(self):
        x, y = np.meshgrid([0, 1, 2], [10, 20, 30])
        p, recs = self.suggestions(
            {"intensity": (x + y).ravel(), "y_um": y.ravel(), "x_um": x.ravel()}
        )
        self.assertTrue(p.grid_like)
        for kind in ("heatmap", "contour"):
            self.assertEqual(recs[kind].encodings, {"x": "x_um", "y": "y_um", "z": "intensity"})
        self.assertNotIn("errorbar", recs)
        self.assertNotIn("spectrum_lines", recs)

    def test_duplicate_coordinate_pair_does_not_fake_a_complete_grid(self):
        x, y = np.meshgrid([0, 1, 2], [0, 1, 2])
        data = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "signal": np.arange(9)})
        data.loc[8, ["x", "y"]] = data.loc[0, ["x", "y"]]
        p = analyze_dataframe(data)
        self.assertFalse(p.grid_like)
        self.assertNotIn("heatmap", {r.id for r in recommend(p)})

    def test_two_by_two_or_two_column_grid_is_not_a_surface(self):
        for side, with_z in [(2, True), (3, False)]:
            x, y = np.meshgrid(np.arange(side), np.arange(side))
            frame = {"x": x.ravel(), "y": y.ravel()}
            if with_z:
                frame["signal"] = np.arange(side**2)
            with self.subTest(side=side):
                p, recs = self.suggestions(frame)
                self.assertFalse(p.grid_like)
                self.assertNotIn("contour", recs)

    def test_missing_grid_measurement_prevents_surface_recommendation(self):
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        z = np.arange(9, dtype=float)
        z[-1] = np.nan
        p, recs = self.suggestions({"x": x.ravel(), "y": y.ravel(), "signal": z})
        self.assertFalse(p.grid_like)
        self.assertNotIn("heatmap", recs)

    def test_polar_preserves_explicit_degrees_and_radians(self):
        for name, angles, unit in [
            ("angle_deg", [0, 90, 180, 270], "deg"),
            ("theta_rad", [0, np.pi / 2, np.pi, 1.5 * np.pi], "rad"),
        ]:
            with self.subTest(unit=unit):
                p, recs = self.suggestions({"response": [1, 0.5, 0.9, 0.4], name: angles})
                self.assertEqual(recs["polar"].encodings["theta"], name)
                self.assertEqual(recs["polar"].encodings["r"], "response")
                self.assertEqual(recs["polar"].encodings["angle_unit"], unit)
                self.assertEqual(recommend(p)[0].id, "polar")

    def test_polar_assumed_units_are_disclosed_and_negative_radius_is_not_used(self):
        p, recs = self.suggestions({"angle": [0, 1, 2, 3], "response": [1, 2, 3, 2]})
        self.assertTrue(any("未注明角度单位" in note for note in p.notes))
        _, negative = self.suggestions(
            {"angle_deg": [0, 90, 180, 270], "response_db": [-3, -10, -6, -2]}
        )
        self.assertNotIn("polar", negative)

    def test_actual_replicates_support_sample_sd(self):
        _, recs = self.suggestions({"time_s": [0, 0, 1, 1, 2, 2], "signal": [1, 2, 3, 5, 4, 6]})
        error = recs["errorbar"]
        self.assertEqual(error.encodings["y"], "signal")
        self.assertEqual(error.encodings["error_kind"], "sd")
        self.assertTrue(error.encodings["replicates"])
        self.assertNotIn("spectrum_lines", recs)

    def test_cross_group_repeats_do_not_count_as_replicates(self):
        _, recs = self.suggestions(
            {
                "time_s": [0, 1, 2, 0, 1, 2],
                "signal": [1, 3, 2, 4, 2, 5],
                "condition": ["a"] * 3 + ["b"] * 3,
            }
        )
        self.assertNotIn("errorbar", recs)
        self.assertEqual(recs["spectrum_lines"].encodings["group"], "condition")

    def test_explicit_sem_is_not_reinterpreted_as_sd(self):
        _, recs = self.suggestions(
            {"time_s": [1, 2, 3, 4], "signal": [2, 3, 4, 5], "signal_sem": [0.1, 0.1, 0.2, 0.2]}
        )
        enc = recs["errorbar"].encodings
        self.assertEqual(enc["error_kind"], "sem")
        self.assertEqual(enc["error"], "signal_sem")
        self.assertNotIn("replicates", enc)

    def test_negative_uncertainty_is_not_recommended(self):
        _, recs = self.suggestions(
            {"time_s": [1, 2, 3, 4], "signal": [2, 3, 4, 5], "signal_sd": [0.1, -0.1, 0.2, 0.2]}
        )
        self.assertNotIn("errorbar", recs)

    def test_dense_data_and_grouped_observations_get_specific_choices(self):
        rng = np.random.default_rng(12)
        _, dense = self.suggestions({"a": rng.normal(size=400), "b": rng.normal(size=400)})
        self.assertIn("density", dense)
        _, grouped = self.suggestions(
            {"condition": ["a", "a", "b", "b"], "measurement": [1, 3, 4, 5]}
        )
        self.assertEqual(grouped["box"].encodings, {"group": "condition", "value": "measurement"})

    def test_flow_requires_explicit_source_target_columns(self):
        _, recs = self.suggestions({"source": ["laser", "lens"], "target": ["lens", "detector"]})
        self.assertEqual(recs["flow"].encodings, {"source": "source", "target": "target"})
        self.assertNotIn("scatter_fit", recs)

    def test_no_rec_uses_unknown_columns_and_choices_are_ranked_bounded(self):
        p, recs = self.suggestions(
            {
                "time_s": np.repeat(np.arange(100), 4),
                "value": np.sin(np.arange(400)),
                "other": np.cos(np.arange(400)),
                "condition": ["a", "a", "b", "b"] * 100,
            }
        )
        suggestions = recommend(p)
        self.assertLessEqual(len(suggestions), 8)
        keys = [(TIER_ORDER.index(r.tier), r.rank) for r in suggestions]
        self.assertEqual(keys, sorted(keys), "recommendations must order by tier, then rank")
        for rec in suggestions:
            for key in (
                "x",
                "y",
                "z",
                "theta",
                "r",
                "value",
                "group",
                "source",
                "target",
                "error",
                "columns",
                "param",
                "wave",
                "right",
            ):
                if key in rec.encodings:
                    value = rec.encodings[key]
                    for col in value if isinstance(value, list) else [value]:
                        self.assertIn(col, p.columns)

    def test_trailing_letters_are_units_only_when_the_name_says_so(self):
        """`forward_voltage_V` carries volts; the A in `device_A` labels a series."""
        from optiplot.core import _distinct_quantity, _unit_of

        self.assertEqual(_unit_of("forward_voltage_V"), "v")
        self.assertEqual(_unit_of("drain_current_A"), "a")
        self.assertIsNone(_unit_of("device_A"))
        self.assertIsNone(_unit_of("channel_B"))
        self.assertTrue(_distinct_quantity("forward_voltage_V", "optical_power_mW"))
        self.assertFalse(_distinct_quantity("device_A", "device_B"))
        self.assertFalse(_distinct_quantity("x_um", "intensity_au"))


class FileInputTestCase(unittest.TestCase):
    def test_delimited_encodings_and_no_header_preserve_first_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            for suffix, delimiter in [("csv", ","), ("tsv", "\t"), ("txt", ";")]:
                with self.subTest(format=suffix):
                    path = Path(temp) / f"sample.{suffix}"
                    path.write_text(
                        delimiter.join(["波长", "信号"])
                        + "\n"
                        + delimiter.join(["400", "1"])
                        + "\n"
                        + delimiter.join(["500", "2"]),
                        encoding="gb18030",
                    )
                    p = analyze_file(path)
                    self.assertEqual(p.numeric_columns, ["波长", "信号"])
                    self.assertEqual(p.n_rows, 2)
            path = Path(temp) / "no_header.txt"
            path.write_text("1 2\n3 4\n5 6", encoding="utf-8")
            p = analyze_file(path, header=False)
            self.assertEqual(p.n_rows, 3)
            self.assertEqual(p.data.iloc[0].tolist(), [1, 2])

    def test_malformed_empty_header_only_and_unsupported_files_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            for i, text in enumerate(["", "x,y\n", "x,y\n1,2\n3,4,5", "x,y\n1\n2,3", 'x,y\n1,"2']):
                path = Path(temp) / f"bad_{i}.csv"
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    analyze_file(path)
            with self.assertRaises(ValueError):
                analyze_file(Path(temp) / "image.png")

    def test_npy_mat_and_excel_numeric_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            array = np.arange(12).reshape(4, 3)
            np.save(root / "matrix.npy", array)
            self.assertEqual(analyze_file(root / "matrix.npy").data.shape, (4, 3))
            np.save(root / "volume.npy", np.zeros((2, 2, 2)))
            with self.assertRaises(ValueError):
                analyze_file(root / "volume.npy")
            savemat(root / "vectors.mat", {"wavelength_nm": np.arange(4), "signal": [1, 3, 2, 4]})
            p = analyze_file(root / "vectors.mat")
            self.assertEqual(p.n_rows, 4)
            self.assertEqual(p.numeric_columns, ["wavelength_nm", "signal"])
            self.assertIn("spectrum_lines", {r.id for r in recommend(p)})
            pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_excel(
                root / "sample.xlsx", index=False, sheet_name="Experiment"
            )
            p = analyze_file(root / "sample.xlsx", sheet_name="Experiment")
            self.assertEqual(p.n_rows, 2)
            self.assertEqual(p.columns, ["x", "y"])


    def test_density_is_not_offered_for_a_measurement_grid(self):
        """Hexbin counts samples per region; on a complete grid each cell holds
        one sample, so the density map is uniform and says nothing."""
        grid = pd.DataFrame(
            {
                "x_um": np.tile(np.arange(40, dtype=float), 40),
                "y_um": np.repeat(np.arange(40, dtype=float), 40),
                "intensity_au": np.random.default_rng(3).normal(size=1600),
            }
        )
        p = analyze_dataframe(grid)
        self.assertTrue(p.grid_like)
        self.assertNotIn("density", {r.id for r in recommend(p)})
        self.assertIn("heatmap", {r.id for r in recommend(p)})

        cloud = pd.DataFrame(
            {
                "power_mW": np.random.default_rng(4).uniform(0, 5, 2000),
                "noise_dB": np.random.default_rng(5).normal(0, 1, 2000),
            }
        )
        c = analyze_dataframe(cloud)
        self.assertFalse(c.grid_like)
        self.assertIn("density", {r.id for r in recommend(c)})

    def test_no_numeric_score_can_creep_back(self):
        """Tiers replaced numeric scores on purpose: a 1-point gap reads as
        confidence the ranking never had."""
        for path in sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.csv")):
            for rec in recommend(analyze_file(path)):
                self.assertIn(rec.tier, TIER_ORDER)
                self.assertEqual(rec.tier_label, TIER_LABELS[rec.tier])
                self.assertFalse(hasattr(rec, "score"), f"{rec.id} still exposes a score")


if __name__ == "__main__":
    unittest.main()
