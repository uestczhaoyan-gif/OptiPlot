"""Command-line access to the same profiler and renderer used by the GUI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .core import analyze_file, recommend, MAX_CANDIDATES
from .render import render, FIT_CHOICES, OptionNotApplicable
from .export import export_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OptiPlot: inspect local data and export recommended figures"
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("optiplot_output"))
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Export a reproducible ZIP for the highest-ranked choice",
    )
    parser.add_argument(
        "--fit",
        default="none",
        choices=FIT_CHOICES,
        help="Overlay a named model and its residual panel; never applied on its own",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Figure option, value read as JSON when it parses: "
        "--set reference=25 --set relative=true --set norm_target=area",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.top <= MAX_CANDIDATES:
        parser.error(f"--top must be between 1 and {MAX_CANDIDATES}")
    overrides = {}
    for item in args.overrides or []:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            parser.error(f"--set needs KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    try:
        sheet = int(args.sheet) if args.sheet.isdecimal() else args.sheet
        p = analyze_file(args.data, header=not args.no_header, sheet_name=sheet)
        rs = recommend(p)
        args.out.mkdir(parents=True, exist_ok=True)
        result = {
            "file": args.data.name,
            "rows": p.n_rows,
            "columns": p.columns,
            "notes": p.notes,
            "recommendations": [asdict(r) for r in rs],
        }
        (args.out / "recommendations.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        options = {"size": "double", "fit": args.fit, **overrides}
        drawn, skipped = 0, []
        for r in rs[: args.top]:
            try:
                for ext in ["png", "svg", "pdf"]:
                    render(p, r, args.out / f"{r.id}.{ext}", options=options).clear()
                drawn += 1
            except OptionNotApplicable as exc:
                # One option set covers a batch of different figures, so a type
                # that has no path for it is passed over and named, not fatal.
                skipped.append(f"  {r.id}: {exc}")
        if args.bundle and rs:
            export_bundle(p, rs[0], args.out / "reproducible.zip", options=options)
        print(f"{drawn} figures exported to {args.out.resolve()}")
        for line in skipped:
            print(line)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"OptiPlot: {exc}\n")


if __name__ == "__main__":
    main()
