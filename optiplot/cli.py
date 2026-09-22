"""Command-line access to the same profiler and renderer used by the GUI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .core import analyze_file, recommend
from .render import render, FIT_CHOICES
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
    args = parser.parse_args(argv)
    if not 1 <= args.top <= 8:
        parser.error("--top must be between 1 and 8")
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
        options = {"size": "double", "fit": args.fit}
        for r in rs[: args.top]:
            for ext in ["png", "svg", "pdf"]:
                render(p, r, args.out / f"{r.id}.{ext}", options=options).clear()
        if args.bundle and rs:
            export_bundle(p, rs[0], args.out / "reproducible.zip", options=options)
        print(f"{len(rs[:args.top])} figures exported to {args.out.resolve()}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"OptiPlot: {exc}\n")


if __name__ == "__main__":
    main()
