#!/usr/bin/env python3
"""Fetch a bounded, cached OpenAlex metadata candidate set (standard library only).

This does not download articles or images, infer journal quartiles, or certify a
figure caption. Candidates stay separate from catalog/papers.json and cases.json.
API documentation: https://developers.openalex.org/ ; https://docs.openalex.org/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://api.openalex.org/works"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = [
    "metasurface photodetector",
    "computational spectroscopy",
    "computational imaging",
]


def cache_path(cache_dir: Path, params: dict) -> Path:
    """Exclude credentials from filenames, logs, and cached metadata."""
    public = {k: v for k, v in params.items() if k != "api_key"}
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    return cache_dir / f"{digest}.json"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def get_page(params: dict, args: argparse.Namespace) -> tuple[dict, bool]:
    cached = cache_path(args.cache_dir, params)
    if cached.exists() and not args.refresh:
        data = json.loads(cached.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RuntimeError(f"Invalid cached response: {cached.name}; use --refresh")
        return data, True
    if args.offline:
        raise RuntimeError(
            f"Missing cached page {cached.name}; offline mode never requests the network"
        )
    request = Request(
        API + "?" + urlencode(params),
        headers={
            "User-Agent": "OptiPlot-literature-catalog/0.1 (bounded metadata research)",
            "Accept": "application/json",
        },
    )
    for attempt in range(args.retries + 1):
        retry_after = None
        try:
            with urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise RuntimeError("OpenAlex returned an unexpected JSON schema")
            atomic_json(cached, data)
            return data, False
        except HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError(
                    f"OpenAlex HTTP {error.code}: access was refused. Supply a valid "
                    "OPENALEX_API_KEY and check the current account/API requirements."
                ) from None
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"OpenAlex HTTP {error.code}; check query/filter syntax"
                ) from None
            try:
                retry_after = float(error.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                pass
            if retry_after is not None and retry_after > 60:
                raise RuntimeError(
                    f"OpenAlex requested a {retry_after:g}-second cooldown; rerun later to use cached progress"
                ) from None
            failure = f"HTTP {error.code}"
        except (URLError, TimeoutError, OSError):
            # Do not print an exception that may contain a credential-bearing URL.
            failure = "network/timeout error"
        if attempt >= args.retries:
            raise RuntimeError(f"OpenAlex request failed after {attempt + 1} attempts: {failure}")
        pause = max(args.delay, retry_after if retry_after is not None else min(30.0, 2.0**attempt))
        print(f"Retrying after {failure} in {pause:g}s", file=sys.stderr)
        time.sleep(pause)
    raise RuntimeError("unreachable retry state")


def normalize(work: dict, query: str) -> dict:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    access = work.get("open_access") or {}
    doi_url = work.get("doi")
    return {
        "id": work.get("id"),
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "source_id": source.get("id"),
        "doi": doi_url.removeprefix("https://doi.org/") if isinstance(doi_url, str) else None,
        "url": doi_url or location.get("landing_page_url") or work.get("id"),
        "kind": work.get("type"),
        "quartile": {"system": None, "year": None, "value": None, "status": "unverified"},
        "evidence": "openalex_metadata_only; figure captions not reviewed",
        "license_status": "metadata_only; figure reuse rights not reviewed",
        "reported_primary_location_license": location.get("license"),
        "open_access": {
            "is_oa": access.get("is_oa"),
            "oa_status": access.get("oa_status"),
            "oa_url": access.get("oa_url"),
        },
        "matched_queries": [query],
        "review_status": "candidate",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", action="append", help="Repeat for multiple search topics")
    p.add_argument("--year-from", type=int, default=2018)
    p.add_argument("--year-to", type=int, default=date.today().year)
    p.add_argument(
        "--filter",
        default="",
        help="Additional OpenAlex filter, e.g. is_oa:true or primary_location.source.id:S...",
    )
    p.add_argument("--per-page", type=int, default=50, help="Records per API page, 1–200")
    p.add_argument("--max-pages", type=int, default=2, help="Maximum pages per query")
    p.add_argument("--max-records", type=int, default=300, help="Global unique-record bound")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "openalex")
    p.add_argument("--output", type=Path, default=ROOT / "catalog" / "openalex_candidates.json")
    p.add_argument(
        "--api-key",
        default=None,
        help="Optional; defaults to OPENALEX_API_KEY environment variable",
    )
    p.add_argument("--mailto", default=None, help="Optional contact email supplied to OpenAlex")
    p.add_argument(
        "--offline", action="store_true", help="Replay cached pages without network access"
    )
    p.add_argument(
        "--refresh", action="store_true", help="Refresh pages instead of using the cache"
    )
    p.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between requests")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=3)
    return p


def collect(args: argparse.Namespace) -> dict:
    queries = args.query or DEFAULT_QUERIES
    records: dict[str, dict] = {}
    stats = {"pages": 0, "cache_hits": 0, "api_requests_succeeded": 0}
    api_key = args.api_key or os.environ.get("OPENALEX_API_KEY")
    filters = (
        f"from_publication_date:{args.year_from}-01-01,to_publication_date:{args.year_to}-12-31"
    )
    if args.filter:
        filters += "," + args.filter
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "queries": queries,
        "filters": filters,
        "limits": {"max_pages_per_query": args.max_pages, "max_records": args.max_records},
        "statistics": stats,
        "warning": "Candidate metadata only. No article images, figure verification, or quartile classification has been performed.",
        "papers": [],
    }
    try:
        for query in queries:
            cursor = "*"
            seen_cursors: set[str] = set()
            for _ in range(args.max_pages):
                if len(records) >= args.max_records or cursor in seen_cursors:
                    break
                seen_cursors.add(cursor)
                params = {
                    "search": query,
                    "filter": filters,
                    "per-page": args.per_page,
                    "cursor": cursor,
                }
                if api_key:
                    params["api_key"] = api_key
                if args.mailto:
                    params["mailto"] = args.mailto
                data, cache_hit = get_page(params, args)
                stats["pages"] += 1
                stats["cache_hits" if cache_hit else "api_requests_succeeded"] += 1
                for work in data["results"]:
                    if not isinstance(work, dict):
                        continue
                    key = work.get("doi") or work.get("id")
                    if not key:
                        continue
                    if key in records:
                        if query not in records[key]["matched_queries"]:
                            records[key]["matched_queries"].append(query)
                    elif len(records) < args.max_records:
                        records[key] = normalize(work, query)
                payload["papers"] = list(records.values())
                atomic_json(args.output, payload)
                print(
                    f"pages={stats['pages']} unique_candidates={len(records)} cache_hit={cache_hit}",
                    file=sys.stderr,
                )
                next_cursor = (data.get("meta") or {}).get("next_cursor")
                if not data["results"] or not next_cursor:
                    break
                cursor = next_cursor
                if not cache_hit:
                    time.sleep(args.delay)
        payload["status"] = "complete_within_requested_limits"
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt):
        payload["status"] = "interrupted_partial_results"
        payload["papers"] = list(records.values())
        atomic_json(args.output, payload)
        raise
    payload["papers"] = list(records.values())
    atomic_json(args.output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not 1 <= args.per_page <= 200:
        p.error("--per-page must be between 1 and 200")
    if args.max_pages < 1 or args.max_records < 1 or args.retries < 0:
        p.error("page/record limits must be positive and retries must be non-negative")
    if args.year_from > args.year_to or args.year_from < 1000 or args.year_to > 9999:
        p.error("invalid publication-year interval")
    if not 0.1 <= args.delay <= 60 or not 0 < args.timeout <= 60:
        p.error("--delay must be 0.1–60 and --timeout must be greater than 0 and at most 60")
    if args.offline and args.refresh:
        p.error("--offline and --refresh cannot be combined")
    if args.output.resolve() in {
        (ROOT / "catalog" / "papers.json").resolve(),
        (ROOT / "catalog" / "cases.json").resolve(),
    }:
        p.error("candidate acquisition must not overwrite the curated papers/cases catalogs")
    try:
        payload = collect(args)
    except KeyboardInterrupt:
        print("Interrupted; completed pages and partial results were saved.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Collection stopped: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(payload['papers'])} candidate records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
