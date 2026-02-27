"""
summarize_report.py

Read pipeline JSON report and emit a concise summary for the Word report.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_number(value: Optional[Any]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _safe_get(dct: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_summary(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})
    memory = report.get("memory", {})
    discards = report.get("discards", {})
    phase_stats = report.get("phase_stats", {}) or _safe_get(report, "full_stats", "phase_stats") or {}
    full_stats = report.get("full_stats", {})

    total_runtime = summary.get("total_runtime_seconds") or full_stats.get("total_runtime_sec")
    input_rows = summary.get("input_rows_total_raw") or full_stats.get("total_input_rows")
    raw_after = summary.get("raw_rows_after_filters") or full_stats.get("total_raw_rows_after_filters")
    intermediate_rows = summary.get("intermediate_rows_total") or full_stats.get("total_output_rows")
    output_rows = summary.get("output_rows_total_wide") or full_stats.get("wide_table_rows")

    peak_mb = memory.get("peak_rss_mb") or full_stats.get("peak_memory_mb")
    peak_gb = memory.get("peak_rss_gb")

    parse_fail = discards.get("parse_failures") or full_stats.get("total_parse_fail_rows")
    month_mismatch = discards.get("month_mismatch") or full_stats.get("total_month_mismatches")
    low_count = discards.get("low_count_cleanup") or full_stats.get("total_removed_rows")

    mismatch_by_month = full_stats.get("month_mismatch_by_month", {})
    files_with_mismatch = full_stats.get("files_with_month_mismatches", "N/A")
    row_breakdown = full_stats.get("row_breakdown_by_year_and_taxi_type", {})

    lines = [
        "# Pipeline Summary (from report.json)",
        "",
        "## Overall",
        f"- Total runtime: {_fmt_seconds(total_runtime)}",
        f"- Peak memory: {peak_mb} MB" + (f" ({peak_gb} GB)" if peak_gb else ""),
        "",
        "## Row Counts",
        f"- Total input rows: {_fmt_number(input_rows)}",
        f"- Raw rows after parse/month filters: {_fmt_number(raw_after)}",
        f"- Intermediate pivoted rows (sum): {_fmt_number(intermediate_rows)}",
        f"- Final wide table rows: {_fmt_number(output_rows)}",
        "",
        "## Discarded Rows (counts)",
        f"- Parse failures: {_fmt_number(parse_fail)}",
        f"- Month mismatch: {_fmt_number(month_mismatch)}",
        f"- Low-count cleanup: {_fmt_number(low_count)}",
        "",
        "## Date Consistency Issues",
        f"- Files with mismatches: {files_with_mismatch}",
        "",
        "### Month mismatch breakdown",
    ]

    if mismatch_by_month:
        for month in sorted(mismatch_by_month.keys()):
            lines.append(f"- {month}: {mismatch_by_month.get(month)}")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Row Breakdown by Year and Taxi Type",
    ]

    if row_breakdown:
        for year in sorted(row_breakdown.keys()):
            taxis = row_breakdown.get(year, {})
            for taxi_type in sorted(taxis.keys()):
                lines.append(f"- {year} / {taxi_type}: {taxis[taxi_type]}")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Phase Timing",
    ]

    if phase_stats:
        for phase, stats in phase_stats.items():
            lines.append(f"- {phase}: {_fmt_seconds(stats.get('time'))} (mem delta {stats.get('memory_mb', 'N/A')} MB)")
    else:
        lines.append("- (not available)")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pipeline JSON report for Word write-up.")
    parser.add_argument("--report", default="output-full/report.json", help="Path to report.json")
    parser.add_argument("--out", default="report_summary.md", help="Output summary markdown")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    with report_path.open("r") as f:
        report = json.load(f)

    summary_md = build_summary(report)

    out_path = Path(args.out)
    out_path.write_text(summary_md)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
