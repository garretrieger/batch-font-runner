#!/usr/bin/env python3
"""Generates comparison reports (text or HTML) for IFT segmenter datasets.

Supports comparing two trials across fonts, computing runtime and cost deltas,
analyzing timeouts and errors, and exporting either terminal text reports or
standalone interactive HTML dashboards.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Callable, List, Optional, Sequence, Tuple


# ============================================================================
# Statistical and Mathematical Helpers (Pure Python, zero external deps)
# ============================================================================


def compute_percentile(values: Sequence[float], q: float, *, presorted: bool = False) -> float:
    """Computes the q-th percentile (0 <= q <= 100) using linear interpolation.

    Equivalent to numpy.percentile(values, q, method="linear").
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    s = values if presorted else sorted(values)
    n = len(s)
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] + frac * (s[hi] - s[lo]))


def compute_median(values: Sequence[float], *, presorted: bool = False) -> float:
    """Computes median (P50) of a sequence of numbers."""
    return compute_percentile(values, 50.0, presorted=presorted)


def compute_mean(values: Sequence[float]) -> float:
    """Computes arithmetic mean."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class HistogramBin:
    label: str
    count: int
    pct: float
    color: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "count": self.count,
            "pct": self.pct,
            "color": self.color,
        }


def compute_histogram(
    values: Sequence[float],
    bins: Sequence[float],
    labels: Sequence[str],
    colors: Sequence[str],
) -> List[HistogramBin]:
    """Computes histogram bin counts and percentages matching numpy.histogram.

    Bins: [b0, b1, ..., bk]. Bin i covers [b_i, b_{i+1}) except the last bin which covers [b_{k-1}, b_k].
    """
    if len(bins) < 2 or len(labels) != len(bins) - 1 or len(colors) != len(labels):
        raise ValueError("Invalid histogram bins/labels/colors specification.")

    num_bins = len(labels)
    counts = [0] * num_bins
    total = len(values) if values else 1

    for v in values:
        if v < bins[0] or v > bins[-1]:
            continue
        assigned = False
        for i in range(num_bins - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                assigned = True
                break
        if not assigned:
            # Last bin includes right endpoint
            if bins[-2] <= v <= bins[-1]:
                counts[-1] += 1

    result = []
    for i in range(num_bins):
        pct = (counts[i] / total) * 100.0
        result.append(
            HistogramBin(
                label=labels[i],
                count=counts[i],
                pct=round(pct, 3),
                color=colors[i],
            )
        )
    return result


# Standard histogram definitions for time and cost
TIME_HIST_BINS = [-100.0, -50.0, -25.0, -15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 10.0, 50.0, math.inf]
TIME_HIST_LABELS = [
    "-100% to -50% (Massive Speedup)",
    "-50% to -25%",
    "-25% to -15%",
    "-15% to -10%",
    "-10% to -5%",
    "-5% to -2%",
    "-2% to 0%",
    "0% to +2% (Minor Noise)",
    "+2% to +10%",
    "+10% to +50%",
    "> +50% (Major Slowdowns)",
]
TIME_HIST_COLORS = [
    "bg-emerald-500", "bg-emerald-500", "bg-emerald-500", "bg-emerald-500",
    "bg-emerald-500", "bg-emerald-400", "bg-emerald-300",
    "bg-amber-400", "bg-amber-500", "bg-rose-400", "bg-rose-500"
]

COST_HIST_BINS = [-100.0, -5.0, -2.0, -1.0, -0.5, -0.0001, 0.0001, 0.5, 1.0, 2.0, 5.0, math.inf]
COST_HIST_LABELS = [
    "< -5% (Major Cost Drop)",
    "-5% to -2%",
    "-2% to -1%",
    "-1% to -0.5%",
    "-0.5% to 0%",
    "0.00% (Identical Cost)",
    "0% to +0.5%",
    "+0.5% to +1.0%",
    "+1.0% to +2.0%",
    "+2.0% to +5.0%",
    "> +5.0% (Major Cost Increase)",
]
COST_HIST_COLORS = [
    "bg-emerald-500", "bg-emerald-500", "bg-emerald-500", "bg-emerald-400",
    "bg-emerald-300", "bg-slate-400", "bg-amber-300", "bg-amber-400",
    "bg-amber-500", "bg-rose-400", "bg-rose-500"
]


# ============================================================================
# In-Memory Report Data Model
# ============================================================================


@dataclass(frozen=True)
class DatasetFiles:
    scsv_path: Path
    dataset_name: str
    notes_text: str
    error_log_path: Optional[Path]


@dataclass(frozen=True)
class RunRecord:
    font_name: str
    font_path: str
    quality_level: int
    trial: str
    ideal_total_cost: float
    ift_total_cost: float
    non_ift_total_cost: float
    uncovered_ift_patch_cost: float
    total_time_s: float
    codepoint_count: int


@dataclass(frozen=True)
class ErrorRecord:
    font_path: str
    font_name: str
    trial: str
    quality_level: int
    error_type: str
    detail: str
    raw: str


@dataclass(frozen=True)
class MatchedFontRecord:
    font_name: str
    font_path: str
    quality_level: int
    codepoint_count: int
    control_time_s: float
    candidate_time_s: float
    time_delta_s: float
    time_delta_pct: float
    control_cost: int
    candidate_cost: int
    cost_delta: int
    cost_delta_pct: float
    control_uncov: int
    candidate_uncov: int
    uncov_delta: int
    uncov_delta_pct: Optional[float]
    ideal_cost: int
    non_ift_cost: int

    def to_js_dict(self) -> dict:
        return {
            "name": self.font_name,
            "path": self.font_path,
            "q": self.quality_level,
            "cps": self.codepoint_count,
            "bt": round(self.control_time_s, 4),
            "ct": round(self.candidate_time_s, 4),
            "td": round(self.time_delta_s, 4),
            "tp": round(self.time_delta_pct, 3),
            "bc": self.control_cost,
            "cc": self.candidate_cost,
            "cd": self.cost_delta,
            "cp": round(self.cost_delta_pct, 3),
            "bu": self.control_uncov,
            "cu": self.candidate_uncov,
            "ud": self.uncov_delta,
            "up": round(self.uncov_delta_pct, 3) if self.uncov_delta_pct is not None else None,
            "ideal": self.ideal_cost,
            "non_ift": self.non_ift_cost,
        }


@dataclass(frozen=True)
class UnmatchedFontRecord:
    font_name: str
    font_path: str
    selected_quality: int
    record: RunRecord
    other_max_quality: Optional[int]

    def to_js_dict(self) -> dict:
        return {
            "font_name": self.font_name,
            "font_path": self.font_path,
            "selected_quality": self.selected_quality,
            "total_time_s": round(self.record.total_time_s, 4),
            "ift_total_cost": self.record.ift_total_cost,
            "uncovered_ift_patch_cost": self.record.uncovered_ift_patch_cost,
            "other_max_quality": self.other_max_quality,
        }


@dataclass(frozen=True)
class MetricAggregate:
    control_sum: float
    candidate_sum: float
    delta_sum: float
    delta_pct: Optional[float]
    control_median: float
    candidate_median: float
    delta_pct_median: Optional[float]
    delta_pct_mean: Optional[float]
    improved_count: int
    regressed_count: int
    unchanged_count: int
    active_count: int

    def to_js_dict(self) -> dict:
        return {
            "control_sum": round(self.control_sum, 4),
            "candidate_sum": round(self.candidate_sum, 4),
            "delta_sum": round(self.delta_sum, 4),
            "delta_pct": round(self.delta_pct, 3) if self.delta_pct is not None else None,
            "control_median": round(self.control_median, 4),
            "candidate_median": round(self.candidate_median, 4),
            "delta_pct_median": round(self.delta_pct_median, 3) if self.delta_pct_median is not None else None,
            "delta_pct_mean": round(self.delta_pct_mean, 3) if self.delta_pct_mean is not None else None,
            "improved_count": self.improved_count,
            "regressed_count": self.regressed_count,
            "unchanged_count": self.unchanged_count,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class QuadrantSummary:
    fast_cheap: int
    fast_expensive: int
    slow_cheap: int
    slow_expensive: int
    total: int

    @property
    def fast_cheap_pct(self) -> float:
        return (self.fast_cheap / self.total * 100.0) if self.total else 0.0

    @property
    def fast_exp_pct(self) -> float:
        return (self.fast_expensive / self.total * 100.0) if self.total else 0.0

    @property
    def slow_cheap_pct(self) -> float:
        return (self.slow_cheap / self.total * 100.0) if self.total else 0.0

    @property
    def slow_exp_pct(self) -> float:
        return (self.slow_expensive / self.total * 100.0) if self.total else 0.0

    def to_js_dict(self) -> dict:
        return {
            "fast_cheap": self.fast_cheap,
            "fast_expensive": self.fast_expensive,
            "slow_cheap": self.slow_cheap,
            "slow_expensive": self.slow_expensive,
            "total": self.total,
        }


@dataclass(frozen=True)
class QuantileRow:
    percentile: int
    label: str
    is_median: bool
    control_time_s: float
    candidate_time_s: float
    time_delta_s: float
    time_delta_pct: float
    control_cost: float
    candidate_cost: float
    cost_delta: float
    cost_delta_pct: float
    control_uncov: float
    candidate_uncov: float
    uncov_delta: float
    uncov_delta_pct: Optional[float]

    def to_js_dict(self) -> dict:
        return {
            "percentile": self.percentile,
            "label": self.label,
            "is_median": self.is_median,
            "control_time_s": round(self.control_time_s, 4),
            "candidate_time_s": round(self.candidate_time_s, 4),
            "time_delta_s": round(self.time_delta_s, 4),
            "time_delta_pct": round(self.time_delta_pct, 3),
            "control_cost": self.control_cost,
            "candidate_cost": self.candidate_cost,
            "cost_delta": self.cost_delta,
            "cost_delta_pct": round(self.cost_delta_pct, 3),
            "control_uncov": self.control_uncov,
            "candidate_uncov": self.candidate_uncov,
            "uncov_delta": self.uncov_delta,
            "uncov_delta_pct": round(self.uncov_delta_pct, 3) if self.uncov_delta_pct is not None else None,
        }


@dataclass(frozen=True)
class RuntimeTierRow:
    name: str
    count: int
    control_time_sum: float
    candidate_time_sum: float
    time_delta_sum: float
    time_delta_pct: float
    median_time_delta_pct: float
    median_cost_delta_pct: float
    median_uncov_delta_pct: Optional[float]

    def to_js_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "control_time_sum": round(self.control_time_sum, 4),
            "candidate_time_sum": round(self.candidate_time_sum, 4),
            "time_delta_sum": round(self.time_delta_sum, 4),
            "time_delta_pct": round(self.time_delta_pct, 3),
            "median_time_delta_pct": round(self.median_time_delta_pct, 3),
            "median_cost_delta_pct": round(self.median_cost_delta_pct, 3),
            "median_uncov_delta_pct": round(self.median_uncov_delta_pct, 3) if self.median_uncov_delta_pct is not None else None,
        }


@dataclass(frozen=True)
class PerQualityTimeoutRow:
    quality_level: int
    control_count: int
    candidate_count: int
    diff_count: int
    resolved_count: int
    new_count: int
    persistent_count: int
    missing_count: int = 0

    def to_js_dict(self) -> dict:
        return {
            "quality_level": self.quality_level,
            "control_count": self.control_count,
            "candidate_count": self.candidate_count,
            "diff_count": self.diff_count,
            "resolved_count": self.resolved_count,
            "new_count": self.new_count,
            "persistent_count": self.persistent_count,
            "missing_count": self.missing_count,
        }


@dataclass(frozen=True)
class TimeoutComparisonRow:
    font_name: str
    font_path: str
    quality_level: int
    status: str
    status_key: str
    status_badge: str
    control_desc: str
    candidate_desc: str

    def to_js_dict(self) -> dict:
        return {
            "font_name": self.font_name,
            "font_path": self.font_path,
            "q": self.quality_level,
            "status": self.status,
            "status_key": self.status_key,
            "status_badge": self.status_badge,
            "base_desc": self.control_desc,
            "cand_desc": self.candidate_desc,
        }


@dataclass(frozen=True)
class TimeoutSummary:
    control_count: int
    candidate_count: int
    diff_count: int
    control_fonts_any: int
    candidate_fonts_any: int
    diff_fonts_any: int
    control_fonts_all: int
    candidate_fonts_all: int
    diff_fonts_all: int
    resolved_count: int
    new_count: int
    persistent_count: int
    missing_count: int
    per_quality: List[PerQualityTimeoutRow]
    rows: List[TimeoutComparisonRow]

    def to_js_dict(self) -> dict:
        return {
            "control_count": self.control_count,
            "candidate_count": self.candidate_count,
            "diff_count": self.diff_count,
            "control_fonts_any": self.control_fonts_any,
            "candidate_fonts_any": self.candidate_fonts_any,
            "diff_fonts_any": self.diff_fonts_any,
            "control_fonts_all": self.control_fonts_all,
            "candidate_fonts_all": self.candidate_fonts_all,
            "diff_fonts_all": self.diff_fonts_all,
            "resolved_count": self.resolved_count,
            "new_count": self.new_count,
            "persistent_count": self.persistent_count,
            "missing_count": self.missing_count,
            "per_quality": [pq.to_js_dict() for pq in self.per_quality],
            "rows": [r.to_js_dict() for r in self.rows],
        }


@dataclass(frozen=True)
class ComparisonReport:
    dataset_name: str
    notes_text: str
    control_trial: str
    candidate_trial: str
    quality_filter: Optional[int]
    quality_mode_desc: str
    quality_counts: dict[int, int]
    time_stats: MetricAggregate
    cost_stats: MetricAggregate
    uncov_stats: MetricAggregate
    quadrants: QuadrantSummary
    time_histogram: List[HistogramBin]
    cost_histogram: List[HistogramBin]
    uncov_histogram: List[HistogramBin]
    quantiles: List[QuantileRow]
    runtime_tiers: List[RuntimeTierRow]
    matched_records: List[MatchedFontRecord]
    unmatched_candidate_only: List[UnmatchedFontRecord]
    unmatched_control_only: List[UnmatchedFontRecord]
    timeouts: TimeoutSummary

    @property
    def total_matched_fonts(self) -> int:
        return len(self.matched_records)

    def to_js_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "notes_text": self.notes_text,
            "base_name": self.control_trial,
            "cand_name": self.candidate_trial,
            "quality_filter": self.quality_filter,
            "quality_mode_desc": self.quality_mode_desc,
            "quality_counts": self.quality_counts,
            "time_stats": self.time_stats.to_js_dict(),
            "cost_stats": self.cost_stats.to_js_dict(),
            "uncov_stats": self.uncov_stats.to_js_dict(),
            "quadrants": self.quadrants.to_js_dict(),
            "time_hist": [b.to_dict() for b in self.time_histogram],
            "cost_hist": [b.to_dict() for b in self.cost_histogram],
            "uncov_hist": [b.to_dict() for b in self.uncov_histogram],
            "quantiles": [q.to_js_dict() for q in self.quantiles],
            "runtime_tiers": [t.to_js_dict() for t in self.runtime_tiers],
            "records": [r.to_js_dict() for r in self.matched_records],
            "unmatched_candidate_only": [r.to_js_dict() for r in self.unmatched_candidate_only],
            "unmatched_control_only": [r.to_js_dict() for r in self.unmatched_control_only],
            "timeouts": self.timeouts.to_js_dict(),
            "timeout_rows": [r.to_js_dict() for r in self.timeouts.rows],
        }


# ============================================================================
# Dataset Resolution & Parsing
# ============================================================================


def resolve_dataset(path_str: str | Path) -> DatasetFiles:
    """Returns DatasetFiles (scsv_path, dataset_name, notes_text, error_log_path)."""
    p = Path(path_str)
    if p.is_dir():
        scsv_path = p / "out.scsv"
        notes_path = p / "NOTES.txt"
        error_log_path = p / "error_log.txt"
        dataset_name = p.name
    else:
        scsv_path = p
        notes_path = p.parent / "NOTES.txt"
        error_log_path = p.parent / "error_log.txt"
        dataset_name = p.parent.name if p.name == "out.scsv" else p.stem

    if not scsv_path.exists():
        raise FileNotFoundError(f"Could not find data file at {scsv_path}")

    notes_text = ""
    if notes_path.exists():
        notes_text = notes_path.read_text(encoding="utf-8").strip()

    err_path = error_log_path if error_log_path.exists() else None
    return DatasetFiles(
        scsv_path=scsv_path,
        dataset_name=dataset_name,
        notes_text=notes_text,
        error_log_path=err_path,
    )


def load_scsv(path: Path) -> Tuple[dict[str, dict[str, dict[int, RunRecord]]], List[str], List[int]]:
    """Loads out.scsv into data[trial][font_path][quality_level] = RunRecord.

    Returns (data, sorted_trials, sorted_qualities).
    """
    data: dict[str, dict[str, dict[int, RunRecord]]] = defaultdict(lambda: defaultdict(dict))
    trials: set[str] = set()
    qualities: set[int] = set()

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty dataset file at {path}")

        header = [h.strip() for h in raw_header]
        col_idx = {name: idx for idx, name in enumerate(header)}
        required = [
            "font_path",
            "quality_level",
            "trial",
            "ideal_total_cost",
            "ift_total_cost",
            "non_ift_total_cost",
            "total_time_s",
            "codepoint_count",
        ]
        for req in required:
            if req not in col_idx:
                raise ValueError(f"Missing required column '{req}' in {path} (found: {header})")

        min_row_len = max(col_idx[c] for c in required) + 1
        has_uncov = "uncovered_ift_patch_cost" in col_idx

        for line_num, row in enumerate(reader, start=2):
            if not row or not any(field.strip() for field in row):
                continue
            # Skip repeated header lines (often occurs if logs were appended)
            if col_idx["font_path"] < len(row) and row[col_idx["font_path"]].strip() == "font_path":
                continue
            if len(row) < min_row_len:
                raise ValueError(
                    f"Malformed row at line {line_num} in {path}: expected at least {min_row_len} columns, got {len(row)}"
                )

            try:
                font_path = row[col_idx["font_path"]].strip()
                q = int(row[col_idx["quality_level"]].strip())
                trial = row[col_idx["trial"]].strip()
                total_time = float(row[col_idx["total_time_s"]].strip())
                if total_time <= 0:
                    raise ValueError(
                        f"Invalid row at line {line_num} in {path}: total_time_s must be > 0, got {total_time}"
                    )
                ift_cost = float(row[col_idx["ift_total_cost"]].strip())
                if ift_cost <= 0:
                    raise ValueError(
                        f"Invalid row at line {line_num} in {path}: ift_total_cost must be > 0, got {ift_cost}"
                    )
                uncov_val = (
                    float(row[col_idx["uncovered_ift_patch_cost"]].strip())
                    if has_uncov and col_idx["uncovered_ift_patch_cost"] < len(row)
                    else 0.0
                )
                rec = RunRecord(
                    font_name=os.path.basename(font_path),
                    font_path=font_path,
                    quality_level=q,
                    trial=trial,
                    ideal_total_cost=float(row[col_idx["ideal_total_cost"]].strip()),
                    ift_total_cost=ift_cost,
                    non_ift_total_cost=float(row[col_idx["non_ift_total_cost"]].strip()),
                    uncovered_ift_patch_cost=uncov_val,
                    total_time_s=total_time,
                    codepoint_count=int(row[col_idx["codepoint_count"]].strip()),
                )
                data[trial][font_path][q] = rec
                trials.add(trial)
                qualities.add(q)
            except ValueError as e:
                raise ValueError(f"Error parsing row at line {line_num} in {path}: {e}") from e

    return data, sorted(trials), sorted(qualities)


def parse_error_log(error_log_path: Optional[Path]) -> dict[str, dict[str, dict[int, ErrorRecord]]]:
    """Parses error_log.txt into errors[trial][font_path][quality_level] = ErrorRecord.

    Keeps the first error encountered for each (trial, font_path, quality_level) so
    underlying root errors are not overwritten by trailing exception handlers.
    """
    errors: dict[str, dict[str, dict[int, ErrorRecord]]] = defaultdict(lambda: defaultdict(dict))
    if not error_log_path or not error_log_path.exists():
        return errors

    timeout_re = re.compile(
        r"Timeout occurred for font:\s*(.+?),\s*trial:\s*(\S+)\s+at quality level:\s*(\d+)"
    )
    error_re_1 = re.compile(
        r"Error processing\s*(.+?),\s*quality(?:\s+level)?:?\s*(\d+),\s*trial:?\s*(\S+?):\s*(.+)"
    )
    error_re_2 = re.compile(
        r"Error processing\s*(.+?),\s*trial:?\s*(\S+?),\s*quality(?:\s+level)?:?\s*(\d+):\s*(.+)"
    )
    exception_re = re.compile(
        r"(.+?),\s*quality\s*(\d+),\s*trial\s*(\S+)\s+generated an exception:\s*(.+)"
    )
    sigterm_re = re.compile(r"return code (?:143|-15)\b|\bsigterm\b|\btimeout\b", re.IGNORECASE)

    for line in error_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("Processed "):
            continue

        m_to = timeout_re.search(line)
        if m_to:
            font_path = m_to.group(1).strip()
            trial = m_to.group(2).strip()
            q = int(m_to.group(3).strip())
            if q not in errors[trial][font_path]:
                errors[trial][font_path][q] = ErrorRecord(
                    font_path=font_path,
                    font_name=os.path.basename(font_path),
                    trial=trial,
                    quality_level=q,
                    error_type="TIMEOUT",
                    detail="Timeout occurred",
                    raw=line,
                )
            continue

        m_err1 = error_re_1.search(line)
        if m_err1:
            font_path = m_err1.group(1).strip()
            q = int(m_err1.group(2).strip())
            trial = m_err1.group(3).strip()
            detail = m_err1.group(4).strip()
            if q not in errors[trial][font_path]:
                is_sigterm = bool(sigterm_re.search(detail))
                errors[trial][font_path][q] = ErrorRecord(
                    font_path=font_path,
                    font_name=os.path.basename(font_path),
                    trial=trial,
                    quality_level=q,
                    error_type="TIMEOUT (SIGTERM/143)" if is_sigterm else "ERROR",
                    detail=detail,
                    raw=line,
                )
            continue

        m_err2 = error_re_2.search(line)
        if m_err2:
            font_path = m_err2.group(1).strip()
            trial = m_err2.group(2).strip()
            q = int(m_err2.group(3).strip())
            detail = m_err2.group(4).strip()
            if q not in errors[trial][font_path]:
                is_sigterm = bool(sigterm_re.search(detail))
                errors[trial][font_path][q] = ErrorRecord(
                    font_path=font_path,
                    font_name=os.path.basename(font_path),
                    trial=trial,
                    quality_level=q,
                    error_type="TIMEOUT (SIGTERM/143)" if is_sigterm else "ERROR",
                    detail=detail,
                    raw=line,
                )
            continue

        m_exc = exception_re.search(line)
        if m_exc:
            font_path = m_exc.group(1).strip()
            q = int(m_exc.group(2).strip())
            trial = m_exc.group(3).strip()
            detail = m_exc.group(4).strip()
            if q not in errors[trial][font_path]:
                errors[trial][font_path][q] = ErrorRecord(
                    font_path=font_path,
                    font_name=os.path.basename(font_path),
                    trial=trial,
                    quality_level=q,
                    error_type="EXCEPTION",
                    detail=detail,
                    raw=line,
                )

    return errors


def select_records(
    data: dict[str, dict[str, dict[int, RunRecord]]],
    control_trial: str,
    candidate_trial: str,
    quality_filter: Optional[int] = None,
    quality_mode: str = "common",
) -> Tuple[
    List[Tuple[str, int, RunRecord, RunRecord]],
    List[UnmatchedFontRecord],
    List[UnmatchedFontRecord],
    dict[int, int],
]:
    """Selects matched (control, candidate) records per font according to quality selection rules.

    Modes:
      - Explicit filter (quality_filter is not None): selects that exact quality level.
      - "common" (default when filter is None): selects the highest quality level where BOTH
        candidate and control completed, maximizing paired statistical coverage.
      - "candidate": selects each font's highest completed quality in candidate and pairs it
        with the matching quality in control (dropping fonts if control timed out at that quality).

    Returns:
      (matched_pairs, unmatched_cand_only, unmatched_ctrl_only, quality_counts)
    """
    ctrl_by_font = data.get(control_trial, {})
    cand_by_font = data.get(candidate_trial, {})
    all_fonts = sorted(set(ctrl_by_font.keys()) | set(cand_by_font.keys()))

    matched_pairs: List[Tuple[str, int, RunRecord, RunRecord]] = []
    unmatched_cand_only: List[UnmatchedFontRecord] = []
    unmatched_ctrl_only: List[UnmatchedFontRecord] = []
    quality_counts: dict[int, int] = defaultdict(int)

    for f in all_fonts:
        c_qualities = cand_by_font.get(f, {})
        b_qualities = ctrl_by_font.get(f, {})

        if quality_filter is not None:
            q_sel = quality_filter
            c_rec = c_qualities.get(q_sel)
            b_rec = b_qualities.get(q_sel)
            if c_rec is not None and b_rec is not None:
                matched_pairs.append((f, q_sel, b_rec, c_rec))
                quality_counts[q_sel] += 1
            elif c_rec is not None and b_rec is None:
                b_max = max(b_qualities.keys()) if b_qualities else None
                unmatched_cand_only.append(
                    UnmatchedFontRecord(
                        font_name=os.path.basename(f),
                        font_path=f,
                        selected_quality=q_sel,
                        record=c_rec,
                        other_max_quality=b_max,
                    )
                )
            elif c_rec is None and b_rec is not None:
                c_max = max(c_qualities.keys()) if c_qualities else None
                unmatched_ctrl_only.append(
                    UnmatchedFontRecord(
                        font_name=os.path.basename(f),
                        font_path=f,
                        selected_quality=q_sel,
                        record=b_rec,
                        other_max_quality=c_max,
                    )
                )
        elif quality_mode == "common":
            common_qs = set(c_qualities.keys()) & set(b_qualities.keys())
            if common_qs:
                q_sel = max(common_qs)
                matched_pairs.append((f, q_sel, b_qualities[q_sel], c_qualities[q_sel]))
                quality_counts[q_sel] += 1
            elif c_qualities:
                q_sel = max(c_qualities.keys())
                b_max = max(b_qualities.keys()) if b_qualities else None
                unmatched_cand_only.append(
                    UnmatchedFontRecord(
                        font_name=os.path.basename(f),
                        font_path=f,
                        selected_quality=q_sel,
                        record=c_qualities[q_sel],
                        other_max_quality=b_max,
                    )
                )
            elif b_qualities:
                q_sel = max(b_qualities.keys())
                unmatched_ctrl_only.append(
                    UnmatchedFontRecord(
                        font_name=os.path.basename(f),
                        font_path=f,
                        selected_quality=q_sel,
                        record=b_qualities[q_sel],
                        other_max_quality=None,
                    )
                )
        else:  # quality_mode == "candidate"
            if not c_qualities:
                if b_qualities:
                    b_max = max(b_qualities.keys())
                    unmatched_ctrl_only.append(
                        UnmatchedFontRecord(
                            font_name=os.path.basename(f),
                            font_path=f,
                            selected_quality=b_max,
                            record=b_qualities[b_max],
                            other_max_quality=None,
                        )
                    )
                continue

            q_sel = max(c_qualities.keys())
            c_rec = c_qualities[q_sel]
            b_rec = b_qualities.get(q_sel)
            if b_rec is not None:
                matched_pairs.append((f, q_sel, b_rec, c_rec))
                quality_counts[q_sel] += 1
            else:
                b_max = max(b_qualities.keys()) if b_qualities else None
                unmatched_cand_only.append(
                    UnmatchedFontRecord(
                        font_name=os.path.basename(f),
                        font_path=f,
                        selected_quality=q_sel,
                        record=c_rec,
                        other_max_quality=b_max,
                    )
                )

    return matched_pairs, unmatched_cand_only, unmatched_ctrl_only, dict(sorted(quality_counts.items()))


def analyze_timeouts(
    errors: dict[str, dict[str, dict[int, ErrorRecord]]],
    data: dict[str, dict[str, dict[int, RunRecord]]],
    control_trial: str,
    candidate_trial: str,
    all_qualities: Sequence[int],
    quality_filter: Optional[int] = None,
) -> TimeoutSummary:
    """Compares timeouts/errors between control and candidate trials.

    Properly differentiates:
      - RESOLVED: failed in control, completed successfully in candidate
      - NEW: completed successfully in control, failed in candidate
      - PERSISTENT: failed in both control and candidate
      - MISSING: failed in one trial and has no record in the other (partial run / killed)
    """
    ctrl_errs = errors.get(control_trial, {})
    cand_errs = errors.get(candidate_trial, {})
    ctrl_data = data.get(control_trial, {})
    cand_data = data.get(candidate_trial, {})

    qualities_to_check = [quality_filter] if quality_filter is not None else list(all_qualities)

    b_err_set: set[Tuple[str, int]] = set()
    c_err_set: set[Tuple[str, int]] = set()
    per_q_summary: List[PerQualityTimeoutRow] = []

    for q in qualities_to_check:
        bq_set = {f for f, qmap in ctrl_errs.items() if q in qmap}
        cq_set = {f for f, qmap in cand_errs.items() if q in qmap}
        for f in bq_set:
            b_err_set.add((f, q))
        for f in cq_set:
            c_err_set.add((f, q))

        res_q = sum(1 for f in (bq_set - cq_set) if q in cand_data.get(f, {}))
        miss_bq = sum(1 for f in (bq_set - cq_set) if q not in cand_data.get(f, {}))
        new_q = sum(1 for f in (cq_set - bq_set) if q in ctrl_data.get(f, {}))
        miss_cq = sum(1 for f in (cq_set - bq_set) if q not in ctrl_data.get(f, {}))
        per_q = len(bq_set & cq_set)
        missing_q = miss_bq + miss_cq

        per_q_summary.append(
            PerQualityTimeoutRow(
                quality_level=q,
                control_count=len(bq_set),
                candidate_count=len(cq_set),
                diff_count=len(cq_set) - len(bq_set),
                resolved_count=res_q,
                new_count=new_q,
                persistent_count=per_q,
                missing_count=missing_q,
            )
        )

    all_err_pairs = sorted(b_err_set | c_err_set, key=lambda x: (os.path.basename(x[0]), x[1]))
    rows: List[TimeoutComparisonRow] = []
    resolved_count = 0
    new_count = 0
    persistent_count = 0
    missing_count = 0

    for f, q in all_err_pairs:
        in_b_err = (f, q) in b_err_set
        in_c_err = (f, q) in c_err_set
        b_ok = q in ctrl_data.get(f, {})
        c_ok = q in cand_data.get(f, {})

        if in_b_err and not in_c_err:
            if c_ok:
                status = "RESOLVED"
                status_key = "resolved"
                status_badge = "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                resolved_count += 1
            else:
                status = "MISSING DATA"
                status_key = "missing"
                status_badge = "bg-slate-500/20 text-slate-400 border-slate-500/30"
                missing_count += 1
        elif not in_b_err and in_c_err:
            if b_ok:
                status = "NEW TIMEOUT/ERROR"
                status_key = "new"
                status_badge = "bg-rose-500/20 text-rose-400 border-rose-500/30"
                new_count += 1
            else:
                status = "MISSING DATA"
                status_key = "missing"
                status_badge = "bg-slate-500/20 text-slate-400 border-slate-500/30"
                missing_count += 1
        else:
            status = "PERSISTENT"
            status_key = "persistent"
            status_badge = "bg-amber-500/20 text-amber-400 border-amber-500/30"
            persistent_count += 1

        if in_b_err:
            be = ctrl_errs[f][q]
            detail_str = f" ({be.detail})" if (be.detail and be.detail != "Timeout occurred") else ""
            b_desc = be.error_type + detail_str
        elif b_ok:
            br = ctrl_data[f][q]
            b_desc = f"Completed in {br.total_time_s:.2f}s (Cost: {int(br.ift_total_cost):,}, Uncov: {int(br.uncovered_ift_patch_cost):,})"
        else:
            b_desc = "Missing"

        if in_c_err:
            ce = cand_errs[f][q]
            detail_str = f" ({ce.detail})" if (ce.detail and ce.detail != "Timeout occurred") else ""
            c_desc = ce.error_type + detail_str
        elif c_ok:
            cr = cand_data[f][q]
            c_desc = f"Completed in {cr.total_time_s:.2f}s (Cost: {int(cr.ift_total_cost):,}, Uncov: {int(cr.uncovered_ift_patch_cost):,})"
        else:
            c_desc = "Missing"

        rows.append(
            TimeoutComparisonRow(
                font_name=os.path.basename(f),
                font_path=f,
                quality_level=q,
                status=status,
                status_key=status_key,
                status_badge=status_badge,
                control_desc=b_desc,
                candidate_desc=c_desc,
            )
        )

    num_q = len(qualities_to_check)
    b_fonts_any = {f for f, q in b_err_set}
    c_fonts_any = {f for f, q in c_err_set}
    b_fonts_all = {f for f in b_fonts_any if sum(1 for q in qualities_to_check if (f, q) in b_err_set) == num_q}
    c_fonts_all = {f for f in c_fonts_any if sum(1 for q in qualities_to_check if (f, q) in c_err_set) == num_q}

    return TimeoutSummary(
        control_count=len(b_err_set),
        candidate_count=len(c_err_set),
        diff_count=len(c_err_set) - len(b_err_set),
        control_fonts_any=len(b_fonts_any),
        candidate_fonts_any=len(c_fonts_any),
        diff_fonts_any=len(c_fonts_any) - len(b_fonts_any),
        control_fonts_all=len(b_fonts_all),
        candidate_fonts_all=len(c_fonts_all),
        diff_fonts_all=len(c_fonts_all) - len(b_fonts_all),
        resolved_count=resolved_count,
        new_count=new_count,
        persistent_count=persistent_count,
        missing_count=missing_count,
        per_quality=per_q_summary,
        rows=rows,
    )


def build_report(
    dataset_name: str,
    notes_text: str,
    data: dict[str, dict[str, dict[int, RunRecord]]],
    errors: dict[str, dict[str, dict[int, ErrorRecord]]],
    available_qualities: Sequence[int],
    control_trial: str,
    candidate_trial: str,
    quality_filter: Optional[int] = None,
    quality_mode: str = "common",
) -> ComparisonReport:
    """Builds the comprehensive in-memory ComparisonReport."""
    all_trials = sorted(set(data.keys()) | set(errors.keys()))
    if control_trial not in all_trials:
        raise ValueError(
            f"Control trial '{control_trial}' not found. Available trials: {', '.join(all_trials)}"
        )
    if candidate_trial not in all_trials:
        raise ValueError(
            f"Candidate trial '{candidate_trial}' not found. Available trials: {', '.join(all_trials)}"
        )
    if quality_filter is not None and quality_filter not in available_qualities:
        raise ValueError(
            f"Quality level {quality_filter} not found in dataset. Available qualities: {', '.join(map(str, available_qualities))}"
        )

    matched_pairs, unmatched_cand, unmatched_ctrl, quality_counts = select_records(
        data, control_trial, candidate_trial, quality_filter=quality_filter, quality_mode=quality_mode
    )
    if not matched_pairs:
        raise ValueError("No matched font records found for the specified trials and quality filter.")

    timeouts = analyze_timeouts(
        errors, data, control_trial, candidate_trial, available_qualities, quality_filter
    )

    matched_records: List[MatchedFontRecord] = []
    for f, q_sel, b, c in matched_pairs:
        bt, ct = b.total_time_s, c.total_time_s
        bc, cc = b.ift_total_cost, c.ift_total_cost
        bu, cu = b.uncovered_ift_patch_cost, c.uncovered_ift_patch_cost

        if bt <= 0:
            raise ValueError(f"Control record for {f} at quality {q_sel} has invalid total_time_s: {bt}")
        if ct <= 0:
            raise ValueError(f"Candidate record for {f} at quality {q_sel} has invalid total_time_s: {ct}")
        if bc <= 0:
            raise ValueError(f"Control record for {f} at quality {q_sel} has invalid ift_total_cost: {bc}")
        if cc <= 0:
            raise ValueError(f"Candidate record for {f} at quality {q_sel} has invalid ift_total_cost: {cc}")

        td = ct - bt
        tp = (td / bt * 100.0)
        cd = cc - bc
        cp = (cd / bc * 100.0)

        bu_int = int(round(bu))
        cu_int = int(round(cu))
        ud_int = cu_int - bu_int
        if bu_int > 0:
            up: Optional[float] = (ud_int / bu_int) * 100.0
        elif cu_int > 0:
            up = 100.0
        else:
            up = None

        matched_records.append(
            MatchedFontRecord(
                font_name=b.font_name,
                font_path=f,
                quality_level=q_sel,
                codepoint_count=b.codepoint_count,
                control_time_s=bt,
                candidate_time_s=ct,
                time_delta_s=td,
                time_delta_pct=tp,
                control_cost=int(round(bc)),
                candidate_cost=int(round(cc)),
                cost_delta=int(round(cd)),
                cost_delta_pct=cp,
                control_uncov=bu_int,
                candidate_uncov=cu_int,
                uncov_delta=ud_int,
                uncov_delta_pct=up,
                ideal_cost=int(round(b.ideal_total_cost)),
                non_ift_cost=int(round(b.non_ift_total_cost)),
            )
        )

    # Time aggregate
    bt_vals = [r.control_time_s for r in matched_records]
    ct_vals = [r.candidate_time_s for r in matched_records]
    td_vals = [r.time_delta_s for r in matched_records]
    tp_vals = [r.time_delta_pct for r in matched_records]

    bt_sorted = sorted(bt_vals)
    ct_sorted = sorted(ct_vals)
    td_sorted = sorted(td_vals)
    tp_sorted = sorted(tp_vals)

    sum_bt = sum(bt_vals)
    sum_ct = sum(ct_vals)
    sum_td = sum_ct - sum_bt
    sum_tp = (sum_td / sum_bt * 100.0) if sum_bt > 0 else 0.0

    time_stats = MetricAggregate(
        control_sum=sum_bt,
        candidate_sum=sum_ct,
        delta_sum=sum_td,
        delta_pct=sum_tp,
        control_median=compute_median(bt_sorted, presorted=True),
        candidate_median=compute_median(ct_sorted, presorted=True),
        delta_pct_median=compute_median(tp_sorted, presorted=True),
        delta_pct_mean=compute_mean(tp_vals),
        improved_count=sum(1 for r in matched_records if r.time_delta_s < 0),
        regressed_count=sum(1 for r in matched_records if r.time_delta_s > 0),
        unchanged_count=sum(1 for r in matched_records if r.time_delta_s == 0),
        active_count=len(matched_records),
    )

    # Cost aggregate
    bc_vals = [float(r.control_cost) for r in matched_records]
    cc_vals = [float(r.candidate_cost) for r in matched_records]
    cd_vals = [float(r.cost_delta) for r in matched_records]
    cp_vals = [r.cost_delta_pct for r in matched_records]

    bc_sorted = sorted(bc_vals)
    cc_sorted = sorted(cc_vals)
    cd_sorted = sorted(cd_vals)
    cp_sorted = sorted(cp_vals)

    sum_bc = sum(bc_vals)
    sum_cc = sum(cc_vals)
    sum_cd = sum_cc - sum_bc
    sum_cp = (sum_cd / sum_bc * 100.0) if sum_bc > 0 else 0.0

    cost_stats = MetricAggregate(
        control_sum=sum_bc,
        candidate_sum=sum_cc,
        delta_sum=sum_cd,
        delta_pct=sum_cp,
        control_median=compute_median(bc_sorted, presorted=True),
        candidate_median=compute_median(cc_sorted, presorted=True),
        delta_pct_median=compute_median(cp_sorted, presorted=True),
        delta_pct_mean=compute_mean(cp_vals),
        improved_count=sum(1 for r in matched_records if r.cost_delta < 0),
        regressed_count=sum(1 for r in matched_records if r.cost_delta > 0),
        unchanged_count=sum(1 for r in matched_records if r.cost_delta == 0),
        active_count=len(matched_records),
    )

    # Uncovered cost aggregate
    bu_vals = [float(r.control_uncov) for r in matched_records]
    cu_vals = [float(r.candidate_uncov) for r in matched_records]
    ud_vals = [float(r.uncov_delta) for r in matched_records]

    bu_sorted = sorted(bu_vals)
    cu_sorted = sorted(cu_vals)
    ud_sorted = sorted(ud_vals)

    sum_bu = sum(bu_vals)
    sum_cu = sum(cu_vals)
    sum_ud = sum_cu - sum_bu

    active_uncov_records = [r for r in matched_records if r.control_uncov > 0 or r.candidate_uncov > 0]
    up_active = sorted(r.uncov_delta_pct for r in active_uncov_records if r.uncov_delta_pct is not None)
    active_uncov_count = len(active_uncov_records)

    if active_uncov_count > 0:
        sum_up: Optional[float] = (sum_ud / sum_bu * 100.0) if sum_bu > 0 else 100.0
        u_med: Optional[float] = compute_median(up_active, presorted=True)
        u_mean: Optional[float] = compute_mean(up_active)
    else:
        sum_up = None
        u_med = None
        u_mean = None

    uncov_stats = MetricAggregate(
        control_sum=sum_bu,
        candidate_sum=sum_cu,
        delta_sum=sum_ud,
        delta_pct=sum_up,
        control_median=compute_median(bu_sorted, presorted=True),
        candidate_median=compute_median(cu_sorted, presorted=True),
        delta_pct_median=u_med,
        delta_pct_mean=u_mean,
        improved_count=sum(1 for r in matched_records if r.uncov_delta < 0),
        regressed_count=sum(1 for r in matched_records if r.uncov_delta > 0),
        unchanged_count=sum(1 for r in matched_records if r.uncov_delta == 0),
        active_count=active_uncov_count,
    )

    # Quadrants
    quadrants = QuadrantSummary(
        fast_cheap=sum(1 for r in matched_records if r.time_delta_s < 0 and r.cost_delta <= 0),
        fast_expensive=sum(1 for r in matched_records if r.time_delta_s < 0 and r.cost_delta > 0),
        slow_cheap=sum(1 for r in matched_records if r.time_delta_s >= 0 and r.cost_delta <= 0),
        slow_expensive=sum(1 for r in matched_records if r.time_delta_s >= 0 and r.cost_delta > 0),
        total=len(matched_records),
    )

    # Histograms
    time_hist = compute_histogram(tp_vals, TIME_HIST_BINS, TIME_HIST_LABELS, TIME_HIST_COLORS)
    cost_hist = compute_histogram(cp_vals, COST_HIST_BINS, COST_HIST_LABELS, COST_HIST_COLORS)
    uncov_hist = compute_histogram(up_active, COST_HIST_BINS, COST_HIST_LABELS, COST_HIST_COLORS)

    # Quantiles table
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    quantiles: List[QuantileRow] = []
    for p in percentiles:
        quantiles.append(
            QuantileRow(
                percentile=p,
                label=f"P{p:02d}" + (" (Median)" if p == 50 else ""),
                is_median=(p == 50),
                control_time_s=compute_percentile(bt_sorted, p, presorted=True),
                candidate_time_s=compute_percentile(ct_sorted, p, presorted=True),
                time_delta_s=compute_percentile(td_sorted, p, presorted=True),
                time_delta_pct=compute_percentile(tp_sorted, p, presorted=True),
                control_cost=compute_percentile(bc_sorted, p, presorted=True),
                candidate_cost=compute_percentile(cc_sorted, p, presorted=True),
                cost_delta=compute_percentile(cd_sorted, p, presorted=True),
                cost_delta_pct=compute_percentile(cp_sorted, p, presorted=True),
                control_uncov=compute_percentile(bu_sorted, p, presorted=True),
                candidate_uncov=compute_percentile(cu_sorted, p, presorted=True),
                uncov_delta=compute_percentile(ud_sorted, p, presorted=True),
                uncov_delta_pct=compute_percentile(up_active, p, presorted=True) if up_active else None,
            )
        )

    # Runtime tiers
    tiers_def: list[tuple[str, Callable[[MatchedFontRecord], bool]]] = [
        ("Fast (< 10s)", lambda r: r.control_time_s < 10.0),
        ("Medium (10s - 60s)", lambda r: 10.0 <= r.control_time_s < 60.0),
        ("Slow (60s - 300s)", lambda r: 60.0 <= r.control_time_s < 300.0),
        ("Very Slow (>= 300s)", lambda r: r.control_time_s >= 300.0),
    ]
    runtime_tiers: List[RuntimeTierRow] = []
    for t_name, pred in tiers_def:
        sub = [r for r in matched_records if pred(r)]
        if not sub:
            continue
        t_bt = sum(r.control_time_s for r in sub)
        t_ct = sum(r.candidate_time_s for r in sub)
        t_td = t_ct - t_bt
        t_tp = (t_td / t_bt * 100.0) if t_bt > 0 else 0.0
        med_tp = compute_median(sorted(r.time_delta_pct for r in sub), presorted=True)
        med_cp = compute_median(sorted(r.cost_delta_pct for r in sub), presorted=True)
        sub_u = sorted(r.uncov_delta_pct for r in sub if r.uncov_delta_pct is not None)
        med_up = compute_median(sub_u, presorted=True) if sub_u else None
        runtime_tiers.append(
            RuntimeTierRow(
                name=t_name,
                count=len(sub),
                control_time_sum=t_bt,
                candidate_time_sum=t_ct,
                time_delta_sum=t_td,
                time_delta_pct=t_tp,
                median_time_delta_pct=med_tp,
                median_cost_delta_pct=med_cp,
                median_uncov_delta_pct=med_up,
            )
        )

    if quality_filter is not None:
        quality_mode_desc = f"Filtered to Quality Level {quality_filter}"
    elif quality_mode == "common":
        quality_mode_desc = "Highest Common Completed Quality Level per Font"
    else:
        quality_mode_desc = "Highest Non-Timeout Candidate Quality per Font (Matched to Equivalent Control Quality)"

    return ComparisonReport(
        dataset_name=dataset_name,
        notes_text=notes_text,
        control_trial=control_trial,
        candidate_trial=candidate_trial,
        quality_filter=quality_filter,
        quality_mode_desc=quality_mode_desc,
        quality_counts=quality_counts,
        time_stats=time_stats,
        cost_stats=cost_stats,
        uncov_stats=uncov_stats,
        quadrants=quadrants,
        time_histogram=time_hist,
        cost_histogram=cost_hist,
        uncov_histogram=uncov_hist,
        quantiles=quantiles,
        runtime_tiers=runtime_tiers,
        matched_records=matched_records,
        unmatched_candidate_only=unmatched_cand,
        unmatched_control_only=unmatched_ctrl,
        timeouts=timeouts,
    )


# ============================================================================
# Output Converters (Text & HTML)
# ============================================================================


def render_ascii_bar(pct: float, max_pct: float, width: int = 30) -> str:
    """Renders a simple ASCII progress bar."""
    if max_pct <= 0:
        return ""
    bar_len = int(round((pct / max_pct) * width))
    bar_len = max(0, min(width, bar_len))
    return "█" * bar_len + " " * (width - bar_len)


def render_text_report(report: ComparisonReport, top_n: int = 10) -> str:
    """Renders the ComparisonReport as a cleanly formatted terminal text document."""
    lines: list[str] = []
    sep = "=" * 80
    subsep = "-" * 80

    lines.append(sep)
    lines.append(f" IFT SEGMENTER COMPARISON REPORT")
    lines.append(f" Control:   {report.control_trial}")
    lines.append(f" Candidate: {report.candidate_trial}")
    lines.append(f" Dataset:   {report.dataset_name} ({report.total_matched_fonts:,} matched fonts)")
    lines.append(f" Quality:   {report.quality_mode_desc}")
    q_breakdown = ", ".join(f"Q{q}: {cnt:,}" for q, cnt in report.quality_counts.items())
    lines.append(f" Matched:   {q_breakdown}")
    if report.notes_text:
        lines.append(f" Notes:     {report.notes_text.splitlines()[0]}")
    lines.append(sep)
    lines.append("")

    # Summary KPIs
    lines.append("SUMMARY METRICS")
    lines.append(subsep)

    # Time
    t = report.time_stats
    lines.append(
        f" Segmentation Time:    {t.control_sum/3600.0:8.2f}h -> {t.candidate_sum/3600.0:8.2f}h "
        f"| Net: {t.delta_sum/3600.0:+7.2f}h ({t.delta_pct:+6.2f}%) "
        f"| Median: {t.delta_pct_median:+6.2f}% | Mean: {t.delta_pct_mean:+6.2f}%"
    )
    lines.append(
        f"   Direction:          {t.improved_count:,} faster ({t.improved_count/t.active_count*100.0:5.1f}%), "
        f"{t.regressed_count:,} slower ({t.regressed_count/t.active_count*100.0:5.1f}%), "
        f"{t.unchanged_count:,} unchanged"
    )

    # Cost
    c = report.cost_stats
    lines.append(
        f" Total IFT Cost:       {c.control_sum/1e6:8.2f}M -> {c.candidate_sum/1e6:8.2f}M "
        f"| Net: {c.delta_sum/1e6:+7.2f}M ({c.delta_pct:+6.2f}%) "
        f"| Median: {c.delta_pct_median:+6.2f}% | Mean: {c.delta_pct_mean:+6.2f}%"
    )
    lines.append(
        f"   Direction:          {c.improved_count:,} lower ({c.improved_count/c.active_count*100.0:5.1f}%), "
        f"{c.regressed_count:,} higher ({c.regressed_count/c.active_count*100.0:5.1f}%), "
        f"{c.unchanged_count:,} unchanged"
    )

    # Uncovered
    u = report.uncov_stats
    net_u_pct_str = f"({u.delta_pct:+6.2f}%)" if u.delta_pct is not None else "(   n/a)"
    med_u_pct_str = f"{u.delta_pct_median:+6.2f}%" if u.delta_pct_median is not None else "   n/a"
    mean_u_pct_str = f"{u.delta_pct_mean:+6.2f}%" if u.delta_pct_mean is not None else "   n/a"
    lines.append(
        f" Uncovered Patch Cost: {u.control_sum/1e6:8.2f}M -> {u.candidate_sum/1e6:8.2f}M "
        f"| Net: {u.delta_sum/1e6:+7.2f}M {net_u_pct_str} "
        f"| Median: {med_u_pct_str} | Mean: {mean_u_pct_str}"
    )
    if u.active_count > 0:
        lines.append(
            f"   Direction:          {u.improved_count:,} lower ({u.improved_count/u.active_count*100.0:5.1f}%), "
            f"{u.regressed_count:,} higher ({u.regressed_count/u.active_count*100.0:5.1f}%), "
            f"{u.unchanged_count:,} unchanged (active fonts: {u.active_count:,})"
        )
    else:
        lines.append(
            f"   Direction:          0 lower (  0.0%), 0 higher (  0.0%), "
            f"{len(report.matched_records):,} unchanged (active fonts: 0)"
        )

    # Timeouts
    to = report.timeouts
    lines.append(
        f" Timeouts & Errors:    {to.control_count:8d}  -> {to.candidate_count:8d}  "
        f"| Net: {to.diff_count:+7d} "
        f"| Resolved: {to.resolved_count} | New: {to.new_count} | Persistent: {to.persistent_count} | Missing: {to.missing_count}"
    )
    lines.append("")

    # Unmatched candidate notices
    if report.unmatched_candidate_only:
        lines.append(
            f" [!] {len(report.unmatched_candidate_only)} font(s) completed in candidate at selected quality "
            f"with no matching completed record in control (excluded from paired deltas)."
        )
        for unrec in report.unmatched_candidate_only[:5]:
            ctrl_q_str = f"Q{unrec.other_max_quality}" if unrec.other_max_quality is not None else "none (timed out/missing)"
            lines.append(
                f"     - {unrec.font_name} (Candidate Q{unrec.selected_quality}: {unrec.record.total_time_s:.2f}s, "
                f"Control: {ctrl_q_str})"
            )
        if len(report.unmatched_candidate_only) > 5:
            lines.append(f"     ... and {len(report.unmatched_candidate_only) - 5} more")
        lines.append("")

    # Unmatched control notices
    if report.unmatched_control_only:
        lines.append(
            f" [!] {len(report.unmatched_control_only)} font(s) completed in control at selected quality "
            f"with no matching completed record in candidate (excluded from paired deltas)."
        )
        for unrec in report.unmatched_control_only[:5]:
            cand_q_str = f"Q{unrec.other_max_quality}" if unrec.other_max_quality is not None else "none (timed out/missing)"
            lines.append(
                f"     - {unrec.font_name} (Control Q{unrec.selected_quality}: {unrec.record.total_time_s:.2f}s, "
                f"Candidate: {cand_q_str})"
            )
        if len(report.unmatched_control_only) > 5:
            lines.append(f"     ... and {len(report.unmatched_control_only) - 5} more")
        lines.append("")

    # Quadrants
    q = report.quadrants
    lines.append("JOINT OUTCOME QUADRANTS (TIME vs. COST)")
    lines.append(subsep)
    lines.append(f" 1. Win-Win (Faster & <= Cost):       {q.fast_cheap:6,} fonts ({q.fast_cheap_pct:6.2f}%) [Strict win]")
    lines.append(f" 2. Faster & Higher Cost:             {q.fast_expensive:6,} fonts ({q.fast_exp_pct:6.2f}%) [Runtime trade-off]")
    lines.append(f" 3. Slower/Same Time & <= Cost:       {q.slow_cheap:6,} fonts ({q.slow_cheap_pct:6.2f}%) [Cost trade-off]")
    lines.append(f" 4. Slower/Same Time & Higher Cost:   {q.slow_expensive:6,} fonts ({q.slow_exp_pct:6.2f}%) [Strict regression]")
    lines.append("")

    # Quantile Distribution Table
    lines.append("QUANTILE DISTRIBUTION COMPARISON")
    lines.append(subsep)
    lines.append(
        f"{'Quantile':<15} {'Control Time':>12} {'Cand Time':>12} {'Time Delta %':>15} "
        f"{'Cost Delta %':>15} {'Uncov Delta %':>15}"
    )
    lines.append("-" * 87)
    for qrow in report.quantiles:
        u_str = f"{qrow.uncov_delta_pct:>+14.2f}%" if qrow.uncov_delta_pct is not None else f"{'n/a':>15}"
        lines.append(
            f"{qrow.label:<15} {qrow.control_time_s:>11.2f}s {qrow.candidate_time_s:>11.2f}s "
            f"{qrow.time_delta_pct:>+14.2f}% {qrow.cost_delta_pct:>+14.2f}% {u_str}"
        )
    lines.append("")

    # Runtime Tiers
    lines.append("BREAKDOWN BY CONTROL RUNTIME TIER")
    lines.append(subsep)
    lines.append(
        f"{'Tier':<20} {'Fonts':>7} {'Control Time':>13} {'Cand Time':>13} {'Time Delta':>16} "
        f"{'Med Time Δ%':>12} {'Med Cost Δ%':>12} {'Med Uncov Δ%':>13}"
    )
    lines.append("-" * 111)
    for tr in report.runtime_tiers:
        td_str = f"{tr.time_delta_pct:+.2f}% ({tr.time_delta_sum:+,.1f}s)"
        u_str = f"{tr.median_uncov_delta_pct:>+12.2f}%" if tr.median_uncov_delta_pct is not None else f"{'n/a':>13}"
        lines.append(
            f"{tr.name:<20} {tr.count:>7,d} {tr.control_time_sum:>12.1f}s {tr.candidate_time_sum:>12.1f}s "
            f"{td_str:>16} {tr.median_time_delta_pct:>+11.2f}% {tr.median_cost_delta_pct:>+11.2f}% {u_str}"
        )
    lines.append("")

    # Histograms
    lines.append("DISTRIBUTION OF SEGMENTATION TIME % CHANGE")
    lines.append(subsep)
    max_t_pct = max((b.pct for b in report.time_histogram), default=1.0)
    for b in report.time_histogram:
        bar = render_ascii_bar(b.pct, max_t_pct, width=25)
        lines.append(f" {b.label:<33} [{bar}] {b.count:5,d} ({b.pct:5.1f}%)")
    lines.append("")

    lines.append("DISTRIBUTION OF IFT COST % CHANGE")
    lines.append(subsep)
    max_c_pct = max((b.pct for b in report.cost_histogram), default=1.0)
    for b in report.cost_histogram:
        bar = render_ascii_bar(b.pct, max_c_pct, width=25)
        lines.append(f" {b.label:<33} [{bar}] {b.count:5,d} ({b.pct:5.1f}%)")
    lines.append("")

    # Timeouts Summary Table
    if report.timeouts.per_quality and len(report.timeouts.per_quality) > 1:
        lines.append("TIMEOUTS & ERRORS PER QUALITY LEVEL")
        lines.append(subsep)
        lines.append(
            f"{'Quality':<10} {'Control':>9} {'Candidate':>11} {'Net Change':>12} "
            f"{'Resolved':>10} {'New':>8} {'Persistent':>12} {'Missing':>10}"
        )
        lines.append("-" * 86)
        for pq in report.timeouts.per_quality:
            lines.append(
                f"Quality {pq.quality_level:<2} {pq.control_count:>9d} {pq.candidate_count:>11d} "
                f"{pq.diff_count:>+12d} {pq.resolved_count:>10d} {pq.new_count:>8d} "
                f"{pq.persistent_count:>12d} {pq.missing_count:>10d}"
            )
        lines.append("")

    # Timeouts detail
    non_persistent_to = [r for r in report.timeouts.rows if r.status_key != "persistent"]
    if non_persistent_to:
        lines.append(f"RESOLVED, NEW & MISSING TIMEOUTS/ERRORS ({len(non_persistent_to)} occurrences)")
        lines.append(subsep)
        for to_row in non_persistent_to[:15]:
            lines.append(
                f" [{to_row.status:<18}] {to_row.font_name} (Q{to_row.quality_level})\n"
                f"   Control:   {to_row.control_desc}\n"
                f"   Candidate: {to_row.candidate_desc}"
            )
        if len(non_persistent_to) > 15:
            lines.append(f" ... and {len(non_persistent_to) - 15} more (see HTML report for complete list)")
        lines.append("")

    # Top Outliers (if top_n > 0)
    if top_n > 0:
        # Largest Speedups
        speedups = sorted([r for r in report.matched_records if r.time_delta_s < 0], key=lambda r: r.time_delta_s)[:top_n]
        if speedups:
            lines.append(f"TOP {len(speedups)} LARGEST TIME SPEEDUPS (by absolute Δs)")
            lines.append(subsep)
            lines.append(f"{'Font':<35} {'Q':>2} {'Control':>10} {'Candidate':>10} {'Delta (s)':>12} {'Delta (%)':>11} {'Cost Δ%':>10}")
            lines.append("-" * 94)
            for r in speedups:
                lines.append(
                    f"{r.font_name[:34]:<35} Q{r.quality_level:<1} {r.control_time_s:>9.2f}s {r.candidate_time_s:>9.2f}s "
                    f"{r.time_delta_s:>+11.2f}s {r.time_delta_pct:>+10.2f}% {r.cost_delta_pct:>+9.2f}%"
                )
            lines.append("")

        # Largest Slowdowns
        slowdowns = sorted([r for r in report.matched_records if r.time_delta_s > 0], key=lambda r: r.time_delta_s, reverse=True)[:top_n]
        if slowdowns:
            lines.append(f"TOP {len(slowdowns)} LARGEST TIME SLOWDOWNS (by absolute Δs)")
            lines.append(subsep)
            lines.append(f"{'Font':<35} {'Q':>2} {'Control':>10} {'Candidate':>10} {'Delta (s)':>12} {'Delta (%)':>11} {'Cost Δ%':>10}")
            lines.append("-" * 94)
            for r in slowdowns:
                lines.append(
                    f"{r.font_name[:34]:<35} Q{r.quality_level:<1} {r.control_time_s:>9.2f}s {r.candidate_time_s:>9.2f}s "
                    f"{r.time_delta_s:>+11.2f}s {r.time_delta_pct:>+10.2f}% {r.cost_delta_pct:>+9.2f}%"
                )
            lines.append("")

        # Largest Cost Reductions
        cost_drops = sorted([r for r in report.matched_records if r.cost_delta_pct < 0], key=lambda r: r.cost_delta_pct)[:top_n]
        if cost_drops:
            lines.append(f"TOP {len(cost_drops)} LARGEST IFT COST REDUCTIONS (by Δ%)")
            lines.append(subsep)
            lines.append(f"{'Font':<35} {'Q':>2} {'Control Cost':>13} {'Cand Cost':>13} {'Cost Δ%':>11} {'Time Δ%':>11}")
            lines.append("-" * 88)
            for r in cost_drops:
                lines.append(
                    f"{r.font_name[:34]:<35} Q{r.quality_level:<1} {r.control_cost:>13,d} {r.candidate_cost:>13,d} "
                    f"{r.cost_delta_pct:>+10.2f}% {r.time_delta_pct:>+10.2f}%"
                )
            lines.append("")

        # Largest Cost Increases
        cost_jumps = sorted([r for r in report.matched_records if r.cost_delta_pct > 0], key=lambda r: r.cost_delta_pct, reverse=True)[:top_n]
        if cost_jumps:
            lines.append(f"TOP {len(cost_jumps)} LARGEST IFT COST INCREASES (by Δ%)")
            lines.append(subsep)
            lines.append(f"{'Font':<35} {'Q':>2} {'Control Cost':>13} {'Cand Cost':>13} {'Cost Δ%':>11} {'Time Δ%':>11}")
            lines.append("-" * 88)
            for r in cost_jumps:
                lines.append(
                    f"{r.font_name[:34]:<35} Q{r.quality_level:<1} {r.control_cost:>13,d} {r.candidate_cost:>13,d} "
                    f"{r.cost_delta_pct:>+10.2f}% {r.time_delta_pct:>+10.2f}%"
                )
            lines.append("")

    return "\n".join(lines)


def _get_template_content() -> str:
    """Finds and reads report_template.html from disk or Bazel runfiles."""
    candidates = [
        Path(__file__).resolve().parent / "report_template.html",
    ]
    if "PYTHON_RUNFILES" in os.environ:
        candidates.append(Path(os.environ["PYTHON_RUNFILES"]) / "_main" / "report_template.html")
    if "TEST_SRCDIR" in os.environ:
        candidates.append(Path(os.environ["TEST_SRCDIR"]) / "_main" / "report_template.html")

    for cand in candidates:
        if cand.exists():
            return cand.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Could not locate report_template.html. Searched: {[str(c) for c in candidates]}"
    )


def render_html_report(report: ComparisonReport) -> str:
    """Renders the ComparisonReport as an interactive standalone HTML dashboard."""
    template_content = _get_template_content()
    title = f"IFT Segmenter Comparison: {report.control_trial} vs {report.candidate_trial}"
    json_blob = json.dumps(report.to_js_dict()).replace("</", "<\\/")

    html_out = template_content.replace("__TITLE__", html.escape(title))
    html_out = html_out.replace("__DATA__", json_blob)
    return html_out


# ============================================================================
# CLI Entrypoint
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two trials from an IFT segmenter dataset (out.scsv + error_log.txt)."
    )
    parser.add_argument(
        "dataset",
        help="Path to dataset directory (containing out.scsv) or out.scsv file",
    )
    parser.add_argument(
        "control",
        help="Name of the control (baseline) trial",
    )
    parser.add_argument(
        "candidate",
        help="Name of the candidate trial",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=None,
        help="Optional quality level filter (e.g. 1..7). If omitted, quality selection is governed by --quality-mode.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["common", "candidate"],
        default="common",
        help=(
            "Rule for selecting quality level per font when --quality is omitted. "
            "'common' (default): compares at the highest quality level where BOTH candidate and control completed successfully. "
            "'candidate': selects each font's highest completed quality in candidate and compares against the same quality in control."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "html"],
        default=None,
        help="Report format ('text' or 'html'). Inferred from --html/--text or -o extension if omitted (defaults to text).",
    )
    parser.add_argument(
        "--html",
        dest="format_html",
        action="store_true",
        help="Generate an HTML report (shorthand for --format=html)",
    )
    parser.add_argument(
        "--text",
        dest="format_text",
        action="store_true",
        help="Generate a text report (shorthand for --format=text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. In text mode, defaults to stdout. In HTML mode, defaults to comparison_<control>_vs_<candidate>[_q<N>].html",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of outliers to display per category in text mode (default: 10). Set 0 to disable.",
    )

    args = parser.parse_args(argv)

    # Support running under `bazel run` by resolving paths relative to BUILD_WORKING_DIRECTORY
    build_working_dir = os.environ.get("BUILD_WORKING_DIRECTORY")
    dataset_path_str = args.dataset
    if build_working_dir and not os.path.isabs(dataset_path_str):
        dataset_path_str = os.path.join(build_working_dir, dataset_path_str)

    output_path_str = args.output
    if output_path_str and build_working_dir and not os.path.isabs(output_path_str):
        output_path_str = os.path.join(build_working_dir, output_path_str)

    # Determine report format
    target_format = args.format
    if args.format_html:
        target_format = "html"
    elif args.format_text:
        target_format = "text"

    if target_format is None:
        if output_path_str and (output_path_str.endswith(".html") or output_path_str.endswith(".htm")):
            target_format = "html"
        else:
            target_format = "text"

    try:
        files = resolve_dataset(dataset_path_str)
        data, available_trials, available_qualities = load_scsv(files.scsv_path)
        errors = parse_error_log(files.error_log_path)
        report = build_report(
            dataset_name=files.dataset_name,
            notes_text=files.notes_text,
            data=data,
            errors=errors,
            available_qualities=available_qualities,
            control_trial=args.control,
            candidate_trial=args.candidate,
            quality_filter=args.quality,
            quality_mode=args.quality_mode,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if target_format == "html":
        if output_path_str:
            out_file = Path(output_path_str)
        else:
            q_suffix = f"_q{args.quality}" if args.quality is not None else ""
            default_filename = f"comparison_{args.control}_vs_{args.candidate}{q_suffix}.html"
            out_file = Path(os.path.join(build_working_dir, default_filename) if build_working_dir else default_filename)

        html_content = render_html_report(report)
        out_file.write_text(html_content, encoding="utf-8")
        print(f"Comparison: {args.control} vs {args.candidate} ({report.quality_mode_desc})")
        print(f"Matched:    {report.total_matched_fonts:,} fonts ({report.quality_counts})")
        print(f"Generated HTML report: {out_file.resolve()}")
    else:
        text_content = render_text_report(report, top_n=args.top_n)
        if output_path_str:
            out_file = Path(output_path_str)
            out_file.write_text(text_content, encoding="utf-8")
            print(f"Generated text report: {out_file.resolve()}")
        else:
            print(text_content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
