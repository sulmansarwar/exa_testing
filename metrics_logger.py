"""Persistence helpers for Exa testing runs.

This module is intentionally UI-free (no Streamlit imports). It is used by app.py to
save/load run artifacts and to attach derived artifacts like `claims_report` to a
previously-saved run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Directory where run JSON artifacts are stored.
# Can be overridden via env var to support different local setups.
_RESULTS_DIR = Path(os.environ.get("EXA_TESTING_RESULTS_DIR", "saved_runs")).resolve()


def _ensure_results_dir() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    # Keep filenames simple and portable.
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))


def _result_path(result_id: str) -> Path:
    _ensure_results_dir()
    rid = _safe_filename(result_id)
    if not rid.endswith(".json"):
        rid = f"{rid}.json"
    return _RESULTS_DIR / rid


def save_result(result_id: str, result_data: Dict[str, Any]) -> Path:
    """Save a run artifact to disk as JSON.

    `result_id` should be stable (often a timestamp or UUID).
    """
    path = _result_path(result_id)
    # Add/refresh lightweight metadata for later listing.
    result_data = dict(result_data)
    result_data.setdefault("_meta", {})
    result_data["_meta"].update(
        {
            "result_id": result_id,
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )

    path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_result(result_id: str) -> Optional[Dict[str, Any]]:
    """Load a saved run JSON artifact. Returns None if not found or unreadable."""
    path = _result_path(result_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_saved_runs() -> List[Dict[str, Any]]:
    """Return saved runs as a list of dicts with id/label/path/mtime.

    The UI can choose to display `label` (query if available) and keep `id` as the value.
    """
    _ensure_results_dir()
    runs: List[Dict[str, Any]] = []

    for p in sorted(_RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        rid = p.stem
        label = rid
        query = None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Prefer a human label if present.
            query = (
                data.get("query")
                or data.get("prompt")
                or data.get("input", {}).get("query")
                or data.get("_meta", {}).get("query")
            )
            if isinstance(query, str) and query.strip():
                label = query.strip()
        except Exception:
            pass

        runs.append(
            {
                "id": rid,
                "label": label,
                "path": str(p),
                "mtime": p.stat().st_mtime,
            }
        )

    return runs


def save_claims_report(result_id: str, claims_report: Dict[str, Any]) -> bool:
    """Attach a computed claims_report to an existing saved run and persist it."""
    data = load_result(result_id)
    if not data:
        return False

    data["claims_report"] = claims_report
    save_result(result_id, data)
    return True


def save_fields(result_id: str, fields: Dict[str, Any]) -> bool:
    """Generic helper to attach/update fields in an existing result."""
    data = load_result(result_id)
    if not data:
        return False

    for k, v in fields.items():
        data[k] = v
    save_result(result_id, data)
    return True



import csv
import os
import json
from datetime import datetime
from pathlib import Path
import hashlib
# Metrics file path
METRICS_FILE = Path(__file__).parent / "comparison_metrics.csv"

# Results storage directory
RESULTS_DIR = Path(__file__).parent / "comparison_results"
RESULTS_DIR.mkdir(exist_ok=True)

# CSV Headers
HEADERS = [
    'timestamp',
    'query',
    'query_category',
    'result_id',
    'app_type',
    'webset_category',
    'search_type',
    'num_results',

    # Exa metrics
    'exa_search_count',
    'exa_sources',
    'exa_domains',
    'exa_tier_a',
    'exa_tier_b',
    'exa_tier_c',
    'exa_traceability',
    'exa_supported',
    'exa_inferences',
    'exa_unsupported',
    'exa_tier_a_claims',
    'exa_tier_b_claims',
    'exa_tier_c_claims',
    'exa_tokens',
    'exa_cost',
    'exa_time_seconds',

    # Claude metrics
    'claude_sources',
    'claude_domains',
    'claude_tier_a',
    'claude_tier_b',
    'claude_tier_c',
    'claude_traceability',
    'claude_supported',
    'claude_inferences',
    'claude_unsupported',
    'claude_tier_a_claims',
    'claude_tier_b_claims',
    'claude_tier_c_claims',
    'claude_tokens',
    'claude_cost',
    'claude_time_seconds'
]

def log_comparison(
    query,
    app_type,
    exa_metrics,
    claude_metrics,
    exa_search_count=0,
    webset_category=None,
    search_type=None,
    num_results=None,
    query_category=None,
    exa_analysis_text=None,
    claude_analysis_text=None,
    exa_sources=None,
    claude_sources=None,
    exa_search_queries=None
):
    """
    Log a comparison to CSV and save full results to JSON

    Args:
        query: The research query
        app_type: 'general' or 'websets'
        exa_metrics: dict from calculate_retrieval_metrics()
        claude_metrics: dict from calculate_retrieval_metrics()
        exa_search_count: number of Exa searches performed
        webset_category: webset category if applicable
        search_type: Exa search type (auto/neural/keyword)
        num_results: number of results requested
        query_category: optional category label (e.g., 'Scientific', 'Business', 'Legal')
        exa_analysis_text: Full analysis text from Exa+Claude
        claude_analysis_text: Full analysis text from Claude Web Search
        exa_sources: List of source dicts from Exa
        claude_sources: List of source dicts from Claude Web Search
        exa_search_queries: List of search queries used by Exa
    """
    # Initialize CSV if it doesn't exist
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()

    # Generate result_id
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    result_id = f"{timestamp_str}_{query_hash}"

    # Save full results to JSON
    result_data = {
        'timestamp': timestamp.isoformat(),
        'query': query,
        'query_category': query_category,
        'app_type': app_type,
        'webset_category': webset_category,
        'search_type': search_type,
        'num_results': num_results,
        'exa': {
            'analysis_text': exa_analysis_text,
            'sources': exa_sources or [],
            'search_queries': exa_search_queries or [],
            'search_count': exa_search_count,
            'time_seconds': exa_metrics.get('time_seconds', 0)
        },
        'claude': {
            'analysis_text': claude_analysis_text,
            'sources': claude_sources or [],
            'search_count': 0,
            'time_seconds': claude_metrics.get('time_seconds', 0)
        },
        'metrics': {
            'exa_traceability': exa_metrics.get('traceability_rate', 0),
            'exa_supported': exa_metrics.get('supported_claims', 0),
            'exa_tier_a': exa_metrics['quality_mix']['A'],
            'exa_tier_a_pct': (exa_metrics['quality_mix']['A'] / exa_metrics['total_sources'] * 100) if exa_metrics['total_sources'] > 0 else 0,
            'exa_cost': round(exa_metrics['cost_usd'] + (exa_search_count / 1000 * 5.0), 4),
            'claude_traceability': claude_metrics.get('traceability_rate', 0),
            'claude_supported': claude_metrics.get('supported_claims', 0),
            'claude_tier_a': claude_metrics['quality_mix']['A'],
            'claude_tier_a_pct': (claude_metrics['quality_mix']['A'] / claude_metrics['total_sources'] * 100) if claude_metrics['total_sources'] > 0 else 0,
            'claude_cost': round(claude_metrics['cost_usd'], 4)
        }
    }

    result_file = RESULTS_DIR / f"{result_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    # Prepare row data
    row = {
        'timestamp': timestamp.isoformat(),
        'query': query,
        'query_category': query_category or '',
        'result_id': result_id,
        'app_type': app_type,
        'webset_category': webset_category or '',
        'search_type': search_type or '',
        'num_results': num_results or '',

        # Exa metrics
        'exa_search_count': exa_search_count,
        'exa_sources': exa_metrics['total_sources'],
        'exa_domains': exa_metrics['unique_domains'],
        'exa_tier_a': exa_metrics['quality_mix']['A'],
        'exa_tier_b': exa_metrics['quality_mix']['B'],
        'exa_tier_c': exa_metrics['quality_mix']['C'],
        'exa_traceability': exa_metrics.get('traceability_rate', 0),
        'exa_supported': exa_metrics.get('supported_claims', 0),
        'exa_inferences': exa_metrics.get('inferences', 0),
        'exa_unsupported': exa_metrics.get('unsupported_claims', 0),
        'exa_tier_a_claims': exa_metrics.get('tier_a_claims', 0),
        'exa_tier_b_claims': exa_metrics.get('tier_b_claims', 0),
        'exa_tier_c_claims': exa_metrics.get('tier_c_claims', 0),
        'exa_tokens': exa_metrics['tokens'],
        'exa_cost': round(exa_metrics['cost_usd'] + (exa_search_count / 1000 * 5.0), 4),
        'exa_time_seconds': round(exa_metrics.get('time_seconds', 0), 2),

        # Claude metrics
        'claude_sources': claude_metrics['total_sources'],
        'claude_domains': claude_metrics['unique_domains'],
        'claude_tier_a': claude_metrics['quality_mix']['A'],
        'claude_tier_b': claude_metrics['quality_mix']['B'],
        'claude_tier_c': claude_metrics['quality_mix']['C'],
        'claude_traceability': claude_metrics.get('traceability_rate', 0),
        'claude_supported': claude_metrics.get('supported_claims', 0),
        'claude_inferences': claude_metrics.get('inferences', 0),
        'claude_unsupported': claude_metrics.get('unsupported_claims', 0),
        'claude_tier_a_claims': claude_metrics.get('tier_a_claims', 0),
        'claude_tier_b_claims': claude_metrics.get('tier_b_claims', 0),
        'claude_tier_c_claims': claude_metrics.get('tier_c_claims', 0),
        'claude_tokens': claude_metrics['tokens'],
        'claude_cost': round(claude_metrics['cost_usd'], 4),
        'claude_time_seconds': round(claude_metrics.get('time_seconds', 0), 2)
    }

    # Append to CSV
    with open(METRICS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)

    return METRICS_FILE

def get_total_stats():
    """
    Get aggregate statistics from all logged comparisons
    Returns dict with totals and averages
    """
    if not METRICS_FILE.exists():
        return None

    with open(METRICS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    total_queries = len(rows)

    # Calculate cost totals and averages
    total_exa_cost = sum(float(r.get('exa_cost', 0)) for r in rows)
    total_claude_cost = sum(float(r.get('claude_cost', 0)) for r in rows)
    avg_exa_cost = total_exa_cost / total_queries
    avg_claude_cost = total_claude_cost / total_queries

    # Calculate traceability averages (percentages)
    avg_exa_traceability = sum(float(r.get('exa_traceability', 0)) for r in rows) / total_queries
    avg_claude_traceability = sum(float(r.get('claude_traceability', 0)) for r in rows) / total_queries

    # Calculate inference rate averages
    def calc_inference_rate(row, prefix):
        try:
            supported = int(float(row.get(f'{prefix}_supported', 0) or 0))
            inferences = int(float(row.get(f'{prefix}_inferences', 0) or 0))
            unsupported = int(float(row.get(f'{prefix}_unsupported', 0) or 0))
            total = supported + inferences + unsupported
            if total == 0:
                return 0
            return (inferences / total) * 100
        except (ValueError, TypeError):
            return 0

    avg_exa_inference_rate = sum(calc_inference_rate(r, 'exa') for r in rows) / total_queries
    avg_claude_inference_rate = sum(calc_inference_rate(r, 'claude') for r in rows) / total_queries

    # Calculate unsupported rate averages
    def calc_unsupported_rate(row, prefix):
        try:
            supported = int(float(row.get(f'{prefix}_supported', 0) or 0))
            inferences = int(float(row.get(f'{prefix}_inferences', 0) or 0))
            unsupported = int(float(row.get(f'{prefix}_unsupported', 0) or 0))
            total = supported + inferences + unsupported
            if total == 0:
                return 0
            return (unsupported / total) * 100
        except (ValueError, TypeError):
            return 0

    avg_exa_unsupported_rate = sum(calc_unsupported_rate(r, 'exa') for r in rows) / total_queries
    avg_claude_unsupported_rate = sum(calc_unsupported_rate(r, 'claude') for r in rows) / total_queries

    return {
        'total_queries': total_queries,
        'total_exa_cost': total_exa_cost,
        'total_claude_cost': total_claude_cost,
        'avg_exa_cost': avg_exa_cost,
        'avg_claude_cost': avg_claude_cost,
        'avg_exa_traceability': avg_exa_traceability,
        'avg_claude_traceability': avg_claude_traceability,
        'avg_exa_inference_rate': avg_exa_inference_rate,
        'avg_claude_inference_rate': avg_claude_inference_rate,
        'avg_exa_unsupported_rate': avg_exa_unsupported_rate,
        'avg_claude_unsupported_rate': avg_claude_unsupported_rate,
        'csv_path': str(METRICS_FILE)
    }

def get_detailed_history():
    """
    Get detailed comparison history as a list of dicts
    Returns list of all comparison rows for display
    """
    if not METRICS_FILE.exists():
        return None

    with open(METRICS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    return rows

def load_result(result_id):
    """
    Load full result data from JSON file by result_id

    Args:
        result_id: The result ID (timestamp_hash format)

    Returns:
        dict with full result data or None if not found
    """
    result_file = RESULTS_DIR / f"{result_id}.json"

    if not result_file.exists():
        return None

    with open(result_file, 'r') as f:
        return json.load(f)


import json
import os

CLAIMS_REPORT_FILENAME = "claims_report.json"

def _get_result_dir_for_id(result_id: str) -> str:
    """
    MUST match the directory logic used inside load_result(result_id).
    Copy/paste the same 'result_dir = ...' construction from load_result here.
    """
    rid = str(result_id).strip()
    # Example ONLY: replace with your real logic from load_result
    # result_dir = os.path.join(RESULTS_DIR, rid)
    return result_dir


def save_claims_report(result_id: str, report: dict) -> bool:
    """Persist claims report for a result_id to disk."""
    try:
        result_dir = _get_result_dir_for_id(result_id)
        os.makedirs(result_dir, exist_ok=True)
        path = os.path.join(result_dir, CLAIMS_REPORT_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_claims_report(result_id: str) -> dict | None:
    """Load persisted claims report for a result_id from disk."""
    try:
        result_dir = _get_result_dir_for_id(result_id)
        path = os.path.join(result_dir, CLAIMS_REPORT_FILENAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None