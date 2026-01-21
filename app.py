"""
Streamlit Web App - Exa + Claude vs Claude Web Search Comparison
Interactive prompt experimentation tool for newsroom verification
"""

import streamlit as st
import os
from dotenv import load_dotenv
from exa_py import Exa
from anthropic import Anthropic
from datetime import datetime
import json
import re
import time
import metrics_logger
from claims_analyzer import classify_source_quality,render_claims_report

from prompts import (
    build_exa_initial_prompt,
    build_web_search_prompt,
    build_exa_article_prompt,
    build_web_article_prompt,
    QUERY_CATEGORIES,
    get_example_prompts,
    get_measurement_guide_text,
)


# Load environment variables
#load_dotenv()

# Initialize clients
@st.cache_resource
def get_clients():
    exa = Exa(api_key=st.secrets["EXA_API_KEY"])
    anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    return exa, anthropic



exa, anthropic = get_clients()

# --- METRICS CSV/ROW HELPERS ---
def _resolve_metrics_csv_path() -> str:
    return getattr(metrics_logger, "CSV_PATH", "comparison_metrics.csv")


def _get_metrics_row_for_result(result_id: str, result_data: dict | None = None) -> dict | None:
    """Source of truth for saved-search metrics (tokens/cost/etc)."""
    rid = (str(result_id).strip() if result_id else "")
    qtxt = ((result_data or {}).get("query") or "").strip()

    # 1) Prefer metrics_logger.get_detailed_history() if it returns full CSV rows
    if hasattr(metrics_logger, "get_detailed_history"):
        try:
            rows = metrics_logger.get_detailed_history() or []
            if rid:
                for row in rows:
                    if str(row.get("result_id") or "").strip() == rid:
                        return row
            if qtxt:
                for row in reversed(rows):
                    if str(row.get("query") or "").strip() == qtxt:
                        return row
        except Exception:
            pass

    # 2) Fallback: read CSV directly
    try:
        import csv
        csv_path = _resolve_metrics_csv_path()
        if os.path.exists(csv_path):
            rows = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
                    if rid and str(row.get("result_id") or "").strip() == rid:
                        return row
            if qtxt:
                for row in reversed(rows):
                    if str(row.get("query") or "").strip() == qtxt:
                        return row
    except Exception:
        pass

    return None

# --- TOKEN USAGE EXTRACTOR (for demo/history and flexible saved runs) ---
def _extract_token_usage(block: dict, analysis_text: str = ""):
    """Best-effort extraction of (input_tokens, output_tokens) from a saved result block.

    Saved runs may differ in where token usage is stored.
    This function avoids API calls and keeps demo/history rendering accurate.
    """
    if not isinstance(block, dict):
        block = {}

    # Common direct fields
    in_tok = block.get("input_tokens")
    out_tok = block.get("output_tokens")

    # Nested usage dicts
    if (in_tok is None or out_tok is None) and isinstance(block.get("usage"), dict):
        u = block.get("usage") or {}
        in_tok = in_tok if in_tok is not None else u.get("input_tokens") or u.get("input")
        out_tok = out_tok if out_tok is not None else u.get("output_tokens") or u.get("output")

    if (in_tok is None or out_tok is None) and isinstance(block.get("token_usage"), dict):
        u = block.get("token_usage") or {}
        in_tok = in_tok if in_tok is not None else u.get("input_tokens") or u.get("input")
        out_tok = out_tok if out_tok is not None else u.get("output_tokens") or u.get("output")

    # Some saves may only include total tokens
    total = block.get("tokens")
    if (in_tok is None and out_tok is None) and isinstance(total, (int, float)):
        # Best effort: treat as input tokens (cost calc will still be approximate)
        in_tok = int(total)
        out_tok = 0

    # Final fallback: estimate from text length if everything is missing
    try:
        in_tok = int(in_tok or 0)
        out_tok = int(out_tok or 0)
    except Exception:
        in_tok, out_tok = 0, 0

    if (in_tok + out_tok) == 0 and isinstance(analysis_text, str) and analysis_text.strip():
        # Very rough: ~4 chars per token
        est_total = max(1, int(len(analysis_text) / 4))
        in_tok = est_total
        out_tok = 0


    return in_tok, out_tok


# --- Robust helper to extract token usage from Anthropic responses ---
def _get_usage_tokens(usage_obj) -> tuple[int, int]:
    """Best-effort extraction of (input_tokens, output_tokens) from Anthropic usage.

    Anthropic SDK usage may be an object with attributes or a dict.
    """
    if usage_obj is None:
        return 0, 0

    # Dict style
    if isinstance(usage_obj, dict):
        in_tok = usage_obj.get("input_tokens") or usage_obj.get("input") or 0
        out_tok = usage_obj.get("output_tokens") or usage_obj.get("output") or 0
        try:
            return int(in_tok or 0), int(out_tok or 0)
        except Exception:
            return 0, 0

    # Attribute style
    try:
        in_tok = getattr(usage_obj, "input_tokens", 0) or 0
        out_tok = getattr(usage_obj, "output_tokens", 0) or 0
        return int(in_tok), int(out_tok)
    except Exception:
        return 0, 0

# --- METRICS LOGGER COMPATIBILITY LAYER ---
def _safe_get_total_stats():
    """Return aggregate stats for history page.

    Prefers metrics_logger.get_total_stats() if available.
    Falls back to computing aggregates from saved results in comparison_results/*.json.
    """
    # 1) Prefer logger totals if they exist AND are non-empty
    if hasattr(metrics_logger, "get_total_stats"):
        try:
            stats = metrics_logger.get_total_stats()
            if stats and stats.get("total_queries", 0) > 0:
                return stats
        except Exception:
            pass

    # 2) Fallback: compute from saved JSON results
    folder = "comparison_results"
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".json")]
        files = sorted(files, reverse=True)
    except Exception:
        files = []

    if not files:
        return None

    rows = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        m = data.get("metrics") or {}
        exa_block = data.get("exa") or {}
        claude_block = data.get("claude") or {}

        # Pull from saved metrics if present
        exa_cost = m.get("exa_cost")
        claude_cost = m.get("claude_cost")
        exa_trace = m.get("exa_traceability")
        claude_trace = m.get("claude_traceability")
        exa_inf = m.get("exa_inference_rate")
        claude_inf = m.get("claude_inference_rate")
        exa_uns = m.get("exa_unsupported_rate")
        claude_uns = m.get("claude_unsupported_rate")

        # If missing, compute traceability from analysis_text + sources
        if exa_trace is None:
            t = calculate_evidence_traceability(
                exa_block.get("analysis_text", ""),
                exa_block.get("sources", [])
            )
            exa_trace = float(t.get("traceability_rate", 0.0) or 0.0)
            total_claims = max(1, int(t.get("total_claims", 0) or 0))
            exa_inf = float((int(t.get("inferences", 0) or 0) / total_claims) * 100)
            exa_uns = float((int(t.get("unsupported_claims", 0) or 0) / total_claims) * 100)

        if claude_trace is None:
            t = calculate_evidence_traceability(
                claude_block.get("analysis_text", ""),
                claude_block.get("sources", [])
            )
            claude_trace = float(t.get("traceability_rate", 0.0) or 0.0)
            total_claims = max(1, int(t.get("total_claims", 0) or 0))
            claude_inf = float((int(t.get("inferences", 0) or 0) / total_claims) * 100)
            claude_uns = float((int(t.get("unsupported_claims", 0) or 0) / total_claims) * 100)

        # If missing, compute costs from tokens + sources
        if exa_cost is None:
            exa_cost = float(
                calculate_retrieval_metrics(
                    exa_block.get("sources", []),
                    int(exa_block.get("input_tokens", 0) or 0),
                    int(exa_block.get("output_tokens", 0) or 0),
                ).get("cost_usd", 0.0)
            )

        if claude_cost is None:
            claude_cost = float(
                calculate_retrieval_metrics(
                    claude_block.get("sources", []),
                    int(claude_block.get("input_tokens", 0) or 0),
                    int(claude_block.get("output_tokens", 0) or 0),
                ).get("cost_usd", 0.0)
            )

        rows.append({
            "exa_cost": float(exa_cost or 0.0),
            "claude_cost": float(claude_cost or 0.0),
            "exa_traceability": float(exa_trace or 0.0),
            "claude_traceability": float(claude_trace or 0.0),
            "exa_inference_rate": float(exa_inf or 0.0),
            "claude_inference_rate": float(claude_inf or 0.0),
            "exa_unsupported_rate": float(exa_uns or 0.0),
            "claude_unsupported_rate": float(claude_uns or 0.0),
        })

    if not rows:
        return None

    def _avg(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    total_exa_cost = sum(r.get("exa_cost", 0.0) for r in rows)
    total_claude_cost = sum(r.get("claude_cost", 0.0) for r in rows)

    return {
        "total_queries": len(rows),
        "total_exa_cost": float(total_exa_cost),
        "total_claude_cost": float(total_claude_cost),
        "avg_exa_cost": float(_avg("exa_cost")),
        "avg_claude_cost": float(_avg("claude_cost")),
        "avg_exa_traceability": float(_avg("exa_traceability")),
        "avg_claude_traceability": float(_avg("claude_traceability")),
        "avg_exa_inference_rate": float(_avg("exa_inference_rate")),
        "avg_claude_inference_rate": float(_avg("claude_inference_rate")),
        "avg_exa_unsupported_rate": float(_avg("exa_unsupported_rate")),
        "avg_claude_unsupported_rate": float(_avg("claude_unsupported_rate")),
        "csv_path": getattr(metrics_logger, "CSV_PATH", "comparison_metrics.csv"),
    }


def _safe_get_detailed_history():
    """Return per-run history rows.

    Prefers metrics_logger.get_detailed_history() if available.
    Falls back to list_saved_runs() + load_result() if available.
    Finally, falls back to scanning comparison_results/*.json.
    """
    if hasattr(metrics_logger, "get_detailed_history"):
        try:
            return metrics_logger.get_detailed_history()
        except Exception:
            pass

    # Fallback: build history from saved runs if the logger supports it
    if hasattr(metrics_logger, "list_saved_runs") and hasattr(metrics_logger, "load_result"):
        rows = []
        try:
            runs = metrics_logger.list_saved_runs()
        except Exception:
            runs = []
        for r in runs:
            rid = r.get("id") or r.get("result_id")
            if not rid:
                continue
            try:
                data = metrics_logger.load_result(rid)
                # Auto-load saved claims report so saved searches behave like live runs
                if isinstance(result_data, dict) and not result_data.get("claims_report"):
                    if hasattr(metrics_logger, "load_claims_report"):
                        _cr = metrics_logger.load_claims_report(rid)
                        if _cr:
                            result_data["claims_report"] = _cr
            except Exception:
                data = None
            if not data:
                continue
            rows.append({
                "timestamp": data.get("timestamp") or r.get("timestamp") or "",
                "query": data.get("query") or r.get("label") or "",
                "result_id": rid,
            })
        if rows:
            return rows

    # Final fallback: scan comparison_results folder (this is where demo mode reads from)
    rows = []
    folder = "comparison_results"
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".json")]
        files = sorted(files, reverse=True)
    except Exception:
        files = []

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        rid = str(
            data.get("result_id")
            or fname.rsplit("_", 1)[-1].replace(".json", "")
        )
        q = (data.get("query") or "").strip()
        ts = (data.get("timestamp") or "").strip()
        if not ts:
            ts = fname.replace(".json", "")

        m = data.get("metrics") or {}
        exa_block = data.get("exa") or {}
        claude_block = data.get("claude") or {}

        rows.append({
            "timestamp": ts,
            "category": data.get("query_category") or "",
            "query": q,
            "run": 1,
            "exa_search_type": data.get("search_type") or "",
            "exa_search_count": int(exa_block.get("search_count", 0) or 0),
            "exa_tier_a_pct": float(m.get("exa_tier_a_pct", 0.0) or 0.0),
            "claude_tier_a_pct": float(m.get("claude_tier_a_pct", 0.0) or 0.0),
            "exa_traceability": float(m.get("exa_traceability", 0.0) or 0.0),
            "claude_traceability": float(m.get("claude_traceability", 0.0) or 0.0),
            "exa_cost": float(m.get("exa_cost", 0.0) or 0.0),
            "claude_cost": float(m.get("claude_cost", 0.0) or 0.0),
            "exa_time_seconds": float(exa_block.get("time_seconds", 0.0) or 0.0),
            "claude_time_seconds": float(claude_block.get("time_seconds", 0.0) or 0.0),
            "result_id": rid,
        })

    return rows


def _safe_load_result(result_id: str):
    """Load a saved run by result_id.

    Tries metrics_logger.load_result() first. Falls back to scanning
    comparison_results/*.json for a filename containing the id.
    """
    rid = (str(result_id).strip() if result_id is not None else "")
    if not rid:
        return None

    # 1) Preferred: metrics_logger.load_result(result_id)
    if hasattr(metrics_logger, "load_result"):
        try:
            data = metrics_logger.load_result(rid)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    # 2) Fallback: scan comparison_results directory
    try:
        base_dir = "comparison_results"
        if not os.path.isdir(base_dir):
            return None

        candidates = []
        for fname in os.listdir(base_dir):
            if not fname.endswith(".json"):
                continue
            # Common pattern: ..._{result_id}.json
            if fname.endswith(f"_{rid}.json") or (rid in fname):
                candidates.append(os.path.join(base_dir, fname))

        if not candidates:
            return None

        # Prefer exact suffix match, then newest file
        def _rank(p: str):
            f = os.path.basename(p)
            exact = 1 if f.endswith(f"_{rid}.json") else 0
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            return (exact, mtime)

        candidates.sort(key=_rank, reverse=True)
        chosen = candidates[0]

        with open(chosen, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --- Ensure loaded result metrics (for history view, fills missing fields from analysis text/sources) ---
def _ensure_loaded_result_metrics(result_data: dict) -> dict:
    """Ensure result_data['metrics'] contains the fields used by the History view.

    Older saved JSONs may be missing inference/unsupported rates or tier-A percentages.
    This fills them in from the saved analysis text + sources (no API calls).
    """
    if not result_data or not isinstance(result_data, dict):
        return result_data

    result_data.setdefault("metrics", {})
    m = result_data["metrics"]

    exa_block = result_data.get("exa") or {}
    claude_block = result_data.get("claude") or {}

    exa_sources = exa_block.get("sources") or []
    claude_sources = claude_block.get("sources") or []

    exa_text = exa_block.get("analysis_text") or ""
    claude_text = claude_block.get("analysis_text") or ""

    # Traceability-derived rates
    exa_t = calculate_evidence_traceability(exa_text, exa_sources)
    claude_t = calculate_evidence_traceability(claude_text, claude_sources)

    def _rate(n: int, d: int) -> float:
        d = max(1, int(d or 0))
        return round((float(n or 0) / d) * 100.0, 1)

    # Ensure traceability % exists
    if m.get("exa_traceability") is None:
        m["exa_traceability"] = float(exa_t.get("traceability_rate", 0.0) or 0.0)
    if m.get("claude_traceability") is None:
        m["claude_traceability"] = float(claude_t.get("traceability_rate", 0.0) or 0.0)

    # Ensure inference/unsupported rates exist
    if m.get("exa_inference_rate") is None:
        m["exa_inference_rate"] = _rate(exa_t.get("inferences", 0), exa_t.get("total_claims", 0))
    if m.get("claude_inference_rate") is None:
        m["claude_inference_rate"] = _rate(claude_t.get("inferences", 0), claude_t.get("total_claims", 0))

    if m.get("exa_unsupported_rate") is None:
        m["exa_unsupported_rate"] = _rate(exa_t.get("unsupported_claims", 0), exa_t.get("total_claims", 0))
    if m.get("claude_unsupported_rate") is None:
        m["claude_unsupported_rate"] = _rate(claude_t.get("unsupported_claims", 0), claude_t.get("total_claims", 0))

    # Ensure Tier-A % exists (source quality mix)
    def _tier_a_pct(sources: list) -> float:
        try:
            total = len(sources)
            if total <= 0:
                return 0.0
            a = 0
            for s in sources:
                url = (s or {}).get("url")
                if url and classify_source_quality(url) == "A":
                    a += 1
            return round((a / total) * 100.0, 1)
        except Exception:
            return 0.0

    if m.get("exa_tier_a_pct") is None:
        m["exa_tier_a_pct"] = _tier_a_pct(exa_sources)
    if m.get("claude_tier_a_pct") is None:
        m["claude_tier_a_pct"] = _tier_a_pct(claude_sources)

    return result_data

# --- SCORECARD METRICS NORMALIZER (prevents KeyError across live/demo/history objects) ---
def _normalize_scorecard_metrics(metrics: dict | None, sources: list | None = None) -> dict:
    """Return a consistent metrics dict for the scorecard.

    The app has multiple pathways (live run, demo saved JSON, history CSV). Some paths
    produce slightly different key names. This normalizer makes the UI resilient.

    Expected output keys:
      - quality_mix: {'A': int, 'B': int, 'C': int}
      - unique_domains: int
      - total_sources: int
      - input_tokens: int
      - output_tokens: int
      - tokens: int
      - cost_usd: float
    """
    if not isinstance(metrics, dict):
        metrics = {}

    def pick(*keys, default=None):
        for k in keys:
            if k in metrics and metrics.get(k) is not None:
                return metrics.get(k)
        return default

    # Source counts
    total_sources = pick("total_sources", "sources_total", "num_sources", default=None)
    if total_sources is None:
        # fall back to sources list length if provided
        total_sources = len(sources or [])

    # Quality mix
    qm = pick("quality_mix", "quality", "quality_counts", "tier_counts", default=None)
    if not isinstance(qm, dict):
        qm = {}
    quality_mix = {
        "A": int(qm.get("A", qm.get("tier_a", 0)) or 0),
        "B": int(qm.get("B", qm.get("tier_b", 0)) or 0),
        "C": int(qm.get("C", qm.get("tier_c", 0)) or 0),
    }

    # Domains
    unique_domains = pick("unique_domains", "domains", "unique_domain_count", default=0)
    try:
        unique_domains = int(unique_domains or 0)
    except Exception:
        unique_domains = 0

    # Token usage
    input_tokens = pick("input_tokens", "in_tokens", "prompt_tokens", default=0)
    output_tokens = pick("output_tokens", "out_tokens", "completion_tokens", default=0)
    total_tokens = pick("tokens", "total_tokens", default=None)

    try:
        input_tokens = int(input_tokens or 0)
    except Exception:
        input_tokens = 0
    try:
        output_tokens = int(output_tokens or 0)
    except Exception:
        output_tokens = 0

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    try:
        total_tokens = int(total_tokens or 0)
    except Exception:
        total_tokens = 0

    # Cost
    cost_usd = pick("cost_usd", "cost", "total_cost", "usd_cost", default=0.0)
    try:
        cost_usd = float(cost_usd or 0.0)
    except Exception:
        cost_usd = 0.0

    # Final normalized dict
    return {
        "quality_mix": quality_mix,
        "unique_domains": unique_domains,
        "total_sources": int(total_sources or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": total_tokens,
        "cost_usd": cost_usd,
    }

# --- HELPER FUNCTIONS FOR METRICS ---
def calculate_retrieval_metrics(sources, input_tokens, output_tokens):
    """
    Calculate retrieval quality metrics
    Returns dict with quality mix, independence, coverage, and cost
    """
    # Source quality classification
    quality_counts = {'A': 0, 'B': 0, 'C': 0}
    for source in sources:
        tier = classify_source_quality(source['url'])
        quality_counts[tier] += 1

    # Independence metrics (unique domains)
    domains = set()
    for source in sources:
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(source['url']).netloc
            # Remove www. prefix
            domain = domain.replace('www.', '')
            domains.add(domain)
        except:
            pass

    # Cost calculation
    # Claude Sonnet 4.5: $3/M input, $15/M output
    input_cost = (input_tokens / 1_000_000) * 3.0
    output_cost = (output_tokens / 1_000_000) * 15.0
    total_cost = input_cost + output_cost

    return {
        'quality_mix': quality_counts,
        'unique_domains': len(domains),
        'total_sources': len(sources),
        'input_tokens': int(input_tokens or 0),
        'output_tokens': int(output_tokens or 0),
        'tokens': int((input_tokens or 0) + (output_tokens or 0)),
        'cost_usd': float(total_cost)
    }

def calculate_evidence_traceability(analysis_text, sources=None):
    """
    Count citations in analysis text and calculate traceability rate

    Three categories:
    1. Supported: Has Claim + Quote + Source (or old [Source: ...] format)
    2. Inferences: Marked with [INFERENCE] - analyst's interpretation
    3. Unsupported: Claims without evidence (stand-alone **Claim:** without Quote/Source)

    If sources list provided, also calculates tier breakdown of supported claims.

    Returns:
        dict with supported_claims, inferences, unsupported_claims, total_claims, traceability_rate,
        and optionally tier_a_claims, tier_b_claims, tier_c_claims
    """
    if not analysis_text:
        return {
            'supported_claims': 0,
            'inferences': 0,
            'unsupported_claims': 0,
            'total_claims': 0,
            'traceability_rate': 0,
            'tier_a_claims': 0,
            'tier_b_claims': 0,
            'tier_c_claims': 0
        }

    # 1. Count SUPPORTED claims: **Claim:** ... **Quote:** ... **Source:** URL
    structured_claims = re.findall(
        r'\*\*Claim:\*\*.*?\*\*Quote:\*\*.*?\*\*Source:\*\*\s*(https?://[^\s]+)',
        analysis_text,
        re.DOTALL
    )

    # Also count OLD format for backward compatibility: [Source: ... - URL]
    old_format_citations = re.findall(r'\[Source:.*?(https?://[^\]]+)\]', analysis_text)

    # Total supported claims (new format + old format)
    supported_claims = len(structured_claims) + len(old_format_citations)

    # Extract all URLs from supported claims
    claim_urls = structured_claims + old_format_citations

    # 2. Count INFERENCES: marked with [INFERENCE]
    inferences = re.findall(r'\[INFERENCE\]', analysis_text)
    inference_count = len(inferences)

    # 3. Count UNSUPPORTED claims: **Claim:** without Quote/Source
    # Find all **Claim:** statements
    all_claims = re.findall(r'\*\*Claim:\*\*[^\n]+', analysis_text)
    # Unsupported = claims that aren't followed by Quote + Source
    # This is approximate - we already counted supported structured claims
    # So unsupported = total **Claim:** statements - supported claims from new format
    unsupported_claims = max(0, len(all_claims) - len(structured_claims))

    # Total claims = supported + inferences + unsupported
    total_claims = supported_claims + inference_count + unsupported_claims

    # 4. Calculate tier breakdown if sources provided
    tier_a_claims = 0
    tier_b_claims = 0
    tier_c_claims = 0

    if sources and claim_urls:
        # Create URL to tier mapping
        url_to_tier = {}
        for source in sources:
            tier = classify_source_quality(source['url'])
            url_to_tier[source['url']] = tier

        # Classify each claim URL by tier
        for url in claim_urls:
            # Clean URL (remove trailing punctuation, whitespace)
            url = url.strip().rstrip('.,;)')
            # Find matching tier
            tier = url_to_tier.get(url, classify_source_quality(url))
            if tier == 'A':
                tier_a_claims += 1
            elif tier == 'B':
                tier_b_claims += 1
            else:
                tier_c_claims += 1

    if total_claims == 0:
        return {
            'supported_claims': 0,
            'inferences': 0,
            'unsupported_claims': 0,
            'total_claims': 0,
            'traceability_rate': 0,
            'tier_a_claims': 0,
            'tier_b_claims': 0,
            'tier_c_claims': 0
        }

    traceability_rate = (supported_claims / total_claims) * 100

    return {
        'supported_claims': supported_claims,
        'inferences': inference_count,
        'unsupported_claims': unsupported_claims,
        'total_claims': total_claims,
        'traceability_rate': round(traceability_rate, 1),
        'tier_a_claims': tier_a_claims,
        'tier_b_claims': tier_b_claims,
        'tier_c_claims': tier_c_claims
    }

def display_retrieval_scorecard(metrics, search_tool_name="", exa_search_count=0, analysis_text="", sources=None):
    """
    Display retrieval scorecard with metrics
    """
    st.markdown("#### 📊 Scorecard")

    # Normalize metrics from different pathways (live/demo/history) so the UI doesn't break
    metrics = _normalize_scorecard_metrics(metrics, sources)

    # Quality Mix
    st.markdown("**Source Quality Distribution:**")

    # Calculate percentages
    total_sources = int(metrics.get('total_sources', 0) or 0)
    tier_a_pct = (metrics.get('quality_mix', {}).get('A', 0) / total_sources * 100) if total_sources > 0 else 0
    tier_b_pct = (metrics.get('quality_mix', {}).get('B', 0) / total_sources * 100) if total_sources > 0 else 0
    tier_c_pct = (metrics.get('quality_mix', {}).get('C', 0) / total_sources * 100) if total_sources > 0 else 0

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("🥇 Tier A",
                  f"{metrics.get('quality_mix', {}).get('A', 0)} ({tier_a_pct:.0f}%)",
                  help="Gov/edu/major news")
    with col_b:
        st.metric("🥈 Tier B",
                  f"{metrics.get('quality_mix', {}).get('B', 0)} ({tier_b_pct:.0f}%)",
                  help="Industry pubs")
    with col_c:
        st.metric("🥉 Tier C",
                  f"{metrics.get('quality_mix', {}).get('C', 0)} ({tier_c_pct:.0f}%)",
                  help="Other sources")

    # Independence & Coverage
    st.markdown("**Independence & Coverage:**")

    # Calculate domain diversity percentage
    domain_diversity_pct = (metrics.get('unique_domains', 0) / total_sources * 100) if total_sources > 0 else 0

    col_ind, col_cov = st.columns(2)
    with col_ind:
        st.metric(
            "🌐 Unique Domains",
            f"{metrics.get('unique_domains', 0)} ({domain_diversity_pct:.0f}%)",
            help="Domain diversity: higher % = more independent sources",
        )
    with col_cov:
        st.metric("📚 Total Sources", metrics.get('total_sources', 0))

    # Evidence Traceability (if analysis text provided)
    has_precomputed_trace = (
            metrics.get("traceability_rate") is not None and
            metrics.get("supported_claims") is not None and
            metrics.get("inferences") is not None and
            metrics.get("unsupported_claims") is not None
    )

    if has_precomputed_trace or analysis_text:
        if has_precomputed_trace:
            supported = int(metrics.get("supported_claims") or 0)
            inferences = int(metrics.get("inferences") or 0)
            unsupported = int(metrics.get("unsupported_claims") or 0)
            total = max(1, supported + inferences + unsupported)
            trace_pct = float(metrics.get("traceability_rate") or 0.0)
            tier_a = int(metrics.get("tier_a_claims") or 0)
            tier_b = int(metrics.get("tier_b_claims") or 0)
            tier_c = int(metrics.get("tier_c_claims") or 0)
        else:
            traceability = calculate_evidence_traceability(analysis_text, sources)
            supported = int(traceability.get("supported_claims", 0) or 0)
            inferences = int(traceability.get("inferences", 0) or 0)
            unsupported = int(traceability.get("unsupported_claims", 0) or 0)
            total = max(1, int(traceability.get("total_claims", 0) or 0))
            trace_pct = float(traceability.get("traceability_rate", 0.0) or 0.0)
            tier_a = int(traceability.get("tier_a_claims", 0) or 0)
            tier_b = int(traceability.get("tier_b_claims", 0) or 0)
            tier_c = int(traceability.get("tier_c_claims", 0) or 0)

        st.markdown("**Evidence Traceability:**")
        st.metric("✅ Traceability", f"{supported}/{total} ({trace_pct:.0f}%)",
                  help="Claims backed by direct quotes + URLs")

        if supported > 0:
            st.caption(f"🥇 Tier-A backed claims: {tier_a}/{supported} ({(tier_a / supported) * 100:.0f}%)")
            st.caption(f"🥈 Tier-B backed claims: {tier_b}/{supported} ({(tier_b / supported) * 100:.0f}%)")
            st.caption(f"🥉 Tier-C backed claims: {tier_c}/{supported} ({(tier_c / supported) * 100:.0f}%)")

        if inferences > 0:
            st.caption(f"🔶 Inferences: {inferences}/{total} ({(inferences / total) * 100:.0f}%)")

        if unsupported > 0:
            st.caption(f"⚠️ Unsupported: {unsupported}/{total} ({(unsupported / total) * 100:.0f}%)")
            st.warning(f"⚠️ {unsupported} unsupported claim(s) need verification or evidence")

    # Resource Usage: Cost & Tokens
    st.markdown("**Resource Usage:**")
    col_tok, col_cost = st.columns(2)
    with col_tok:
        st.metric("🔢 Tokens", f"{int(metrics.get('tokens', 0) or 0):,}")
    with col_cost:
        cost_display = f"${float(metrics.get('cost_usd', 0.0) or 0.0):.4f}"
        # Add Exa search cost if applicable
        if exa_search_count > 0:
            exa_cost = (exa_search_count / 1000) * 5.0  # $5 per 1000 searches
            total_with_exa = float(metrics.get('cost_usd', 0.0) or 0.0) + exa_cost
            cost_display = f"${total_with_exa:.4f}"
            st.metric(
                "💰 Total Cost",
                cost_display,
                help=f"Claude: ${float(metrics.get('cost_usd', 0.0) or 0.0):.4f} + Exa: ${exa_cost:.4f}",
            )
        else:
            st.metric("💰 Cost (Claude)", cost_display)

def render_history_page():
    st.title("📈 Comparison History")
    st.caption("Browse past runs, inspect saved results, and run deeper analysis without triggering new API calls.")

    # --- BEGIN: moved History UI from bottom of file ---
    # st.markdown("## 📈 Comparison History")

    # Button to load history on demand (doesn't cause page refresh)
    if st.button("📊 Load Comparison History", type="secondary"):
        st.session_state.show_history = True
        st.rerun()

    # Show history if button was clicked or if just logged a comparison
    if st.session_state.get('show_history', False) or st.session_state.get('comparison_logged', False):
        # Fetch fresh stats and detailed history
        try:
            stats = _safe_get_total_stats()
            history = _safe_get_detailed_history()
        except Exception as e:
            st.error(f"Failed to load history: {e}")
            return

        # Basic visibility so the button doesn't feel like it did nothing
        total_q = (stats or {}).get('total_queries', 0) if stats else 0
        st.caption(f"Loaded stats: {total_q} run(s). History rows: {len(history) if history else 0}.")

        if not history:
            st.info("No saved runs found yet. Run a comparison once to generate a saved result JSON.")
            return

        # If CSV-derived stats are empty, still proceed using folder-scan history
        if not stats or total_q == 0:
            st.warning(
                "No comparisons are logged in the CSV yet, but saved result JSONs were found. "
                "History below is populated from comparison_results/*.json."
            )
            # Create a minimal stats object from history for display
            stats = stats or {}
            stats["total_queries"] = len(history)
            stats.setdefault("total_exa_cost", 0.0)
            stats.setdefault("total_claude_cost", 0.0)
            stats.setdefault("avg_exa_cost", 0.0)
            stats.setdefault("avg_claude_cost", 0.0)
            stats.setdefault("avg_exa_traceability", 0.0)
            stats.setdefault("avg_claude_traceability", 0.0)
            stats.setdefault("avg_exa_inference_rate", 0.0)
            stats.setdefault("avg_claude_inference_rate", 0.0)
            stats.setdefault("avg_exa_unsupported_rate", 0.0)
            stats.setdefault("avg_claude_unsupported_rate", 0.0)
            stats.setdefault("csv_path", getattr(metrics_logger, "CSV_PATH", "comparison_metrics.csv"))

        # Proceed with normal rendering
        if stats and stats.get('total_queries', 0) > 0 and history:
            # Summary metrics at top
            st.metric("Total Queries", stats['total_queries'])

            st.markdown("### 📊 Average Traceability Metrics")

            # Traceability comparison table
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Exa + Claude")
                st.metric("Total Cost", f"${stats['total_exa_cost']:.4f}")
                st.metric("Avg Cost per Run", f"${stats['avg_exa_cost']:.4f}")
                st.metric("Traceability", f"{float(stats.get('avg_exa_traceability', 0.0)):.1f}%")
                st.metric("Inference Rate", f"{float(stats.get('avg_exa_inference_rate', 0.0)):.1f}%")
                st.metric("Unsupported Rate", f"{float(stats.get('avg_exa_unsupported_rate', 0.0)):.1f}%")

            with col2:
                st.markdown("#### Claude Web Search")
                st.metric("Total Cost", f"${stats['total_claude_cost']:.4f}")
                st.metric("Avg Cost per Run", f"${stats['avg_claude_cost']:.4f}")
                st.metric("Traceability", f"{float(stats.get('avg_claude_traceability', 0.0)):.1f}%")
                st.metric("Inference Rate", f"{float(stats.get('avg_claude_inference_rate', 0.0)):.1f}%")
                st.metric("Unsupported Rate", f"{float(stats.get('avg_claude_unsupported_rate', 0.0)):.1f}%")

            st.caption(f"📁 Metrics saved to: `{stats['csv_path']}`")

            # Show when last updated
            if st.session_state.get('comparison_logged', False):
                st.success("✅ Updated with latest comparison")

            st.markdown("---")
            st.markdown("### 📋 Detailed Comparison History")
            st.caption("Per-run summary extracted from saved result JSONs. This does not call any APIs.")

            try:
                import pandas as pd
                hist_rows = _safe_get_detailed_history() or []
                if not hist_rows:
                    st.info("No saved runs found.")
                else:
                    df = pd.DataFrame(hist_rows)
                    preferred = [
                        "timestamp",
                        "category",
                        "query",
                        "exa_search_type",
                        "exa_search_count",
                        "exa_tier_a_pct",
                        "claude_tier_a_pct",
                        "exa_traceability",
                        "claude_traceability",
                        "exa_cost",
                        "claude_cost",
                        "exa_time_seconds",
                        "claude_time_seconds",
                        "result_id",
                    ]
                    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
                    df = df[cols]
                    st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not render detailed history table: {e}")

            st.markdown("---")
            st.markdown("#### 🔍 View Saved Result")
            st.caption("Load the full analysis text and sources from any previous comparison")

            # Prefer query-first labels from metrics_logger.list_saved_runs() when available
            if hasattr(metrics_logger, "list_saved_runs"):
                try:
                    runs = metrics_logger.list_saved_runs()
                except Exception:
                    runs = []
                result_options = {}
                for r in runs:
                    rid = r.get("id") or r.get("result_id")
                    label = (r.get("label") or r.get("query") or "").strip()
                    if not label:
                        label = rid or ""
                    if not rid:
                        continue
                    if len(label) > 120:
                        label = label[:120] + "…"
                    display = f"{label}  [{rid}]"
                    result_options[display] = rid
            else:
                result_options = {}
            # If list_saved_runs() wasn't available (or returned empty), fall back to history rows
            if not result_options:
                for row in history:
                    result_id = row.get('result_id', '')
                    if result_id:
                        query_txt = (row.get('query') or '').strip()
                        label = query_txt if query_txt else result_id
                        if len(label) > 120:
                            label = label[:120] + "…"
                        display = f"{label}  [{result_id}]"
                        result_options[display] = result_id

            if result_options:
                selected_label = st.selectbox("Select a comparison to view:", [""] + list(result_options.keys()))

                if selected_label:
                    result_id = result_options[selected_label]
                    result_data = _safe_load_result(result_id)

                    if not result_data:
                        st.error(
                            "Could not load this saved result. "
                            "This usually means the JSON file is missing/corrupt or load_result() is failing."
                        )
                        st.caption(f"result_id: {result_id}")
                        st.caption(f"comparison_results exists: {os.path.isdir('comparison_results')}")
                        return

                    st.success(f"✅ Loaded result: {result_id}")

                    # --- Scorecard + Claims first (before long analysis text) ---
                    st.markdown("---")

                    # Ensure metrics fields exist even for older saved runs
                    result_data = _ensure_loaded_result_metrics(result_data)

                    st.markdown("#### 📊 Metrics Comparison")

                    # Pull content from saved JSON
                    exa_branch = (result_data.get("exa") or result_data.get("exa_claude") or result_data.get("exa_plus_claude") or {})
                    claude_branch = (result_data.get("claude") or result_data.get("claude_web") or {})

                    exa_sources = (exa_branch.get("sources") or [])
                    claude_sources = (claude_branch.get("sources") or [])
                    exa_text = (exa_branch.get("analysis_text") or exa_branch.get("analysis") or exa_branch.get("answer") or "")
                    claude_text = (claude_branch.get("analysis_text") or claude_branch.get("analysis") or claude_branch.get("answer") or "")

                    # Find the metrics CSV/history row (preferred), else derive what we can from the saved JSON
                    csv_row = None
                    try:
                        if "_get_metrics_row_for_result" in globals():
                            csv_row = _get_metrics_row_for_result(result_id, result_data)
                        else:
                            for r in history:
                                if str(r.get("result_id", "")).strip() == str(result_id).strip():
                                    csv_row = r
                                    break
                            if not csv_row:
                                qtxt = (result_data.get("query") or "").strip()
                                if qtxt:
                                    for r in reversed(history):
                                        if str(r.get("query") or "").strip() == qtxt:
                                            csv_row = r
                                            break
                    except Exception:
                        csv_row = None

                    def _num(x, default=0):
                        try:
                            return int(float(x)) if x not in (None, "", " ") else default
                        except Exception:
                            return default

                    def _flt(x, default=0.0):
                        try:
                            return float(x) if x not in (None, "", " ") else default
                        except Exception:
                            return default

                    if not csv_row:
                        # Derive scorecard metrics from sources/text; tokens/cost will be 0 for older runs
                        exa_metrics = calculate_retrieval_metrics(exa_sources, 0, 0)
                        claude_metrics = calculate_retrieval_metrics(claude_sources, 0, 0)

                        exa_t = calculate_evidence_traceability(exa_text, exa_sources)
                        claude_t = calculate_evidence_traceability(claude_text, claude_sources)

                        exa_metrics["traceability_rate"] = exa_t.get("traceability_rate", 0.0)
                        exa_metrics["supported_claims"] = exa_t.get("supported_claims", 0)
                        exa_metrics["inferences"] = exa_t.get("inferences", 0)
                        exa_metrics["unsupported_claims"] = exa_t.get("unsupported_claims", 0)
                        exa_metrics["tier_a_claims"] = exa_t.get("tier_a_claims", 0)
                        exa_metrics["tier_b_claims"] = exa_t.get("tier_b_claims", 0)
                        exa_metrics["tier_c_claims"] = exa_t.get("tier_c_claims", 0)
                        exa_metrics["tokens"] = 0
                        exa_metrics["cost_usd"] = 0.0

                        claude_metrics["traceability_rate"] = claude_t.get("traceability_rate", 0.0)
                        claude_metrics["supported_claims"] = claude_t.get("supported_claims", 0)
                        claude_metrics["inferences"] = claude_t.get("inferences", 0)
                        claude_metrics["unsupported_claims"] = claude_t.get("unsupported_claims", 0)
                        claude_metrics["tier_a_claims"] = claude_t.get("tier_a_claims", 0)
                        claude_metrics["tier_b_claims"] = claude_t.get("tier_b_claims", 0)
                        claude_metrics["tier_c_claims"] = claude_t.get("tier_c_claims", 0)
                        claude_metrics["tokens"] = 0
                        claude_metrics["cost_usd"] = 0.0
                    else:
                        exa_metrics = {
                            "quality_mix": {
                                "A": _num((csv_row or {}).get("exa_tier_a")),
                                "B": _num((csv_row or {}).get("exa_tier_b")),
                                "C": _num((csv_row or {}).get("exa_tier_c")),
                            },
                            "unique_domains": _num((csv_row or {}).get("exa_domains")),
                            "total_sources": _num((csv_row or {}).get("exa_sources")),
                            "tokens": _num((csv_row or {}).get("exa_tokens")),
                            "cost_usd": _flt((csv_row or {}).get("exa_cost")),
                            "traceability_rate": _flt((csv_row or {}).get("exa_traceability")),
                            "supported_claims": _num((csv_row or {}).get("exa_supported")),
                            "inferences": _num((csv_row or {}).get("exa_inferences")),
                            "unsupported_claims": _num((csv_row or {}).get("exa_unsupported")),
                            "tier_a_claims": _num((csv_row or {}).get("exa_tier_a_claims")),
                            "tier_b_claims": _num((csv_row or {}).get("exa_tier_b_claims")),
                            "tier_c_claims": _num((csv_row or {}).get("exa_tier_c_claims")),
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }

                        claude_metrics = {
                            "quality_mix": {
                                "A": _num((csv_row or {}).get("claude_tier_a")),
                                "B": _num((csv_row or {}).get("claude_tier_b")),
                                "C": _num((csv_row or {}).get("claude_tier_c")),
                            },
                            "unique_domains": _num((csv_row or {}).get("claude_domains")),
                            "total_sources": _num((csv_row or {}).get("claude_sources")),
                            "tokens": _num((csv_row or {}).get("claude_tokens")),
                            "cost_usd": _flt((csv_row or {}).get("claude_cost")),
                            "traceability_rate": _flt((csv_row or {}).get("claude_traceability")),
                            "supported_claims": _num((csv_row or {}).get("claude_supported")),
                            "inferences": _num((csv_row or {}).get("claude_inferences")),
                            "unsupported_claims": _num((csv_row or {}).get("claude_unsupported")),
                            "tier_a_claims": _num((csv_row or {}).get("claude_tier_a_claims")),
                            "tier_b_claims": _num((csv_row or {}).get("claude_tier_b_claims")),
                            "tier_c_claims": _num((csv_row or {}).get("claude_tier_c_claims")),
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }

                    # Render scorecards
                    left, right = st.columns(2)
                    with left:
                        st.markdown("### 🔷 Exa + Claude")
                        display_retrieval_scorecard(exa_metrics, "Exa", 0, exa_text, exa_sources)
                    with right:
                        st.markdown("### 🔶 Claude Web Search")
                        display_retrieval_scorecard(claude_metrics, "Claude Web Search", 0, claude_text, claude_sources)

                    # Claims block
                    st.markdown("---")
                    st.markdown("#### 🔬 Claims")
                    with st.expander("Claims (saved)", expanded=False):
                        st.caption("Saved claim-level analysis for this run. If missing, generate it from the Run page and it will be persisted.")
                        saved_report = result_data.get("claims_report")
                        if saved_report:
                            try:
                                from claims_analyzer import render_claims_report
                                render_claims_report(saved_report)
                            except Exception:
                                st.json(saved_report)
                        else:
                            st.info("No saved claims report for this run. Go to Run, click ‘Analyze claims’, then come back here.")

                    # --- Then show the long-form results (analysis + sources) ---
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🔍 Exa + Claude Results")
                        st.markdown(f"**Query:** {result_data.get('query','')}")
                        st.markdown(f"**Searches:** {exa_branch.get('search_count', 0)}")
                        st.markdown(f"**Time:** {float(exa_branch.get('time_seconds', 0) or 0):.1f}s")

                        with st.expander("📝 Analysis", expanded=False):
                            st.markdown(exa_text)

                        with st.expander(f"📦 Sources ({len(exa_sources)})", expanded=False):
                            for i, source in enumerate(exa_sources, 1):
                                st.markdown(f"**{i}. {source.get('title','')}**")
                                st.caption(f"🔗 {source.get('url','')}")
                                if source.get('excerpt'):
                                    st.text(str(source.get('excerpt'))[:150] + "...")
                                st.markdown("---")

                        with st.expander(f"🔍 Search Queries ({len(exa_branch.get('search_queries', []) or [])})", expanded=False):
                            for q in (exa_branch.get('search_queries', []) or []):
                                st.markdown(f"- {q}")

                    with col2:
                        st.markdown("### 🌐 Claude Web Search Results")
                        st.markdown(f"**Query:** {result_data.get('query','')}")
                        st.markdown(f"**Time:** {float(claude_branch.get('time_seconds', 0) or 0):.1f}s")

                        with st.expander("📝 Analysis", expanded=False):
                            st.markdown(claude_text)

                        with st.expander(f"📦 Sources ({len(claude_sources)})", expanded=False):
                            for i, source in enumerate(claude_sources, 1):
                                st.markdown(f"**{i}. {source.get('title','')}**")
                                st.caption(f"🔗 {source.get('url','')}")
                                st.markdown("---")

                    # IMPORTANT: keep the rest of your existing dataframe/history table rendering here unchanged.
    # --- END: moved History UI from bottom of file ---

# Page config
# Page config
st.set_page_config(
    page_title="Exa + Claude vs Claude Web Search",
    page_icon="🔍",
    layout="wide"
)

# --- Compact UI styling (reduce vertical whitespace) ---
st.markdown(
    """
    <style>
      /* Main page spacing: tight but never clipped */
      section.main > div.block-container {
        padding-top: 4.5rem;
        padding-bottom: 1.0rem;
        padding-left: 2.0rem;
        padding-right: 2.0rem;
        max-width: 100%;
      }

      /* Ensure Streamlit header overlay doesn't clip the top content */
      div[data-testid="stHeader"] { background: rgba(0,0,0,0); }

      /* Tight headings without collapsing into the top edge */
      h1 { margin-top: 0.25rem; margin-bottom: 0.35rem; }
      h2 { margin-top: 0.75rem; margin-bottom: 0.25rem; }
      h3 { margin-top: 0.65rem; margin-bottom: 0.25rem; }

      /* Reduce vertical gaps a bit (avoid :has()) */
      div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.35rem; }

      /* Expander spacing */
      div[data-testid="stExpander"] { margin-top: 0.25rem; }

      /* Hide footer */
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Exa + Claude vs Claude Web Search")
st.caption("Interactive comparison tool. Test prompts and retrieval configurations.")

# --- COMPARISON CRITERIA SECTION ---

# with st.expander("📊 What You're Measuring (Comparison Criteria)", expanded=False):
#     st.markdown(get_measurement_guide_text())

with st.sidebar:
    st.markdown("### Exa Verification Ops")
    st.caption("Run comparisons, then inspect History without new API calls.")

    st.subheader("Navigation")
    nav_page = st.radio(
        "Go to",
        ["Run", "History"],
        index=0,
        key="nav_page",
        help="Run comparisons or browse saved history and deep analysis"
    )

    st.markdown("---")
    with st.expander("⚙️ Retrieval settings", expanded=True):
        search_type = st.selectbox(
            "Search type",
            ["neural", "auto", "keyword"],
            index=["neural", "auto", "keyword"].index(st.session_state.get("search_type", "neural")),
            key="search_type",
            help="Neural: meaning-based. Keyword: traditional. Auto: best of both."
        )

        num_results = st.slider(
            "Results",
            min_value=3,
            max_value=20,
            value=int(st.session_state.get("num_results", 10)),
            key="num_results",
            help="How many sources to retrieve"
        )

        with st.expander("More controls", expanded=False):
            max_characters = st.slider(
                "Max characters per source",
                min_value=500,
                max_value=3000,
                value=int(st.session_state.get("max_characters", 1500)),
                step=100,
                key="max_characters",
                help="Amount of text to extract from each source"
            )

            use_autoprompt = st.checkbox(
                "Use Autoprompt",
                value=bool(st.session_state.get("use_autoprompt", True)),
                key="use_autoprompt",
                help="Let Exa optimize your query"
            )

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                include_domains = st.text_area(
                    "Include domains",
                    value=st.session_state.get("include_domains", ""),
                    placeholder="faa.gov\nntsb.gov\nreuters.com",
                    key="include_domains",
                    help="Only search these domains (optional)"
                )
            with col_d2:
                exclude_domains = st.text_area(
                    "Exclude domains",
                    value=st.session_state.get("exclude_domains", ""),
                    placeholder="spam.com\nseo-site.com",
                    key="exclude_domains",
                    help="Never search these domains (optional)"
                )

            start_published_date = st.date_input(
                "Start date",
                value=st.session_state.get("start_published_date", None),
                key="start_published_date",
                help="Only articles published after this date"
            )

        st.caption("Tip: keep these stable during the demo so differences come from retrieval, not knobs.")

    st.markdown("---")
    st.subheader("Quick links")
    st.markdown("- App: https://exatesting-angfz5745qjd52jhk5jkbo.streamlit.app/")
    st.markdown("- Slides: https://docs.google.com/presentation/d/1wSS9aNud553dh0YOtq1KfzO2LMhn3Ht_n-4aEknv6dU/edit")
    st.markdown("- One-pager: https://docs.google.com/document/d/1Ub9tDcu48N3uqcEdSlXCPO0BkX-m8syceCtIzu4Wxfs/edit?usp=sharing")

# Route to History view (no API calls)
if st.session_state.get("nav_page") == "History":
    render_history_page()
    st.stop()

# Main search interface
st.markdown("### 🎯 Enter Your Research Question")

# Example Prompts Section
# Category selector (outside expander so it's always visible)
query_category = st.selectbox(
    "Query Category (for pattern analysis)",
    QUERY_CATEGORIES,
    help="Categorize your query to identify which types Exa+Claude vs Claude WS excel at",
    key="query_category_select"
)


# Example Prompts Section
with st.expander("📰 Example Newsroom Prompts", expanded=False):
    selected_prompt = st.selectbox(
        "Select an example",
        get_example_prompts(query_category),
        key="example_prompt_select"
    )
    if st.button("Use this prompt", key="use_example_prompt"):
        st.session_state["query_textarea"] = selected_prompt
        st.rerun()

# Use example prompt if selected, otherwise use default
default_query = "A tech company hid research showing their algorithm harms teens. Find historical examples from OTHER industries where companies concealed internal research about product harm."

# Initialize with default if not present
if "query_textarea" not in st.session_state:
    st.session_state["query_textarea"] = default_query

query = st.text_area(
    "Research Query",
    key="query_textarea",
    height=120,
    help="This query will be sent to both approaches",
)

#
# --- RUN MODE (Live vs Demo) ---
# Remove any heading for run mode above the radio (e.g. st.markdown("## 🧪 Run mode")), so only the radio remains.
run_mode = st.radio(
    "Run mode",
    ["Live (call APIs)", "Demo (use saved results, no API calls)"],
    index=0,
    horizontal=True,
    key="run_mode",
    label_visibility="collapsed",
    help="Demo mode renders a previously saved run so you can test layout without spending tokens."
)

# --- Hydrate session state from a loaded saved result (Demo) so downstream features (articles, etc.) work ---
# NOTE: Article generation section runs BEFORE the bottom "DEMO RENDER" block, so this must happen early.
if st.session_state.get("run_mode", "").startswith("Demo") and st.session_state.get("demo_loaded"):
    _rd = st.session_state.get("demo_result_data") or {}
    _exa = (
        _rd.get("exa")
        or _rd.get("exa_claude")
        or _rd.get("exa_plus_claude")
        or {}
    )

    st.session_state["loaded_result_data"] = _rd
    st.session_state["loaded_query"] = (_rd.get("query") or "").strip()

    # These names match what live-run + article generation expects
    st.session_state["exa_sources"] = _exa.get("sources") or []
    st.session_state["exa_analysis_text"] = (
        _exa.get("analysis_text")
        or _exa.get("analysis")
        or _exa.get("answer")
        or ""
    )

if run_mode.startswith("Demo"):
    try:
        files = [f for f in os.listdir("comparison_results") if f.endswith(".json")]
        files = sorted(files, reverse=True)
    except Exception:
        files = []

    if not files:
        st.info("No saved results found in comparison_results/")
    else:
        options = []
        label_to_file = {}

        for fname in files:
            fpath = os.path.join("comparison_results", fname)
            query_text = ""
            result_id = fname.rsplit("_", 1)[-1].replace(".json", "")

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                query_text = (data.get("query") or "").strip()
                if data.get("result_id"):
                    result_id = str(data.get("result_id"))
            except Exception:
                query_text = ""

            label = query_text if query_text else fname
            if len(label) > 120:
                label = label[:120] + "…"

            display_label = f"{label}  [{result_id}]"
            options.append(display_label)
            label_to_file[display_label] = fpath

        selected_label = st.selectbox("Select a saved search", options, index=0)
        selected_path = label_to_file.get(selected_label)
        st.session_state["demo_selected_path"] = selected_path
        # st.session_state["demo_loaded"] = False
        # st.session_state["demo_result_data"] = None
        # st.session_state["demo_result_id"] = None

        st.info("🧪 Demo mode is ON: click the main button below to load a saved search. No API calls will be made.")
else:
    # Ensure demo is off when switching back to Live
    st.session_state.demo_loaded = False

# Run comparison button
run_label = "🚀 Run Comparison"
if st.session_state.get("run_mode", "").startswith("Demo"):
    run_label = "🎛️ Load Saved Result"

if st.button(run_label, type="primary", use_container_width=True):
    # If Demo: load the saved result, store in session_state, then rerun.
    # Important: do NOT fall through into the Live comparison path.
    if st.session_state.get("run_mode", "").startswith("Demo"):
        selected_path = st.session_state.get("demo_selected_path")
        if not selected_path or not os.path.exists(selected_path):
            st.error("Please select a saved run first, or switch Run mode back to Live.")
            st.stop()

        try:
            with open(selected_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
        except Exception as e:
            st.error(f"Could not load this saved result: {e}")
            st.stop()

        demo_result_id = str(
            result_data.get("result_id")
            or os.path.basename(selected_path).rsplit("_", 1)[-1].replace(".json", "")
        )

        st.session_state.demo_result_data = result_data
        st.session_state.demo_result_id = demo_result_id
        st.session_state.demo_loaded = True

        # Rerun so the persistent Demo renderer at the bottom takes over.
        st.rerun()

    # Live mode: proceed with the normal API comparison run
    if not query.strip():
        st.error("Please enter a research query")
        st.stop()

    # Create two columns for results
    col1, col2 = st.columns(2)

    # --- LEFT COLUMN: Exa + Claude (Tool Use) ---
    with col1:
        st.markdown("### 🔷 Exa + Claude (Tool Use)")

        try:
            st.info("🤖 Initializing Exa tool...")

            # Define Exa as a tool for Claude
            exa_tool = {
                "name": "exa_search",
                "description": f"Search the web using Exa's semantic/neural search. Returns structured results with URLs, titles, excerpts, and publish dates. Search type: {search_type}. Max results: {num_results}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Can be semantic/conceptual for neural search or keyword-based."
                        },
                        "num_results": {
                            "type": "integer",
                            "description": f"Number of results to return (max {num_results})",
                            "default": num_results
                        }
                    },
                    "required": ["query"]
                }
            }

            # Build domain filter description
            domain_info = ""
            if include_domains.strip():
                domain_info += f" Restricted to domains: {include_domains.strip()}."
            if exclude_domains.strip():
                domain_info += f" Excluding domains: {exclude_domains.strip()}."
            if domain_info:
                exa_tool["description"] += domain_info

            initial_prompt = build_exa_initial_prompt(query)

            messages = [{"role": "user", "content": initial_prompt}]
            all_sources = []
            tool_use_count = 0
            # Track total LLM token usage across the full agentic loop (multiple Claude calls)
            exa_total_input_tokens = 0
            exa_total_output_tokens = 0

            # Clear and initialize session state for sources
            st.session_state.exa_sources = []

            # Start timing
            exa_start_time = time.time()

            # Create a status container that updates in place
            status_container = st.status("🔍 Exa + Claude Search", expanded=True)

            # Create placeholder containers inside status for dynamic updates
            with status_container:
                search_count_placeholder = st.empty()
                search_list_placeholder = st.empty()

            search_queries = []

            # Agentic loop
            while True:
                try:
                    response = anthropic.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=4000,
                        tools=[exa_tool],
                        messages=messages
                    )
                except Exception as api_error:
                    if "rate_limit_error" in str(api_error):
                        import time
                        countdown_placeholder = st.empty()
                        for remaining in range(60, 0, -1):
                            countdown_placeholder.warning(f"⚠️ Rate limit reached. Retrying in {remaining} seconds...")
                            time.sleep(1)
                        countdown_placeholder.success("✓ Resuming search...")
                        time.sleep(1)
                        countdown_placeholder.empty()

                        response = anthropic.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=4000,
                            tools=[exa_tool],
                            messages=messages
                        )
                    else:
                        raise

                # Accumulate token usage for this Claude call (robust across SDK shapes)
                try:
                    in_tok, out_tok = _get_usage_tokens(getattr(response, "usage", None))
                    exa_total_input_tokens += int(in_tok)
                    exa_total_output_tokens += int(out_tok)
                except Exception:
                    pass

                # Check for tool use
                tool_uses = [block for block in response.content if block.type == "tool_use"]

                if not tool_uses:
                    # No more tool calls, we have final answer
                    break

                # Add assistant message
                messages.append({"role": "assistant", "content": response.content})

                # Execute tool calls
                tool_results = []
                for tool_use in tool_uses:
                    if tool_use.name == "exa_search":
                        tool_use_count += 1
                        query_text = tool_use.input.get('query', 'N/A')
                        search_queries.append(f"Search #{tool_use_count}: {query_text[:60]}...")

                        # Update placeholders with current search list
                        search_count_placeholder.markdown(f"**Searches performed:** {tool_use_count}")
                        search_list_placeholder.markdown("\n".join([f"🔍 {q}" for q in search_queries]))

                        # Build search params
                        search_params = {
                            "query": tool_use.input["query"],
                            "type": search_type,
                            "num_results": tool_use.input.get("num_results", num_results)
                        }

                        # Add filters
                        if include_domains.strip():
                            domains = [d.strip() for d in include_domains.split("\n") if d.strip()]
                            if domains:
                                search_params["include_domains"] = domains

                        if exclude_domains.strip():
                            domains = [d.strip() for d in exclude_domains.split("\n") if d.strip()]
                            if domains:
                                search_params["exclude_domains"] = domains

                        if start_published_date:
                            search_params["start_published_date"] = start_published_date.isoformat()

                        # Execute search with contents (using new API)
                        results = exa.search(
                            **search_params,
                            contents={"text": True}
                        )

                        # Build result sources
                        sources = []
                        for r in results.results:
                            source = {
                                "title": r.title,
                                "url": r.url,
                                "published": r.published_date if hasattr(r, 'published_date') else None,
                                "excerpt": r.text[:500] if hasattr(r, 'text') and r.text else ""
                            }
                            sources.append(source)
                            all_sources.append(source)
                            st.session_state.exa_sources.append(source)  # Save to session state

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(sources, indent=2)
                        })

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})

            # Update status to show we're analyzing now
            search_count_placeholder.markdown(f"**Searches performed:** {tool_use_count} ✓")
            search_list_placeholder.markdown("\n".join([f"🔍 {q}" for q in search_queries]) + "\n\n📝 Analyzing results...")

            total_time = time.time() - exa_start_time

            # Extract response content (final assistant text)
            response_content = ""
            for block in response.content:
                if block.type == "text":
                    response_content += block.text

            # Build metrics (requires sources + token usage)
            if all_sources:
                retrieval_metrics = calculate_retrieval_metrics(
                    all_sources,
                    exa_total_input_tokens,
                    exa_total_output_tokens
                )

                traceability = calculate_evidence_traceability(response_content, all_sources)
                retrieval_metrics['traceability_rate'] = traceability['traceability_rate']
                retrieval_metrics['supported_claims'] = traceability['supported_claims']
                retrieval_metrics['inferences'] = traceability['inferences']
                retrieval_metrics['unsupported_claims'] = traceability['unsupported_claims']
                retrieval_metrics['tier_a_claims'] = traceability['tier_a_claims']
                retrieval_metrics['tier_b_claims'] = traceability['tier_b_claims']
                retrieval_metrics['tier_c_claims'] = traceability['tier_c_claims']
                retrieval_metrics['time_seconds'] = total_time

                # Persist for history/logging
                st.session_state.exa_metrics = retrieval_metrics
                st.session_state.exa_search_count = tool_use_count
                st.session_state.exa_analysis_text = response_content
                st.session_state.exa_sources = all_sources
                st.session_state.exa_search_queries = search_queries
                # Persist total token usage for history/demo rendering
                st.session_state.exa_input_tokens = exa_total_input_tokens
                st.session_state.exa_output_tokens = exa_total_output_tokens

                # Present results with less vertical sprawl
                tab_score, tab_answer, tab_sources = st.tabs(["📊 Scorecard", "📝 Answer", f"📦 Sources ({len(all_sources)})"])

                with tab_score:
                    display_retrieval_scorecard(retrieval_metrics, "Exa", tool_use_count, response_content, all_sources)
                    st.caption(f"⏱️ Time: {total_time:.1f}s")

                with tab_answer:
                    st.markdown("#### 📊 Analysis Result")
                    st.markdown(response_content)

                with tab_sources:
                    st.caption("Sources retrieved for this run")
                    for i, source in enumerate(all_sources, 1):
                        st.markdown(f"**{i}. {source['title']}**")
                        st.caption(f"🔗 {source['url']}")
                        st.caption(f"📅 Published: {source.get('published', 'N/A')}")
                        if source.get('excerpt'):
                            st.text(source['excerpt'][:150] + "...")
                        st.markdown("---")
            else:
                st.warning("No sources were retrieved.")

        except Exception as e:
            st.error(f"❌ Exa + Claude failed: {str(e)}")

    # --- RIGHT COLUMN: Claude Web Search ---
    with col2:
        st.markdown("### 🔶 Claude Web Search")

        with st.spinner("🤖 Claude searching and analyzing..."):
            try:
                claude_start_time = time.time()

                # Use Claude's built-in web search
                tools = [{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }]

                prompt = build_web_search_prompt(query, num_results)

                # Create status container
                status_container = st.status("🌐 Claude Web Search", expanded=True)

                with status_container:
                    st.write("🎯 Performing web search with Claude...")

                try:
                    response = anthropic.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=4000,
                        tools=tools,
                        messages=[{"role": "user", "content": prompt}]
                    )
                except Exception as api_error:
                    if "rate_limit_error" in str(api_error):
                        import time
                        countdown_placeholder = st.empty()
                        for remaining in range(60, 0, -1):
                            countdown_placeholder.warning(f"⚠️ Rate limit reached. Retrying in {remaining} seconds...")
                            time.sleep(1)
                        countdown_placeholder.success("✓ Resuming search...")
                        time.sleep(1)
                        countdown_placeholder.empty()

                        response = anthropic.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=4000,
                            tools=tools,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        raise

                total_time = time.time() - claude_start_time

                # Extract results
                response_content = ""
                search_count = 0
                sources = []

                for block in response.content:
                    if block.type == "text":
                        response_content += block.text
                    elif block.type == "tool_use" and block.name == "web_search":
                        search_count += 1
                    elif block.type == "web_search_tool_result":
                        for result in block.content:
                            if hasattr(result, 'url'):
                                sources.append({
                                    "title": result.title if hasattr(result, 'title') else "N/A",
                                    "url": result.url,
                                    "page_age": result.page_age if hasattr(result, 'page_age') else "N/A"
                                })

                # Update status to complete
                status_container.update(
                    label=f"✅ Claude Web Search Complete ({search_count} searches, {total_time:.1f}s)",
                    state="complete",
                    expanded=False
                )

                # Build metrics (requires sources + token usage)
                if sources:
                    in_tok, out_tok = _get_usage_tokens(getattr(response, "usage", None))
                    retrieval_metrics = calculate_retrieval_metrics(
                        sources,
                        in_tok,
                        out_tok
                    )

                    traceability = calculate_evidence_traceability(response_content, sources)
                    retrieval_metrics['traceability_rate'] = traceability['traceability_rate']
                    retrieval_metrics['supported_claims'] = traceability['supported_claims']
                    retrieval_metrics['inferences'] = traceability['inferences']
                    retrieval_metrics['unsupported_claims'] = traceability['unsupported_claims']
                    retrieval_metrics['tier_a_claims'] = traceability['tier_a_claims']
                    retrieval_metrics['tier_b_claims'] = traceability['tier_b_claims']
                    retrieval_metrics['tier_c_claims'] = traceability['tier_c_claims']
                    retrieval_metrics['time_seconds'] = total_time

                    # Persist for history/logging
                    st.session_state.claude_metrics = retrieval_metrics
                    st.session_state.claude_analysis_text = response_content
                    st.session_state.claude_sources = sources

                    tab_score, tab_answer, tab_sources = st.tabs(["📊 Scorecard", "📝 Answer", f"📦 Sources ({len(sources)})"])

                    with tab_score:
                        display_retrieval_scorecard(retrieval_metrics, "Claude Web Search", 0, response_content, sources)
                        st.caption(f"⏱️ Time: {total_time:.1f}s")

                    with tab_answer:
                        st.markdown("#### 📊 Analysis Result")
                        st.markdown(response_content)

                    with tab_sources:
                        st.caption("Sources found for this run")
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}. {source['title']}**")
                            st.caption(f"🔗 {source['url']}")
                            st.caption(f"📅 Age: {source.get('page_age', 'N/A')}")
                            st.markdown("---")
                else:
                    st.warning("No sources were found.")

            except Exception as e:
                st.error(f"❌ Claude web search failed: {str(e)}")

    # --- LOG COMPARISON METRICS (After both columns complete) ---
    # Log metrics if we have data from both approaches
    if 'exa_metrics' in st.session_state and 'claude_metrics' in st.session_state:
        result_id = metrics_logger.log_comparison(
            query=query,
            app_type='general',
            exa_metrics=st.session_state.exa_metrics,
            claude_metrics=st.session_state.claude_metrics,
            exa_search_count=st.session_state.get('exa_search_count', 0),
            search_type=search_type,
            num_results=num_results,
            query_category=query_category if query_category else None,
            exa_analysis_text=st.session_state.get('exa_analysis_text'),
            claude_analysis_text=st.session_state.get('claude_analysis_text'),
            exa_sources=st.session_state.get('exa_sources'),
            claude_sources=st.session_state.get('claude_sources'),
            exa_search_queries=st.session_state.get('exa_search_queries'),
        )
        st.session_state.last_result_id = result_id
        st.success("✅ Comparison metrics logged to CSV")
        st.info("💡 Scroll to bottom and click '📊 Load Comparison History' to view all logged comparisons")
        # Set flag to show updated stats
        st.session_state.comparison_logged = True

    elif 'exa_metrics' in st.session_state:
        # Log with empty Claude metrics if Claude failed
        empty_metrics = {'quality_mix': {'A': 0, 'B': 0, 'C': 0}, 'unique_domains': 0, 'total_sources': 0, 'tokens': 0, 'cost_usd': 0.0}
        result_id = metrics_logger.log_comparison(
            query=query,
            app_type='general',
            exa_metrics=st.session_state.exa_metrics,
            claude_metrics=empty_metrics,
            exa_search_count=st.session_state.get('exa_search_count', 0),
            search_type=search_type,
            num_results=num_results,
            query_category=query_category if query_category else None,
            exa_analysis_text=st.session_state.get('exa_analysis_text'),
            claude_analysis_text=None,
            exa_sources=st.session_state.get('exa_sources'),
            claude_sources=None,
            exa_search_queries=st.session_state.get('exa_search_queries'),
        )
        st.session_state.last_result_id = result_id
        st.warning("⚠️ Logged Exa metrics only (Claude search failed)")
        # Set flag to show updated stats
        st.session_state.comparison_logged = True

    # --- CLAIMS ANALYSIS SECTION ---
    st.markdown("---")
    st.markdown("## 🔬 Claims analysis")
    st.caption("Generate once, then it is saved with this run so History loads it instantly.")

    can_analyze = (
        st.session_state.get('exa_analysis_text') and st.session_state.get('claude_analysis_text') and
        st.session_state.get('exa_sources') and st.session_state.get('claude_sources')
    )

    if not can_analyze:
        st.info("Run a comparison first to enable claims analysis.")
    else:
        if st.session_state.get('claims_report'):
            st.success("✅ Claims report ready")
            try:
                from claims_analyzer import render_claims_report
                render_claims_report(st.session_state['claims_report'])
            except Exception:
                st.json(st.session_state['claims_report'])

        analyze_clicked = st.button("🔬 Analyze claims", type="primary", use_container_width=True)
        st.caption("This does not re-run retrieval. It only analyzes the existing outputs.")

        if analyze_clicked:
            with st.spinner("🔎 Extracting and scoring claims..."):
                from claims_analyzer import generate_claims_report
                exa_data = {'analysis_text': st.session_state.get('exa_analysis_text',''), 'sources': st.session_state.get('exa_sources', [])}
                claude_data = {'analysis_text': st.session_state.get('claude_analysis_text',''), 'sources': st.session_state.get('claude_sources', [])}
                report = generate_claims_report(exa_data, claude_data, query)
                st.session_state['claims_report'] = report

                result_id = st.session_state.get('last_result_id')
                if hasattr(metrics_logger, "save_claims_report"):
                    metrics_logger.save_claims_report(result_id, report)
                if result_id:
                    try:
                        if hasattr(metrics_logger, "save_claims_report"):
                            metrics_logger.save_claims_report(result_id, report)
                        else:
                            raise RuntimeError("metrics_logger.save_claims_report is not available")
                        st.success("✅ Saved claims report into this run")
                    except Exception as e:
                        st.warning(f"Claims report generated but could not be saved: {e}")
                else:
                    st.warning("Claims report generated but no result_id was found to persist it.")

                try:
                    from claims_analyzer import render_claims_report
                    render_claims_report(report)
                except Exception:
                    st.json(report)

# --- ARTICLE GENERATION SECTION (Outside comparison block) ---
st.markdown("---")
st.markdown("## 📰 Article Generation Comparison")
st.markdown("Generate full articles from both approaches to compare depth, accuracy, and quality.")

if st.button("📝 Generate Articles", type="secondary", use_container_width=True):
    article_col1, article_col2 = st.columns(2)

    # --- LEFT: Exa Article ---
    with article_col1:
        st.markdown("### 📄 Article from Exa + Claude")

        with st.spinner("✍️ Writing article from Exa sources..."):
            try:
                # Check if we have sources from Exa session state (works for Live and Saved/Demo)
                exa_sources_for_article = st.session_state.get("exa_sources") or []
                if not exa_sources_for_article:
                    st.warning("⚠️ Run the comparison first (or load a saved search) to generate Exa sources")
                else:
                    # Prefer the saved query when a saved result is loaded
                    query_for_article = (st.session_state.get("loaded_query") or query).strip() or query

                    article_prompt = build_exa_article_prompt(
                        query=query_for_article,
                        sources_json=json.dumps(exa_sources_for_article, indent=2)
                    )

                    exa_article_response = anthropic.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=2500,
                        messages=[{"role": "user", "content": article_prompt}]
                    )

                    exa_article = exa_article_response.content[0].text
                    st.markdown(exa_article)

                    # Metrics
                    st.caption(f"📊 Sources used: {len(exa_sources_for_article)}")
                    st.caption(
                        f"🔢 Tokens: {exa_article_response.usage.input_tokens + exa_article_response.usage.output_tokens:,}"
                    )

                    # Download button
                    st.download_button(
                        "⬇️ Download Article",
                        exa_article,
                        file_name="exa_article.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                st.error(f"❌ Failed to generate Exa article: {str(e)}")

    # --- RIGHT: Claude Web Search Article ---
    with article_col2:
        st.markdown("### 📄 Article from Claude Web Search")

        with st.spinner("✍️ Writing article from web search..."):
            try:
                # Use Claude to write article based on its web search results
                article_prompt = build_web_article_prompt(query)

                web_article_response = anthropic.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2500,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search"
                    }],
                    messages=[{"role": "user", "content": article_prompt}]
                )

                # Extract article text
                web_article = ""
                for block in web_article_response.content:
                    if block.type == "text":
                        web_article += block.text

                st.markdown(web_article)

                # Metrics
                st.caption(f"🔢 Tokens: {web_article_response.usage.input_tokens + web_article_response.usage.output_tokens:,}")

                # Download button
                st.download_button(
                    "⬇️ Download Article",
                    web_article,
                    file_name="web_search_article.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"❌ Failed to generate web search article: {str(e)}")


# --- DEMO RENDER (persists across reruns so Analyze Claims works) ---
if st.session_state.get("run_mode", "").startswith("Demo") and st.session_state.get("demo_loaded"):
    result_data = st.session_state.get("demo_result_data") or {}
    demo_result_id = str(st.session_state.get("demo_result_id") or "").strip()
    if not demo_result_id:
        st.error("Demo result id missing. Please reload the saved search.")
        st.stop()

    # --- Render Demo result (NO API calls) ---
    exa_sources = result_data.get('exa', {}).get('sources', [])
    claude_sources = result_data.get('claude', {}).get('sources', [])

    exa_text = result_data.get('exa', {}).get('analysis_text', '')
    claude_text = result_data.get('claude', {}).get('analysis_text', '')

    # Build scorecards from CSV (source of truth). This mirrors History behavior.
    csv_row = None
    try:
        hist_rows = _safe_get_detailed_history() or []
    except Exception:
        hist_rows = []

    rid = str(demo_result_id).strip()
    if rid and hist_rows:
        for row in hist_rows:
            if str(row.get("result_id") or "").strip() == rid:
                csv_row = row
                break

    if not csv_row:
        try:
            qtxt = (result_data.get("query") or "").strip()
            if qtxt and hist_rows:
                for row in reversed(hist_rows):
                    if str(row.get("query") or "").strip() == qtxt:
                        csv_row = row
                        break
        except Exception:
            csv_row = None

    def _num(val, default=0):
        try:
            if val is None:
                return default
            s = str(val).strip()
            if not s:
                return default
            return int(float(s))
        except Exception:
            return default

    def _flt(val, default=0.0):
        try:
            if val is None:
                return default
            s = str(val).strip()
            if not s:
                return default
            return float(s)
        except Exception:
            return default

    with st.expander("Debug (saved-run metrics)", expanded=False):
        st.caption("Verifies that Demo is reading the same CSV row as History.")
        st.write({
            "demo_result_id": demo_result_id,
            "csv_path": getattr(metrics_logger, "CSV_PATH", "comparison_metrics.csv"),
            "csv_row_found": bool(csv_row),
            "csv_row_result_id": (csv_row or {}).get("result_id"),
        })

    if not csv_row:
        exa_metrics = calculate_retrieval_metrics(exa_sources, 0, 0)
        claude_metrics = calculate_retrieval_metrics(claude_sources, 0, 0)
        exa_metrics["tokens"] = 0
        exa_metrics["cost_usd"] = 0.0
        claude_metrics["tokens"] = 0
        claude_metrics["cost_usd"] = 0.0
        st.warning("Saved-run tokens/cost not found in comparison_metrics.csv for this result.")
    else:
        exa_metrics = {
            "quality_mix": {"A": _num(csv_row.get("exa_tier_a")), "B": _num(csv_row.get("exa_tier_b")), "C": _num(csv_row.get("exa_tier_c"))},
            "unique_domains": _num(csv_row.get("exa_domains")),
            "total_sources": _num(csv_row.get("exa_sources")),
            "tokens": _num(csv_row.get("exa_tokens")),
            "cost_usd": _flt(csv_row.get("exa_cost")),
            "traceability_rate": _flt(csv_row.get("exa_traceability")),
            "supported_claims": _num(csv_row.get("exa_supported")),
            "inferences": _num(csv_row.get("exa_inferences")),
            "unsupported_claims": _num(csv_row.get("exa_unsupported")),
            "tier_a_claims": _num(csv_row.get("exa_tier_a_claims")),
            "tier_b_claims": _num(csv_row.get("exa_tier_b_claims")),
            "tier_c_claims": _num(csv_row.get("exa_tier_c_claims")),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        claude_metrics = {
            "quality_mix": {"A": _num(csv_row.get("claude_tier_a")), "B": _num(csv_row.get("claude_tier_b")), "C": _num(csv_row.get("claude_tier_c"))},
            "unique_domains": _num(csv_row.get("claude_domains")),
            "total_sources": _num(csv_row.get("claude_sources")),
            "tokens": _num(csv_row.get("claude_tokens")),
            "cost_usd": _flt(csv_row.get("claude_cost")),
            "traceability_rate": _flt(csv_row.get("claude_traceability")),
            "supported_claims": _num(csv_row.get("claude_supported")),
            "inferences": _num(csv_row.get("claude_inferences")),
            "unsupported_claims": _num(csv_row.get("claude_unsupported")),
            "tier_a_claims": _num(csv_row.get("claude_tier_a_claims")),
            "tier_b_claims": _num(csv_row.get("claude_tier_b_claims")),
            "tier_c_claims": _num(csv_row.get("claude_tier_c_claims")),
            "input_tokens": 0,
            "output_tokens": 0,
        }

    exa_time = float(result_data.get('exa', {}).get('time_seconds', 0) or 0)
    claude_time = float(result_data.get('claude', {}).get('time_seconds', 0) or 0)

    exa_search_count = int(result_data.get('exa', {}).get('search_count', 0) or 0)
    if csv_row:
        exa_search_count = _num((csv_row or {}).get("exa_search_count"), exa_search_count)
    exa_search_queries = result_data.get('exa', {}).get('search_queries', [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔷 Exa + Claude (Saved)")
        tab_score, tab_answer, tab_sources = st.tabs(["📊 Scorecard", "📝 Answer", f"📦 Sources ({len(exa_sources)})"])
        with tab_score:
            display_retrieval_scorecard(exa_metrics, "Exa", exa_search_count, exa_text, exa_sources)
            st.caption(f"⏱️ Time: {exa_time:.1f}s")
            if exa_search_queries:
                with st.expander(f"🔍 Search Queries ({len(exa_search_queries)})", expanded=False):
                    for q in exa_search_queries:
                        st.markdown(f"- {q}")
        with tab_answer:
            st.markdown("#### 📊 Analysis Result")
            st.markdown(exa_text)
        with tab_sources:
            for i, source in enumerate(exa_sources, 1):
                st.markdown(f"**{i}. {source.get('title','N/A')}**")
                st.caption(f"🔗 {source.get('url','')}")
                if source.get('published'):
                    st.caption(f"📅 Published: {source.get('published')}")
                if source.get('excerpt'):
                    st.text(str(source.get('excerpt'))[:150] + "...")
                st.markdown("---")

    with col2:
        st.markdown("### 🔶 Claude Web Search (Saved)")
        tab_score, tab_answer, tab_sources = st.tabs(["📊 Scorecard", "📝 Answer", f"📦 Sources ({len(claude_sources)})"])
        with tab_score:
            display_retrieval_scorecard(claude_metrics, "Claude Web Search", 0, claude_text, claude_sources)
            st.caption(f"⏱️ Time: {claude_time:.1f}s")
        with tab_answer:
            st.markdown("#### 📊 Analysis Result")
            st.markdown(claude_text)
        with tab_sources:
            for i, source in enumerate(claude_sources, 1):
                st.markdown(f"**{i}. {source.get('title','N/A')}**")
                st.caption(f"🔗 {source.get('url','')}")
                st.markdown("---")

    # --- Claims analysis (saved) ---
    st.markdown("---")
    st.markdown("## 🔬 Claims analysis (saved search)")
    st.caption("Generate once, then it is saved with this search so History loads it instantly.")

    # If the JSON doesn't already contain the report, try loading from disk (if supported)
    saved_report = result_data.get("claims_report")
    if not saved_report and hasattr(metrics_logger, "load_claims_report"):
        try:
            saved_report = metrics_logger.load_claims_report(demo_result_id)
            if saved_report:
                result_data["claims_report"] = saved_report
                st.session_state.demo_result_data = result_data
        except Exception:
            saved_report = result_data.get("claims_report")

    if saved_report:
        try:
            from claims_analyzer import render_claims_report
            render_claims_report(saved_report)
        except Exception as e:
            st.error(f"Claims report UI failed to render: {e}")
            with st.expander("Raw claims report (debug)", expanded=False):
                st.json(saved_report)
    else:
        analyze_key = f"analyze_claims__{demo_result_id}"
        if st.button("🔬 Analyze claims", type="primary", use_container_width=True, key=analyze_key):
            st.session_state[f"do_{analyze_key}"] = True

        st.caption("This does not rerun retrieval. It only analyzes the existing saved outputs.")

        if st.session_state.get(f"do_{analyze_key}", False):
            st.session_state[f"do_{analyze_key}"] = False
            with st.spinner("🔎 Extracting and scoring claims..."):
                from claims_analyzer import generate_claims_report

                exa_data = {"analysis_text": exa_text, "sources": exa_sources}
                claude_data = {"analysis_text": claude_text, "sources": claude_sources}
                report = generate_claims_report(exa_data, claude_data, result_data.get("query", ""))

                # Persist into the in-memory demo result so it renders immediately on rerun
                result_data["claims_report"] = report
                st.session_state.demo_result_data = result_data

                try:
                    if hasattr(metrics_logger, "save_claims_report"):
                        metrics_logger.save_claims_report(demo_result_id, report)
                        st.success("✅ Saved claims report into this search")
                    else:
                        st.warning(
                            "Claims report generated, but save_claims_report() was not found. Showing for this session only."
                        )
                except Exception as e:
                    st.warning(f"Claims report generated but could not be saved: {e}")

                try:
                    from claims_analyzer import render_claims_report
                    render_claims_report(report)
                except Exception as e:
                    st.error(f"Claims report UI failed to render: {e}")
                    with st.expander("Raw claims report (debug)", expanded=False):
                        st.json(report)

    st.stop()