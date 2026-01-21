"""
Adapted Information Retrieval metrics for research claim analysis

Translates traditional precision/recall to the research synthesis domain
where we have continuous relevance and unbounded information space.
"""

from typing import List, Dict
from anthropic import Anthropic
import os
import re
import streamlit as st

def decompose_query_to_info_needs(query: str) -> Dict:
    """
    Decompose query into specific information needs using Claude

    This creates a "ground truth" set of information units that should
    be present in a complete answer.

    Returns:
        {
            'primary_needs': ['funding amount', 'date', ...],  # Must-have info
            'secondary_needs': ['valuation', 'terms', ...],     # Nice-to-have
            'total_info_units': 6
        }
    """
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    prompt = f"""Analyze this research query and identify the specific information units needed for a complete answer.

Query: "{query}"

Break down what information is needed into:
1. PRIMARY needs (must-have information to answer the query)
2. SECONDARY needs (helpful context but not essential)

Format your response as:

PRIMARY:
- [Information unit 1]
- [Information unit 2]
...

SECONDARY:
- [Information unit 1]
- [Information unit 2]
...

Be specific. For example:
- Instead of "financial info", say "funding amount", "funding date", "lead investors"
- Instead of "company details", say "company founding date", "founder names", "headquarters location"
"""

    response = anthropic.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    # Parse response
    primary_section = re.search(r'PRIMARY:(.*?)(?=SECONDARY:|$)', response_text, re.DOTALL)
    secondary_section = re.search(r'SECONDARY:(.*?)$', response_text, re.DOTALL)

    primary_needs = []
    if primary_section:
        primary_needs = [line.strip('- ').strip() for line in primary_section.group(1).strip().split('\n') if line.strip()]

    secondary_needs = []
    if secondary_section:
        secondary_needs = [line.strip('- ').strip() for line in secondary_section.group(1).strip().split('\n') if line.strip()]

    return {
        'primary_needs': primary_needs,
        'secondary_needs': secondary_needs,
        'total_info_units': len(primary_needs) + len(secondary_needs),
        'token_count': response.usage.input_tokens + response.usage.output_tokens
    }

def map_claims_to_info_needs(claims: List[Dict], info_needs: Dict, query: str) -> Dict:
    """
    Map each claim to information needs it addresses

    Returns coverage metrics analogous to recall:
    - What % of information needs are addressed by claims?
    """
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    primary_needs = info_needs['primary_needs']
    secondary_needs = info_needs['secondary_needs']

    # Prepare claims list
    claims_text = "\n".join([f"{i+1}. {c['claim']}" for i, c in enumerate(claims)])

    prompt = f"""Map these claims to the information needs they address.

Query: "{query}"

PRIMARY Information Needs:
{chr(10).join([f"- {need}" for need in primary_needs])}

SECONDARY Information Needs:
{chr(10).join([f"- {need}" for need in secondary_needs])}

Claims to evaluate:
{claims_text}

For each information need, identify which claims (if any) address it.

Format your response as:

PRIMARY COVERAGE:
1. [Info need]: Claim(s) [#] - [Brief note on coverage quality]
2. [Info need]: NOT COVERED

SECONDARY COVERAGE:
1. [Info need]: Claim(s) [#] - [Brief note]
2. [Info need]: NOT COVERED

OFF-TOPIC CLAIMS:
- Claim [#] - [Brief reason why off-topic]

Example format:
1. funding amount: Claim 3 - Fully covered with specific $4B figure
2. funding date: Claim 3, Claim 5 - Multiple claims mention September 2024
3. investor identity: NOT COVERED
"""

    response = anthropic.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    # Parse coverage
    primary_covered = len(re.findall(r'Claim \d+', response_text.split('SECONDARY COVERAGE:')[0] if 'SECONDARY COVERAGE:' in response_text else response_text))
    primary_total = len(primary_needs)

    secondary_covered = 0
    if 'SECONDARY COVERAGE:' in response_text:
        secondary_section = response_text.split('SECONDARY COVERAGE:')[1]
        if 'OFF-TOPIC CLAIMS:' in secondary_section:
            secondary_section = secondary_section.split('OFF-TOPIC CLAIMS:')[0]
        secondary_covered = len(re.findall(r'Claim \d+', secondary_section))

    secondary_total = len(secondary_needs)

    # Count off-topic claims
    off_topic = 0
    if 'OFF-TOPIC CLAIMS:' in response_text:
        off_topic_section = response_text.split('OFF-TOPIC CLAIMS:')[1]
        off_topic = len(re.findall(r'Claim \d+', off_topic_section))

    # Calculate metrics
    total_needs = primary_total + secondary_total
    total_covered = primary_covered + secondary_covered

    recall = (total_covered / total_needs) if total_needs > 0 else 0
    primary_recall = (primary_covered / primary_total) if primary_total > 0 else 0

    precision = ((len(claims) - off_topic) / len(claims)) if len(claims) > 0 else 0

    return {
        'recall': round(recall, 3),  # % of information needs covered
        'primary_recall': round(primary_recall, 3),  # % of must-have info covered
        'precision': round(precision, 3),  # % of claims that are on-topic
        'coverage_map': response_text,
        'primary_covered': primary_covered,
        'primary_total': primary_total,
        'secondary_covered': secondary_covered,
        'secondary_total': secondary_total,
        'off_topic_claims': off_topic,
        'total_claims': len(claims),
        'token_count': response.usage.input_tokens + response.usage.output_tokens
    }

def calculate_adapted_ir_metrics(claims_list: List[Dict], query: str) -> Dict:
    """
    Calculate precision/recall adapted for research synthesis

    Process:
    1. Decompose query into information needs (creates "ground truth")
    2. Map claims to information needs (calculates coverage)
    3. Compute adapted metrics

    Returns:
        {
            'recall': 0.75,  # 75% of information needs covered
            'precision': 0.85,  # 85% of claims are on-topic
            'f1_score': 0.80,  # Harmonic mean
            'primary_recall': 0.90,  # Must-have info covered
            'redundancy': 1.2,  # Avg claims per info need
            'coverage_gaps': ['investor terms', 'valuation']  # Missing info
        }
    """
    # Step 1: Decompose query
    info_needs = decompose_query_to_info_needs(query)

    # Step 2: Map claims to needs
    coverage = map_claims_to_info_needs(claims_list, info_needs, query)

    # Step 3: Calculate composite metrics
    recall = coverage['recall']
    precision = coverage['precision']
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Calculate redundancy (how many claims per info need)
    covered_needs = coverage['primary_covered'] + coverage['secondary_covered']
    redundancy = (coverage['total_claims'] - coverage['off_topic_claims']) / covered_needs if covered_needs > 0 else 0

    return {
        'recall': recall,
        'precision': precision,
        'f1_score': round(f1_score, 3),
        'primary_recall': coverage['primary_recall'],
        'redundancy': round(redundancy, 2),
        'info_needs_covered': covered_needs,
        'info_needs_total': info_needs['total_info_units'],
        'on_topic_claims': coverage['total_claims'] - coverage['off_topic_claims'],
        'off_topic_claims': coverage['off_topic_claims'],
        'total_claims': coverage['total_claims'],
        'coverage_map': coverage['coverage_map'],
        'token_count': info_needs['token_count'] + coverage['token_count']
    }

def compare_approaches_ir_metrics(exa_claims: List, claude_claims: List, query: str) -> Dict:
    """
    Compare two approaches using adapted IR metrics

    Returns side-by-side precision/recall/F1 comparison
    """
    exa_metrics = calculate_adapted_ir_metrics(exa_claims, query)
    claude_metrics = calculate_adapted_ir_metrics(claude_claims, query)

    return {
        'exa': exa_metrics,
        'claude': claude_metrics,
        'comparison': {
            'recall_winner': 'exa' if exa_metrics['recall'] > claude_metrics['recall'] else 'claude',
            'precision_winner': 'exa' if exa_metrics['precision'] > claude_metrics['precision'] else 'claude',
            'f1_winner': 'exa' if exa_metrics['f1_score'] > claude_metrics['f1_score'] else 'claude',
            'recall_diff': round(exa_metrics['recall'] - claude_metrics['recall'], 3),
            'precision_diff': round(exa_metrics['precision'] - claude_metrics['precision'], 3),
            'f1_diff': round(exa_metrics['f1_score'] - claude_metrics['f1_score'], 3)
        }
    }


# ----------------------------
# Streamlit rendering helpers
# ----------------------------

def _safe_int(v, default=0):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v, default=0.0):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def _pct(n, d):
    return (n / d * 100.0) if d else 0.0


def _extract_claim_lists(report: dict):
    """Best-effort extraction of Exa+Claude claims and Claude WS claims from any report schema."""
    if not isinstance(report, dict):
        return [], []

    # Common schemas
    exa = report.get("exa") or report.get("exa_claude") or {}
    claude = report.get("claude") or report.get("claude_web") or report.get("claude_web_search") or {}

    exa_claims = (
        report.get("exa_claims")
        or exa.get("claims")
        or exa.get("claim_list")
        or report.get("claims_exa")
        or []
    )
    claude_claims = (
        report.get("claude_claims")
        or claude.get("claims")
        or claude.get("claim_list")
        or report.get("claims_claude")
        or []
    )

    # If report is already split by approach keys
    if isinstance(exa_claims, dict):
        exa_claims = exa_claims.get("claims") or []
    if isinstance(claude_claims, dict):
        claude_claims = claude_claims.get("claims") or []

    # Normalize to list
    if not isinstance(exa_claims, list):
        exa_claims = []
    if not isinstance(claude_claims, list):
        claude_claims = []

    return exa_claims, claude_claims


def _claim_status(claim: dict) -> str:
    """Return one of: supported | inference | unsupported | unknown"""
    if not isinstance(claim, dict):
        return "unknown"

    # Various field names used over time
    raw = (
        claim.get("status")
        or claim.get("support")
        or claim.get("support_status")
        or claim.get("classification")
        or claim.get("verdict")
        or ""
    )
    raw = str(raw).strip().lower()

    if raw in {"supported", "support", "yes", "true"}:
        return "supported"
    if raw in {"inference", "inferred", "inferences", "mixed"}:
        return "inference"
    if raw in {"unsupported", "no", "false"}:
        return "unsupported"

    # Some schemas store booleans
    if claim.get("supported") is True:
        return "supported"
    if claim.get("unsupported") is True:
        return "unsupported"

    return "unknown"


def _claim_tier(claim: dict) -> str:
    """Return A | B | C | ''"""
    if not isinstance(claim, dict):
        return ""

    raw = (
        claim.get("tier")
        or claim.get("source_tier")
        or claim.get("source_quality_tier")
        or claim.get("quality_tier")
        or ""
    )
    raw = str(raw).strip().upper()
    if raw in {"A", "B", "C"}:
        return raw

    # Sometimes stored as 'Tier A'
    if "A" in raw:
        return "A"
    if "B" in raw:
        return "B"
    if "C" in raw:
        return "C"

    return ""


def _claim_text(claim: dict) -> str:
    if not isinstance(claim, dict):
        return ""
    return (
        claim.get("claim")
        or claim.get("text")
        or claim.get("statement")
        or claim.get("summary")
        or ""
    )


def _claim_score_10(claim: dict) -> float | None:
    if not isinstance(claim, dict):
        return None
    for k in ["score", "confidence", "support_score", "quality_score", "evidence_score"]:
        if k in claim:
            v = _safe_float(claim.get(k), None)
            if v is None:
                continue
            # If a 0-1 score, convert to 0-10
            if 0 <= v <= 1:
                return round(v * 10.0, 1)
            # If already 0-10, keep
            if 0 <= v <= 10:
                return round(v, 1)
    return None


def _summarize_claims(claims: list[dict]):
    total = len(claims)
    supported = sum(1 for c in claims if _claim_status(c) == "supported")
    inferences = sum(1 for c in claims if _claim_status(c) == "inference")
    unsupported = sum(1 for c in claims if _claim_status(c) == "unsupported")

    tier_a = sum(1 for c in claims if _claim_status(c) == "supported" and _claim_tier(c) == "A")
    tier_b = sum(1 for c in claims if _claim_status(c) == "supported" and _claim_tier(c) == "B")
    tier_c = sum(1 for c in claims if _claim_status(c) == "supported" and _claim_tier(c) == "C")

    supported_total = supported if supported else 0
    weak_pct = _pct(tier_c, supported_total)

    return {
        "total": total,
        "supported": supported,
        "inferences": inferences,
        "unsupported": unsupported,
        "tier_a_supported": tier_a,
        "tier_b_supported": tier_b,
        "tier_c_supported": tier_c,
        "weak_supported_pct": weak_pct,
    }


def _render_claims_breakdown(exa_claims: list[dict], claude_claims: list[dict]):
    st.success("✅ Claims analysis complete")

    st.markdown("## 📋 Claims Breakdown")

    left, right = st.columns(2)

    exa_sum = _summarize_claims(exa_claims)
    claude_sum = _summarize_claims(claude_claims)

    with left:
        st.markdown("### 🔎 Exa + Claude Claims")
        st.metric("Total Claims", exa_sum["total"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("✅ Supported", exa_sum["supported"], help="Claims with direct supporting evidence in cited sources")
        with c2:
            st.metric("🟠 Inferences", exa_sum["inferences"], help="Claims that involve interpretation/synthesis beyond explicit citations")
        with c3:
            st.metric("⚠️ Unsupported", exa_sum["unsupported"], help="Claims that could not be backed by provided sources")

        st.markdown("**Supported Claims – Source Quality:**")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("🥇 Tier A", exa_sum["tier_a_supported"], help=".gov/.edu/primary regulators or high-authority sources")
        with q2:
            st.metric("🥈 Tier B", exa_sum["tier_b_supported"], help="Established industry publications / strong secondary sources")
        with q3:
            st.metric("🥉 Tier C", exa_sum["tier_c_supported"], help="Everything else")

    with right:
        st.markdown("### 🌐 Claude Web Search Claims")
        st.metric("Total Claims", claude_sum["total"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("✅ Supported", claude_sum["supported"])
        with c2:
            st.metric("🟠 Inferences", claude_sum["inferences"])
        with c3:
            st.metric("⚠️ Unsupported", claude_sum["unsupported"])

        st.markdown("**Supported Claims – Source Quality:**")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("🥇 Tier A", claude_sum["tier_a_supported"])
        with q2:
            st.metric("🥈 Tier B", claude_sum["tier_b_supported"])
        with q3:
            st.metric("🥉 Tier C", claude_sum["tier_c_supported"])

        if claude_sum["supported"] > 0:
            weak_pct = claude_sum["weak_supported_pct"]
            if weak_pct >= 50:
                st.warning(f"⚠️ Quality Issue: {weak_pct:.0f}% of supported claims use weak sources")


def _render_relevance_section(report: dict, exa_claims: list[dict], claude_claims: list[dict]):
    st.markdown("---")
    st.markdown("## 🎯 Relevance to Query")
    st.caption("How well does each approach answer the actual question?")

    # Support multiple report schemas
    rel = report.get("relevance") or report.get("relevance_metrics") or report.get("ir_metrics") or {}

    # Most common expected shape: { 'exa': {...}, 'claude': {...}, 'comparison': {...} }
    exa_rel = rel.get("exa") or {}
    claude_rel = rel.get("claude") or {}

    # Fallback: if not present, show a small note but do not error
    if not exa_rel and not claude_rel:
        st.info("Relevance metrics not present in this saved report. (Claims breakdown and inspection still available.)")
        return

    left, right = st.columns(2)

    def _render_one(title: str, d: dict):
        avg = d.get("average_relevance") or d.get("avg_relevance") or d.get("score")
        direct = d.get("direct_answers") or d.get("direct")
        tang = d.get("tangential") or d.get("partial")
        off = d.get("off_topic") or d.get("offtopic")

        st.markdown(f"### {title}")
        if avg is not None:
            st.metric("Average Relevance", f"{_safe_float(avg, 0.0):.1f}/10")
        c1, c2, c3 = st.columns(3)
        with c1:
            if direct is not None:
                st.metric("🎯 Direct Answers", _safe_int(direct))
        with c2:
            if tang is not None:
                st.metric("📊 Tangential", _safe_int(tang))
        with c3:
            if off is not None:
                st.metric("❌ Off-topic", _safe_int(off))

    with left:
        _render_one("🔎 Exa + Claude Relevance", exa_rel)
    with right:
        _render_one("🌐 Claude Web Search Relevance", claude_rel)

    comp = rel.get("comparison") or {}
    if comp:
        winner = comp.get("winner") or comp.get("more_relevant")
        diff = comp.get("diff") or comp.get("advantage")
        if winner and diff is not None:
            st.success(f"✅ {str(winner).title()} more relevant: {diff} point advantage")


def _render_coverage_section(report: dict):
    st.markdown("---")
    st.markdown("## 🔄 Coverage Comparison")

    cov = report.get("coverage") or report.get("coverage_comparison") or report.get("coverage_analysis")
    if not cov:
        # Some schemas store the bullet lists directly
        unique_exa = report.get("unique_to_exa")
        unique_claude = report.get("unique_to_claude")
        overlap = report.get("overlapping")
        if unique_exa or unique_claude or overlap:
            cov = {
                "unique_to_exa": unique_exa or [],
                "unique_to_claude": unique_claude or [],
                "overlapping": overlap or [],
            }

    if not cov:
        st.info("Coverage comparison not present in this saved report.")
        return

    unique_exa = cov.get("unique_to_exa") or []
    unique_claude = cov.get("unique_to_claude") or []
    overlapping = cov.get("overlapping") or cov.get("overlap") or []

    st.caption("What each approach found that the other did not.")

    if isinstance(unique_exa, str):
        unique_exa = [unique_exa]
    if isinstance(unique_claude, str):
        unique_claude = [unique_claude]
    if isinstance(overlapping, str):
        overlapping = [overlapping]

    if unique_exa:
        st.markdown("### Unique to Exa")
        for item in unique_exa:
            st.markdown(f"- {item}")

    if unique_claude:
        st.markdown("### Unique to Claude")
        for item in unique_claude:
            st.markdown(f"- {item}")

    if overlapping:
        st.markdown("### Overlapping Claims")
        for item in overlapping:
            st.markdown(f"- {item}")


def _render_quality_issues(report: dict):
    issues = report.get("quality_issues") or report.get("issues") or report.get("quality")
    if not issues:
        return

    st.markdown("---")
    st.markdown("## Quality Issues")

    # Accept either list of strings or dict with sections
    if isinstance(issues, list):
        for item in issues:
            st.markdown(f"- {item}")
        return

    if isinstance(issues, dict):
        for section, items in issues.items():
            st.markdown(f"### {section}")
            if isinstance(items, list):
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.write(items)


def _render_detailed_claims(exa_claims: list[dict], claude_claims: list[dict]):
    st.markdown("---")
    st.markdown("## 📝 Detailed Claims Inspection")

    tabs = st.tabs(["🔎 Exa + Claude Claims", "🌐 Claude Web Search Claims"])

    def _render_claim_list(claims: list[dict]):
        supported = [c for c in claims if _claim_status(c) == "supported"]
        inferred = [c for c in claims if _claim_status(c) == "inference"]
        unsupported = [c for c in claims if _claim_status(c) == "unsupported"]

        if supported:
            st.success(f"✅ Supported Claims ({len(supported)})")
            for i, c in enumerate(supported, start=1):
                score = _claim_score_10(c)
                tier = _claim_tier(c)
                hdr = f"Claim {i}"
                if tier:
                    hdr += f" - {tier} Source"
                if score is not None:
                    hdr += f" - 🎯 {score}/10"
                with st.expander(hdr):
                    st.write(_claim_text(c) or "(no claim text)")
                    # Show evidence/snippets if present
                    ev = c.get("evidence") or c.get("citations") or c.get("sources")
                    if ev:
                        st.markdown("**Evidence / citations:**")
                        st.write(ev)
                    note = c.get("note") or c.get("reason") or c.get("explanation")
                    if note:
                        st.markdown("**Notes:**")
                        st.write(note)

        if inferred:
            st.warning(f"🟠 Inferences ({len(inferred)})")
            for i, c in enumerate(inferred, start=1):
                with st.expander(f"Inference {i}"):
                    st.write(_claim_text(c) or "(no claim text)")
                    note = c.get("note") or c.get("reason") or c.get("explanation")
                    if note:
                        st.write(note)

        if unsupported:
            st.error(f"⚠️ Unsupported ({len(unsupported)})")
            for i, c in enumerate(unsupported, start=1):
                with st.expander(f"Unsupported {i}"):
                    st.write(_claim_text(c) or "(no claim text)")
                    note = c.get("note") or c.get("reason") or c.get("explanation")
                    if note:
                        st.write(note)

        if not claims:
            st.info("No claims found in this report.")

    with tabs[0]:
        _render_claim_list(exa_claims)

    with tabs[1]:
        _render_claim_list(claude_claims)


def render_claims_report(report: dict) -> None:
    """Render the claims report as Streamlit UI cards. Never raises."""
    try:
        exa_claims, claude_claims = _extract_claim_lists(report)

        _render_claims_breakdown(exa_claims, claude_claims)
        _render_relevance_section(report, exa_claims, claude_claims)
        _render_coverage_section(report)
        _render_quality_issues(report)
        _render_detailed_claims(exa_claims, claude_claims)

        # Optional: token usage from the report
        token_usage = report.get("token_usage") or report.get("token_count")
        if token_usage is not None:
            st.caption(f"Token usage: {_safe_int(token_usage, 0):,}")

    except Exception as e:
        # Absolute last-resort fallback
        st.error(f"Could not render claims report UI: {e}")
        st.json(report)


# Backwards-compatible alias (some callers may import this name)
render_claims_analysis = render_claims_report
