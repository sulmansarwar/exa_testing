"""
Claims analyzer for qualitative comparison of research outputs
Extracts, structures, and compares claims from both Exa and Claude approaches
"""

import re
from typing import List, Dict, Tuple
from anthropic import Anthropic
import os


def classify_source_quality(url):
    """
    Classify source quality into tiers:
    A = .gov, .edu, major news outlets
    B = Industry publications, established domains
    C = Everything else
    """
    url_lower = url.lower()

    # Tier A: Government, education, major news, established investigative outlets
    tier_a_patterns = [
        '.gov', '.edu', '.mil',
        # International government/institutions
        'europa.eu', 'ecb.europa.eu', 'un.org', 'who.int', 'imf.org', 'worldbank.org',
        'oecd.org', 'wto.org', 'nato.int',
        # Major news outlets (US)
        'nytimes.com', 'washingtonpost.com', 'wsj.com', 'reuters.com',
        'apnews.com', 'npr.org', 'pbs.org', 'cnbc.com', 'cnn.com',
        'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'usatoday.com',
        # Major news outlets (International)
        'bbc.com', 'bbc.co.uk', 'theguardian.com', 'ft.com', 'economist.com',
        'reuters.com', 'afp.com', 'dw.com', 'aljazeera.com',
        # Business/Finance news
        'bloomberg.com', 'ft.com', 'marketwatch.com', 'barrons.com',
        # Investigative/Quality journalism
        'propublica.org', 'theatlantic.com', 'newyorker.com', 'politico.com',
        'axios.com', 'theintercept.com',
        # Academic/Scientific
        'nature.com', 'science.org', 'nejm.org', 'thelancet.com', 'cell.com',
        'pnas.org', 'sciencemag.org',
        # Tech news (high editorial standards)
        'wired.com', 'arstechnica.com'
    ]

    for pattern in tier_a_patterns:
        if pattern in url_lower:
            return 'A'

    # Tier B: Trade press, tech news sites, industry-specific outlets
    tier_b_patterns = [
        # Tech/Industry news
        'techcrunch.com', 'theverge.com', 'venturebeat.com', 'engadget.com',
        'gizmodo.com', 'mashable.com', 'zdnet.com', 'cnet.com',
        # Entertainment/Media trade
        'variety.com', 'hollywoodreporter.com', 'deadline.com',
        # Business/Marketing trade
        'adweek.com', 'inc.com', 'entrepreneur.com', 'fastcompany.com',
        # Variable quality (contributor models)
        'forbes.com', 'huffpost.com', 'medium.com',
        # News aggregators
        'vice.com', 'buzzfeednews.com', 'vox.com'
    ]

    for pattern in tier_b_patterns:
        if pattern in url_lower:
            return 'B'

    # Everything else is Tier C
    return 'C'

def extract_structured_claims(analysis_text: str, sources: List[Dict]) -> Dict:
    """
    Extract all claims from analysis text with their evidence and metadata

    Returns:
        dict with:
            - supported_claims: list of dicts with {claim, quote, url, tier}
            - inference_claims: list of dicts with {claim}
            - unsupported_claims: list of dicts with {claim}
    """


    if not analysis_text:
        return {
            'supported_claims': [],
            'inference_claims': [],
            'unsupported_claims': []
        }

    # 1. Extract SUPPORTED claims with structured format
    # Pattern: **Claim:** ... **Quote:** ... **Source:** URL
    supported_pattern = r'\*\*Claim:\*\*\s*(.*?)\s*\*\*Quote:\*\*\s*(.*?)\s*\*\*Source:\*\*\s*(https?://[^\s]+)'
    supported_matches = re.findall(supported_pattern, analysis_text, re.DOTALL)

    # Create URL to tier mapping
    url_to_tier = {}
    url_to_title = {}
    if sources:
        for source in sources:
            tier = classify_source_quality(source['url'])
            url_to_tier[source['url']] = tier
            url_to_title[source['url']] = source.get('title', 'Unknown')

    supported_claims = []
    for claim_text, quote_text, url in supported_matches:
        # Clean up text
        claim_text = claim_text.strip()
        quote_text = quote_text.strip().strip('"\'')
        url = url.strip().rstrip('.,;)')

        # Get tier
        tier = url_to_tier.get(url, classify_source_quality(url))
        title = url_to_title.get(url, 'Unknown Source')

        supported_claims.append({
            'claim': claim_text,
            'quote': quote_text,
            'url': url,
            'tier': tier,
            'source_title': title
        })

    # 2. Extract INFERENCE claims
    # Pattern: sentences or paragraphs with [INFERENCE] marker
    inference_pattern = r'([^\n]+)\s*\[INFERENCE\]'
    inference_matches = re.findall(inference_pattern, analysis_text)

    inference_claims = []
    for claim_text in inference_matches:
        claim_text = claim_text.strip().lstrip('-*• ')
        inference_claims.append({
            'claim': claim_text
        })

    # 3. Extract UNSUPPORTED claims
    # Find all **Claim:** statements that aren't part of supported structure
    all_claim_statements = re.findall(r'\*\*Claim:\*\*\s*([^\n]+)', analysis_text)

    # Filter out claims that are already in supported list
    supported_claim_texts = {c['claim'] for c in supported_claims}
    unsupported_claims = []
    for claim_text in all_claim_statements:
        claim_text = claim_text.strip()
        if claim_text not in supported_claim_texts:
            unsupported_claims.append({
                'claim': claim_text
            })

    return {
        'supported_claims': supported_claims,
        'inference_claims': inference_claims,
        'unsupported_claims': unsupported_claims
    }

def compare_claim_coverage(exa_claims: Dict, claude_claims: Dict, query: str) -> Dict:
    """
    Use Claude to analyze which claims overlap and which are unique

    Returns:
        dict with:
            - exa_unique: claims only in Exa
            - claude_unique: claims only in Claude
            - overlapping: claims covered by both (with different evidence)
            - quality_issues: list of problems identified
    """
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Prepare claim lists for comparison
    exa_supported = [f"- {c['claim']}" for c in exa_claims['supported_claims']]
    claude_supported = [f"- {c['claim']}" for c in claude_claims['supported_claims']]

    exa_inferences = [f"- {c['claim']}" for c in exa_claims['inference_claims']]
    claude_inferences = [f"- {c['claim']}" for c in claude_claims['inference_claims']]

    exa_unsupported = [f"- {c['claim']}" for c in exa_claims['unsupported_claims']]
    claude_unsupported = [f"- {c['claim']}" for c in claude_claims['unsupported_claims']]

    prompt = f"""You are analyzing two different research outputs for the same query.

Research Query: "{query}"

## Exa + Claude Approach Claims

### Supported Claims (with evidence):
{chr(10).join(exa_supported) if exa_supported else "None"}

### Inference Claims (analyst interpretation):
{chr(10).join(exa_inferences) if exa_inferences else "None"}

### Unsupported Claims (no evidence):
{chr(10).join(exa_unsupported) if exa_unsupported else "None"}

## Claude Web Search Approach Claims

### Supported Claims (with evidence):
{chr(10).join(claude_supported) if claude_supported else "None"}

### Inference Claims (analyst interpretation):
{chr(10).join(claude_inferences) if claude_inferences else "None"}

### Unsupported Claims (no evidence):
{chr(10).join(claude_unsupported) if claude_unsupported else "None"}

---

Analyze these claims and provide:

1. **Unique to Exa**: Claims/findings only present in Exa approach (provide claim numbers and brief description)
2. **Unique to Claude**: Claims/findings only present in Claude approach (provide claim numbers and brief description)
3. **Overlapping**: Claims covered by BOTH approaches (even if evidence differs)
4. **Quality Issues**: Specific problems you notice:
   - Critical unsupported claims (important statements without evidence)
   - Missing important angles (major aspects of the query not addressed)
   - Redundant or duplicate claims
   - Overly broad or vague claims

Format your response as:

## Unique to Exa
- [Brief description of unique claim/angle]
- [Another unique claim]

## Unique to Claude
- [Brief description of unique claim/angle]
- [Another unique claim]

## Overlapping Claims
- [Description of topic/claim covered by both]
- [Another overlapping topic]

## Quality Issues

### Exa Approach
- [Specific issue with claim or coverage]

### Claude Approach
- [Specific issue with claim or coverage]

### Overall
- [Any general quality observations]

Be specific and reference actual claims. Focus on qualitative assessment, not just counting."""

    response = anthropic.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    comparison_text = ""
    for block in response.content:
        if block.type == "text":
            comparison_text += block.text

    return {
        'comparison_text': comparison_text,
        'token_count': response.usage.input_tokens + response.usage.output_tokens
    }

def score_claim_relevance(claims: Dict, query: str) -> Dict:
    """
    Score each claim's relevance to the original query using Claude

    Returns:
        dict with:
            - relevance_scores: list of {claim, score, reasoning}
            - avg_relevance: average score across all claims
            - high_relevance_count: claims with score >= 8
            - low_relevance_count: claims with score < 5
    """
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Prepare all claims for scoring
    all_claims = []

    for claim in claims['supported_claims']:
        all_claims.append({
            'text': claim['claim'],
            'type': 'supported',
            'tier': claim['tier']
        })

    for claim in claims['inference_claims']:
        all_claims.append({
            'text': claim['claim'],
            'type': 'inference',
            'tier': 'N/A'
        })

    for claim in claims['unsupported_claims']:
        all_claims.append({
            'text': claim['claim'],
            'type': 'unsupported',
            'tier': 'N/A'
        })

    if not all_claims:
        return {
            'relevance_scores': [],
            'avg_relevance': 0,
            'high_relevance_count': 0,
            'low_relevance_count': 0,
            'directly_answers_query': 0,
            'tangentially_relevant': 0,
            'off_topic': 0
        }

    # Prepare claims list for Claude
    claims_list = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(all_claims)])

    prompt = f"""You are evaluating how relevant each claim is to the user's research query.

Research Query: "{query}"

Claims to evaluate:
{claims_list}

For each claim, score its relevance to the query on a scale of 0-10:

**10 = Directly answers the core question**
- This claim is exactly what the user asked for
- Provides specific, actionable information about the query

**7-9 = Highly relevant supporting information**
- Directly relates to the query
- Provides important context or details
- Helps understand the answer

**4-6 = Tangentially relevant**
- Related to the topic but not what was asked
- Provides background that may or may not be needed
- Somewhat helpful but not core to answering the query

**1-3 = Barely relevant**
- Loosely connected to topic
- Doesn't help answer the query
- More like general background or trivia

**0 = Off-topic**
- Unrelated to the query
- Doesn't belong in this research output

For each claim, provide:
1. Score (0-10)
2. Brief reasoning (one sentence)

Format your response as:

**Claim 1:** Score X/10 - [Reasoning]
**Claim 2:** Score X/10 - [Reasoning]
...

Be strict in your scoring. Most claims should be in the 4-8 range. Only give 9-10 if the claim directly answers the user's question."""

    response = anthropic.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    # Parse scores from response
    # Pattern: **Claim N:** Score X/10 - [reasoning]
    score_pattern = r'\*\*Claim (\d+):\*\*\s*Score\s*(\d+)/10\s*-\s*(.+?)(?=\*\*Claim|\Z)'
    matches = re.findall(score_pattern, response_text, re.DOTALL)

    relevance_scores = []
    total_score = 0
    high_relevance = 0
    low_relevance = 0
    directly_answers = 0  # 9-10
    tangentially = 0       # 4-6
    off_topic = 0          # 0-3

    for claim_num, score_str, reasoning in matches:
        idx = int(claim_num) - 1
        if idx < len(all_claims):
            score = int(score_str)
            total_score += score

            if score >= 8:
                high_relevance += 1
            if score >= 9:
                directly_answers += 1
            if score < 5:
                low_relevance += 1
            if 4 <= score <= 6:
                tangentially += 1
            if score <= 3:
                off_topic += 1

            relevance_scores.append({
                'claim': all_claims[idx]['text'],
                'claim_type': all_claims[idx]['type'],
                'tier': all_claims[idx]['tier'],
                'score': score,
                'reasoning': reasoning.strip()
            })

    avg_score = (total_score / len(relevance_scores)) if relevance_scores else 0

    return {
        'relevance_scores': relevance_scores,
        'avg_relevance': round(avg_score, 1),
        'high_relevance_count': high_relevance,
        'low_relevance_count': low_relevance,
        'directly_answers_query': directly_answers,
        'tangentially_relevant': tangentially,
        'off_topic': off_topic,
        'token_count': response.usage.input_tokens + response.usage.output_tokens
    }

def analyze_claim_quality(claims: Dict) -> Dict:
    """
    Analyze quality metrics for extracted claims

    Returns:
        dict with quality metrics and issues
    """
    supported = claims['supported_claims']
    inferences = claims['inference_claims']
    unsupported = claims['unsupported_claims']

    total_claims = len(supported) + len(inferences) + len(unsupported)

    # Quality metrics
    metrics = {
        'total_claims': total_claims,
        'supported_count': len(supported),
        'inference_count': len(inferences),
        'unsupported_count': len(unsupported),
        'supported_pct': (len(supported) / total_claims * 100) if total_claims > 0 else 0,
        'inference_pct': (len(inferences) / total_claims * 100) if total_claims > 0 else 0,
        'unsupported_pct': (len(unsupported) / total_claims * 100) if total_claims > 0 else 0
    }

    # Tier breakdown for supported claims
    tier_a = sum(1 for c in supported if c['tier'] == 'A')
    tier_b = sum(1 for c in supported if c['tier'] == 'B')
    tier_c = sum(1 for c in supported if c['tier'] == 'C')

    metrics['tier_breakdown'] = {
        'A': tier_a,
        'B': tier_b,
        'C': tier_c,
        'A_pct': (tier_a / len(supported) * 100) if supported else 0,
        'B_pct': (tier_b / len(supported) * 100) if supported else 0,
        'C_pct': (tier_c / len(supported) * 100) if supported else 0
    }

    # Identify specific issues
    issues = []

    # Issue: Too many unsupported claims
    if metrics['unsupported_pct'] > 20:
        issues.append({
            'severity': 'high',
            'type': 'unsupported_claims',
            'message': f"{metrics['unsupported_pct']:.0f}% of claims lack evidence"
        })

    # Issue: Too many inferences
    if metrics['inference_pct'] > 30:
        issues.append({
            'severity': 'medium',
            'type': 'high_inference_rate',
            'message': f"{metrics['inference_pct']:.0f}% of claims are analyst inferences"
        })

    # Issue: Too many Tier C sources
    if metrics['tier_breakdown']['C_pct'] > 40:
        issues.append({
            'severity': 'medium',
            'type': 'weak_sourcing',
            'message': f"{metrics['tier_breakdown']['C_pct']:.0f}% of supported claims use weak sources"
        })

    # Issue: Very few claims overall
    if total_claims < 5:
        issues.append({
            'severity': 'high',
            'type': 'insufficient_claims',
            'message': f"Only {total_claims} claims total - may be incomplete analysis"
        })

    metrics['issues'] = issues

    return metrics

def generate_claims_report(exa_result: Dict, claude_result: Dict, query: str, include_relevance: bool = True) -> Dict:
    """
    Generate a comprehensive claims comparison report

    Args:
        exa_result: dict with 'analysis_text' and 'sources'
        claude_result: dict with 'analysis_text' and 'sources'
        query: the research query
        include_relevance: whether to score claim relevance (adds ~3-5s and API cost)

    Returns:
        dict with all analysis results
    """
    # Extract claims
    exa_claims = extract_structured_claims(exa_result['analysis_text'], exa_result['sources'])
    claude_claims = extract_structured_claims(claude_result['analysis_text'], claude_result['sources'])

    # Analyze quality
    exa_quality = analyze_claim_quality(exa_claims)
    claude_quality = analyze_claim_quality(claude_claims)

    # Score relevance if requested
    exa_relevance = None
    claude_relevance = None
    if include_relevance:
        exa_relevance = score_claim_relevance(exa_claims, query)
        claude_relevance = score_claim_relevance(claude_claims, query)

    # Compare coverage
    coverage = compare_claim_coverage(exa_claims, claude_claims, query)

    return {
        'exa_claims': exa_claims,
        'claude_claims': claude_claims,
        'exa_quality': exa_quality,
        'claude_quality': claude_quality,
        'exa_relevance': exa_relevance,
        'claude_relevance': claude_relevance,
        'coverage_comparison': coverage
    }


def render_claims_report(report: Dict):
    """Render the claims report as metrics + structured sections (avoid dumping raw JSON by default)."""
    import streamlit as st

    if not report or not isinstance(report, dict):
        st.info("No claims report available.")
        return

    def _pct(n: int, d: int) -> float:
        d = max(1, int(d or 0))
        return (float(n or 0) / d) * 100.0

    def _safe_list(x):
        return x if isinstance(x, list) else []

    def _safe_dict(x):
        return x if isinstance(x, dict) else {}

    # Primary current shape from generate_claims_report()
    exa_claims = _safe_dict(report.get("exa_claims"))
    claude_claims = _safe_dict(report.get("claude_claims"))
    exa_quality = _safe_dict(report.get("exa_quality"))
    claude_quality = _safe_dict(report.get("claude_quality"))
    exa_relevance = report.get("exa_relevance")
    claude_relevance = report.get("claude_relevance")
    coverage = _safe_dict(report.get("coverage_comparison"))

    def _headline_from_claims(claims: Dict) -> Dict:
        supported = _safe_list(claims.get("supported_claims"))
        inferences = _safe_list(claims.get("inference_claims"))
        unsupported = _safe_list(claims.get("unsupported_claims"))
        total = len(supported) + len(inferences) + len(unsupported)
        tier_a = sum(1 for c in supported if (c.get("tier") == "A"))
        tier_b = sum(1 for c in supported if (c.get("tier") == "B"))
        tier_c = sum(1 for c in supported if (c.get("tier") == "C"))
        return {
            "total": total,
            "supported": len(supported),
            "inferences": len(inferences),
            "unsupported": len(unsupported),
            "tier_a": tier_a,
            "tier_b": tier_b,
            "tier_c": tier_c,
        }

    def _headline_from_quality(quality: Dict, claims: Dict) -> Dict:
        # Prefer precomputed quality fields
        total = int(quality.get("total_claims") or 0)
        supported = int(quality.get("supported_count") or 0)
        inferences = int(quality.get("inference_count") or 0)
        unsupported = int(quality.get("unsupported_count") or 0)

        # Fallback to raw claims if the quality block is missing
        if total <= 0 and (claims.get("supported_claims") or claims.get("inference_claims") or claims.get("unsupported_claims")):
            return _headline_from_claims(claims)

        if total <= 0:
            total = supported + inferences + unsupported

        tb = _safe_dict(quality.get("tier_breakdown"))
        return {
            "total": total,
            "supported": supported,
            "inferences": inferences,
            "unsupported": unsupported,
            "tier_a": int(tb.get("A") or 0),
            "tier_b": int(tb.get("B") or 0),
            "tier_c": int(tb.get("C") or 0),
        }

    exa_h = _headline_from_quality(exa_quality, exa_claims)
    claude_h = _headline_from_quality(claude_quality, claude_claims)

    # ===== Claims Breakdown =====
    st.markdown("## 📋 Claims Breakdown")

    left, right = st.columns(2)

    def _render_claims_breakdown(col, title: str, h: Dict):
        with col:
            st.markdown(f"### {title}")
            st.metric("Total Claims", f"{h['total']}")
            a, b, c = st.columns(3)
            a.metric("✅ Supported", f"{h['supported']}")
            b.metric("🔶 Inferences", f"{h['inferences']}")
            c.metric("⚠️ Unsupported", f"{h['unsupported']}")

            trace = _pct(h["supported"], h["total"]) if h["total"] else 0.0
            st.metric("📌 Traceability", f"{trace:.1f}%")

            if h["supported"] > 0:
                st.caption(f"Tier A: {h['tier_a']}/{h['supported']} ({_pct(h['tier_a'], h['supported']):.0f}%)")
                st.caption(f"Tier B: {h['tier_b']}/{h['supported']} ({_pct(h['tier_b'], h['supported']):.0f}%)")
                st.caption(f"Tier C: {h['tier_c']}/{h['supported']} ({_pct(h['tier_c'], h['supported']):.0f}%)")

    _render_claims_breakdown(left, "🔎 Exa + Claude Claims", exa_h)
    _render_claims_breakdown(right, "🌐 Claude Web Search Claims", claude_h)

    # ===== Relevance to Query =====
    st.markdown("---")
    st.markdown("## 🎯 Relevance to Query")
    st.caption("How well does each approach answer the actual question?")

    def _render_relevance(col, title: str, rel: Dict | None):
        with col:
            st.markdown(f"### {title}")
            if not rel or not isinstance(rel, dict):
                st.info("No relevance scoring found in this saved report.")
                return
            st.metric("Average Relevance", f"{rel.get('avg_relevance', 0)}/10")
            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Direct Answers", f"{rel.get('directly_answers_query', 0)}")
            c2.metric("📊 Tangential", f"{rel.get('tangentially_relevant', 0)}")
            c3.metric("❌ Off-topic", f"{rel.get('off_topic', 0)}")
            tok = rel.get("token_count")
            if tok is not None:
                st.caption(f"Token usage: {int(tok):,}")

    l2, r2 = st.columns(2)
    _render_relevance(l2, "🔎 Exa + Claude Relevance", exa_relevance)
    _render_relevance(r2, "🌐 Claude Web Search Relevance", claude_relevance)

    # ===== Coverage Comparison =====
    st.markdown("---")
    st.markdown("## 🔁 Coverage Comparison")

    comparison_text = (coverage.get("comparison_text") or "").strip()
    if comparison_text:
        st.markdown(comparison_text)
        tok = coverage.get("token_count")
        if tok is not None:
            st.caption(f"Token usage: {int(tok):,}")
    else:
        st.info("No coverage comparison text found in this report.")

    # ===== Quality Issues =====
    st.markdown("---")
    st.markdown("## ⚠️ Quality Issues")

    def _render_issues(title: str, quality: Dict):
        st.markdown(f"### {title}")
        issues = _safe_list(quality.get("issues"))
        if not issues:
            st.success("No major quality issues flagged.")
            return
        for it in issues:
            msg = (it.get("message") or "").strip() or str(it)
            sev = (it.get("severity") or "").lower()
            if sev == "high":
                st.error(msg)
            elif sev == "medium":
                st.warning(msg)
            else:
                st.info(msg)

    _render_issues("Exa Approach", exa_quality)
    _render_issues("Claude Approach", claude_quality)

    # ===== Detailed Claims Inspection =====
    st.markdown("---")
    st.markdown("## 📝 Detailed Claims Inspection")

    def _relevance_map(rel: Dict | None) -> Dict[str, Dict]:
        """Map claim text -> relevance record."""
        if not rel or not isinstance(rel, dict):
            return {}
        scores = _safe_list(rel.get("relevance_scores"))
        out = {}
        for r in scores:
            txt = (r.get("claim") or "").strip()
            if txt:
                out[txt] = r
        return out

    exa_rel_map = _relevance_map(exa_relevance)
    claude_rel_map = _relevance_map(claude_relevance)

    def _render_claim_list(prefix: str, claims: Dict, rel_map: Dict[str, Dict]):
        supported = _safe_list(claims.get("supported_claims"))
        inferences = _safe_list(claims.get("inference_claims"))
        unsupported = _safe_list(claims.get("unsupported_claims"))

        st.markdown(f"### ✅ Supported Claims ({len(supported)})")
        if not supported:
            st.caption("No supported claims found.")
        for i, c in enumerate(supported, start=1):
            claim_txt = (c.get("claim") or "").strip()
            tier = (c.get("tier") or "?")
            score = rel_map.get(claim_txt, {}).get("score")
            score_txt = f" • 🎯 {score}/10" if score is not None else ""
            header = f"Claim {i} (Tier {tier}){score_txt}"
            with st.expander(header, expanded=False):
                if claim_txt:
                    st.markdown(f"**Claim:** {claim_txt}")
                quote = (c.get("quote") or "").strip()
                if quote:
                    st.markdown(f"**Quote:** {quote}")
                url = (c.get("url") or "").strip()
                if url:
                    st.markdown(f"**Source:** {url}")
                title = (c.get("source_title") or "").strip()
                if title and title != "Unknown Source":
                    st.caption(f"Source title: {title}")
                reasoning = (rel_map.get(claim_txt, {}).get("reasoning") or "").strip()
                if reasoning:
                    st.caption(f"Relevance note: {reasoning}")

        st.markdown(f"### 🔶 Inference Claims ({len(inferences)})")
        if inferences:
            for i, c in enumerate(inferences, start=1):
                claim_txt = (c.get("claim") or "").strip()
                score = rel_map.get(claim_txt, {}).get("score")
                score_txt = f" • 🎯 {score}/10" if score is not None else ""
                with st.expander(f"Inference {i}{score_txt}", expanded=False):
                    if claim_txt:
                        st.markdown(claim_txt)
                    reasoning = (rel_map.get(claim_txt, {}).get("reasoning") or "").strip()
                    if reasoning:
                        st.caption(f"Relevance note: {reasoning}")
        else:
            st.caption("No inference claims found.")

        st.markdown(f"### ⚠️ Unsupported Claims ({len(unsupported)})")
        if unsupported:
            for i, c in enumerate(unsupported, start=1):
                claim_txt = (c.get("claim") or "").strip()
                score = rel_map.get(claim_txt, {}).get("score")
                score_txt = f" • 🎯 {score}/10" if score is not None else ""
                with st.expander(f"Unsupported {i}{score_txt}", expanded=False):
                    if claim_txt:
                        st.markdown(claim_txt)
                    reasoning = (rel_map.get(claim_txt, {}).get("reasoning") or "").strip()
                    if reasoning:
                        st.caption(f"Relevance note: {reasoning}")
        else:
            st.caption("No unsupported claims found.")

    tab1, tab2 = st.tabs(["🔎 Exa + Claude Claims", "🌐 Claude Web Search Claims"])
    with tab1:
        _render_claim_list("exa", exa_claims, exa_rel_map)
    with tab2:
        _render_claim_list("claude", claude_claims, claude_rel_map)

    # Raw JSON is still available for debugging but collapsed by default
    with st.expander("Raw claims report (debug)", expanded=False):
        st.json(report)


def render_claims_report(report: dict) -> None:
    """Render the claims report in Streamlit.

    This is a pure UI function. It does not re-run retrieval or call any APIs.
    """
    try:
        import streamlit as st
    except Exception:
        # If Streamlit isn't available, fall back silently
        return

    if not isinstance(report, dict) or not report:
        st.info("No claims report to display.")
        return

    exa_quality = report.get("exa_quality") or {}
    claude_quality = report.get("claude_quality") or {}
    exa_relevance = report.get("exa_relevance") or {}
    claude_relevance = report.get("claude_relevance") or {}
    coverage = report.get("coverage_comparison") or {}

    def _pct(part: int, whole: int) -> float:
        try:
            if not whole:
                return 0.0
            return round((float(part) / float(whole)) * 100.0, 1)
        except Exception:
            return 0.0

    def _safe_int(x, default=0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _safe_float(x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    st.success("✅ Claims analysis complete")

    st.markdown("# 📋 Claims Breakdown")

    left, right = st.columns(2)

    def _render_quality_col(col, title: str, q: dict):
        total = _safe_int(q.get("total_claims"), 0)
        supported = _safe_int(q.get("supported_count"), 0)
        inf = _safe_int(q.get("inference_count"), 0)
        unsup = _safe_int(q.get("unsupported_count"), 0)
        tiers = (q.get("tier_breakdown") or {})
        tier_a = _safe_int(tiers.get("A"), 0)
        tier_b = _safe_int(tiers.get("B"), 0)
        tier_c = _safe_int(tiers.get("C"), 0)

        traceability = _pct(supported, total)

        with col:
            st.markdown(f"## {title}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Claims", f"{total}")
            with c2:
                st.metric("✅ Supported", f"{supported}", f"{_pct(supported, total):.0f}%")
            with c3:
                st.metric("⚠️ Unsupported", f"{unsup}", f"{_pct(unsup, total):.0f}%")

            c4, c5 = st.columns(2)
            with c4:
                st.metric("🔶 Inferences", f"{inf}", f"{_pct(inf, total):.0f}%")
            with c5:
                st.metric("📌 Traceability", f"{traceability:.1f}%")

            st.markdown("**Supported Claims – Source Quality:**")
            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("🥇 Tier A", f"{tier_a}", f"{_pct(tier_a, max(1, supported)):.0f}%")
            with t2:
                st.metric("🥈 Tier B", f"{tier_b}", f"{_pct(tier_b, max(1, supported)):.0f}%")
            with t3:
                st.metric("🥉 Tier C", f"{tier_c}", f"{_pct(tier_c, max(1, supported)):.0f}%")

            issues = q.get("issues") or []
            if issues:
                st.markdown("### ⚠️ Quality Issues")
                for it in issues:
                    msg = (it.get("message") or "").strip()
                    if msg:
                        st.warning(msg)

    _render_quality_col(left, "🔍 Exa + Claude Claims", exa_quality)
    _render_quality_col(right, "🌐 Claude Web Search Claims", claude_quality)

    # Relevance section
    if exa_relevance or claude_relevance:
        st.markdown("---")
        st.markdown("# 🎯 Relevance to Query")
        st.caption("How well does each approach answer the actual question?")

        rl, rr = st.columns(2)

        def _render_relevance(col, title: str, r: dict):
            avg_rel = _safe_float(r.get("avg_relevance"), 0.0)
            direct = _safe_int(r.get("directly_answers_query"), 0)
            tang = _safe_int(r.get("tangentially_relevant"), 0)
            off = _safe_int(r.get("off_topic"), 0)
            with col:
                st.markdown(f"## {title}")
                st.metric("Average Relevance", f"{avg_rel}/10")
                a, b, c = st.columns(3)
                with a:
                    st.metric("🎯 Direct Answers", f"{direct}")
                with b:
                    st.metric("📊 Tangential", f"{tang}")
                with c:
                    st.metric("❌ Off-topic", f"{off}")

        _render_relevance(rl, "🔍 Exa + Claude Relevance", exa_relevance)
        _render_relevance(rr, "🌐 Claude Web Search Relevance", claude_relevance)

    # Coverage comparison (full width)
    if (coverage.get("comparison_text") or "").strip():
        st.markdown("---")
        st.markdown("# 🔄 Coverage Comparison")
        tok = coverage.get("token_count")
        if tok is not None:
            st.caption(f"Token usage: {tok}")
        st.markdown(coverage.get("comparison_text") or "")

    # Detailed inspection
    st.markdown("---")
    st.markdown("# 📝 Detailed Claims Inspection")

    exa_claims = (report.get("exa_claims") or {})
    claude_claims = (report.get("claude_claims") or {})

    tabs = st.tabs(["🔍 Exa + Claude Claims", "🌐 Claude Web Search Claims"])

    def _render_claim_list(container, claims_dict: dict, relevance_dict: dict | None):
        supported = claims_dict.get("supported_claims") or []
        inferences = claims_dict.get("inference_claims") or []
        unsupported = claims_dict.get("unsupported_claims") or []

        # Build a score lookup by claim text
        score_lookup = {}
        if isinstance(relevance_dict, dict):
            for rs in (relevance_dict.get("relevance_scores") or []):
                txt = (rs.get("claim") or "").strip()
                if txt:
                    score_lookup[txt] = rs

        with container:
            st.markdown(f"## ✅ Supported Claims ({len(supported)})")
            for i, c in enumerate(supported, 1):
                ct = (c.get("claim") or "").strip()
                tier = (c.get("tier") or "").strip()
                url = (c.get("url") or "").strip()
                quote = (c.get("quote") or "").strip()
                rs = score_lookup.get(ct) or {}
                score = rs.get("score")
                header = f"Claim {i}"
                if tier:
                    header += f" - {tier} Source"
                if score is not None:
                    header += f" - 🎯 {score}/10"
                with st.expander(header, expanded=False):
                    if ct:
                        st.markdown(f"**Claim:** {ct}")
                    if quote:
                        st.markdown(f"**Quote:** {quote}")
                    if url:
                        st.markdown(f"**Source:** {url}")
                    if rs.get("reasoning"):
                        st.markdown(f"**Relevance reasoning:** {rs.get('reasoning')}")

            st.markdown(f"## 🔶 Inferences ({len(inferences)})")
            for i, c in enumerate(inferences, 1):
                ct = (c.get("claim") or "").strip()
                rs = score_lookup.get(ct) or {}
                score = rs.get("score")
                header = f"Inference {i}"
                if score is not None:
                    header += f" - 🎯 {score}/10"
                with st.expander(header, expanded=False):
                    if ct:
                        st.markdown(ct)
                    if rs.get("reasoning"):
                        st.markdown(f"**Relevance reasoning:** {rs.get('reasoning')}")

            st.markdown(f"## ⚠️ Unsupported ({len(unsupported)})")
            for i, c in enumerate(unsupported, 1):
                ct = (c.get("claim") or "").strip()
                rs = score_lookup.get(ct) or {}
                score = rs.get("score")
                header = f"Unsupported {i}"
                if score is not None:
                    header += f" - 🎯 {score}/10"
                with st.expander(header, expanded=False):
                    if ct:
                        st.markdown(ct)
                    if rs.get("reasoning"):
                        st.markdown(f"**Relevance reasoning:** {rs.get('reasoning')}")

    _render_claim_list(tabs[0], exa_claims, exa_relevance)
    _render_claim_list(tabs[1], claude_claims, claude_relevance)

    # Optional debug
    with st.expander("Raw claims report (debug)", expanded=False):
        st.json(report)