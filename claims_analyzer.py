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

def generate_claims_report(exa_result: Dict, claude_result: Dict, query: str, include_relevance: bool = True) -> str:
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
