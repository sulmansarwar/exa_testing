"""
Adapted Information Retrieval metrics for research claim analysis

Translates traditional precision/recall to the research synthesis domain
where we have continuous relevance and unbounded information space.
"""

from typing import List, Dict
from anthropic import Anthropic
import os
import re

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
