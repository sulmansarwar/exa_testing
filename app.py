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
from metrics_logger import log_comparison, get_total_stats, get_detailed_history, load_result

# Load environment variables
load_dotenv()

# Initialize clients
@st.cache_resource
def get_clients():
    exa = Exa(api_key=os.environ.get("EXA_API_KEY"))
    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return exa, anthropic

exa, anthropic = get_clients()

# --- HELPER FUNCTIONS FOR METRICS ---

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
        'tokens': input_tokens + output_tokens,
        'cost_usd': total_cost
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
    st.markdown("#### 📊 Retrieval Scorecard")

    # Quality Mix
    st.markdown("**Source Quality Distribution:**")

    # Calculate percentages
    total_sources = metrics['total_sources']
    tier_a_pct = (metrics['quality_mix']['A'] / total_sources * 100) if total_sources > 0 else 0
    tier_b_pct = (metrics['quality_mix']['B'] / total_sources * 100) if total_sources > 0 else 0
    tier_c_pct = (metrics['quality_mix']['C'] / total_sources * 100) if total_sources > 0 else 0

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("🥇 Tier A",
                  f"{metrics['quality_mix']['A']} ({tier_a_pct:.0f}%)",
                  help="Gov/edu/major news")
    with col_b:
        st.metric("🥈 Tier B",
                  f"{metrics['quality_mix']['B']} ({tier_b_pct:.0f}%)",
                  help="Industry pubs")
    with col_c:
        st.metric("🥉 Tier C",
                  f"{metrics['quality_mix']['C']} ({tier_c_pct:.0f}%)",
                  help="Other sources")

    # Independence & Coverage
    st.markdown("**Independence & Coverage:**")

    # Calculate domain diversity percentage
    domain_diversity_pct = (metrics['unique_domains'] / total_sources * 100) if total_sources > 0 else 0

    col_ind, col_cov = st.columns(2)
    with col_ind:
        st.metric("🌐 Unique Domains",
                  f"{metrics['unique_domains']} ({domain_diversity_pct:.0f}%)",
                  help="Domain diversity: higher % = more independent sources")
    with col_cov:
        st.metric("📚 Total Sources", metrics['total_sources'])

    # Evidence Traceability (if analysis text provided)
    if analysis_text:
        traceability = calculate_evidence_traceability(analysis_text, sources)

        st.markdown("**Evidence Traceability:**")

        # Main traceability metric
        supported = traceability['supported_claims']
        total = traceability['total_claims']
        trace_pct = traceability['traceability_rate']

        st.metric(
            "✅ Traceability",
            f"{supported}/{total} ({trace_pct:.0f}%)",
            help="Claims backed by direct quotes + URLs"
        )

        # Tier breakdown of supported claims
        if supported > 0:
            tier_a = traceability.get('tier_a_claims', 0)
            tier_b = traceability.get('tier_b_claims', 0)
            tier_c = traceability.get('tier_c_claims', 0)

            tier_a_pct = (tier_a / supported * 100) if supported > 0 else 0
            tier_b_pct = (tier_b / supported * 100) if supported > 0 else 0
            tier_c_pct = (tier_c / supported * 100) if supported > 0 else 0

            st.caption(f"🥇 Tier-A backed claims: {tier_a}/{supported} ({tier_a_pct:.0f}%)")
            st.caption(f"🥈 Tier-B backed claims: {tier_b}/{supported} ({tier_b_pct:.0f}%)")
            st.caption(f"🥉 Tier-C backed claims: {tier_c}/{supported} ({tier_c_pct:.0f}%)")

        # Other claim categories
        if total > 0:
            inferences = traceability['inferences']
            unsupported = traceability['unsupported_claims']

            if inferences > 0:
                inf_pct = (inferences / total * 100)
                st.caption(f"🔶 Inferences: {inferences}/{total} ({inf_pct:.0f}%)")

            if unsupported > 0:
                unsupp_pct = (unsupported / total * 100)
                st.caption(f"⚠️ Unsupported: {unsupported}/{total} ({unsupp_pct:.0f}%)")

            # Warning for unsupported claims
            if unsupported > 0:
                st.warning(f"⚠️ {unsupported} unsupported claim(s) need verification or evidence")

    # Cost & Tokens
    st.markdown("**Resource Usage:**")
    col_tok, col_cost = st.columns(2)
    with col_tok:
        st.metric("🔢 Tokens", f"{metrics['tokens']:,}")
    with col_cost:
        cost_display = f"${metrics['cost_usd']:.4f}"
        # Add Exa search cost if applicable
        if exa_search_count > 0:
            exa_cost = (exa_search_count / 1000) * 5.0  # $5 per 1000 searches
            total_with_exa = metrics['cost_usd'] + exa_cost
            cost_display = f"${total_with_exa:.4f}"
            st.metric("💰 Total Cost", cost_display,
                      help=f"Claude: ${metrics['cost_usd']:.4f} + Exa: ${exa_cost:.4f}")
        else:
            st.metric("💰 Cost (Claude)", cost_display)

# Page config
st.set_page_config(
    page_title="Exa + Claude vs Claude Web Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Exa + Claude vs Claude Web Search")
st.markdown("**Interactive Comparison Tool** - Test different prompts and search configurations")

# --- COMPARISON CRITERIA SECTION ---
with st.expander("📊 What You're Measuring (Comparison Criteria)", expanded=True):
    st.markdown("""
This tool compares **two research approaches** side-by-side. Here's what to look for:

### 🎯 **Key Metrics Explained**

#### **1. Source Quality Distribution (Tier A/B/C)**
- **🥇 Tier A** = Government (.gov), Education (.edu), Major News (NYT, WSJ, Reuters, BBC, etc.)
- **🥈 Tier B** = Industry Publications (TechCrunch, The Verge, trade press)
- **🥉 Tier C** = Everything else (blogs, niche sites, forums)

**What to look for:**
- ✅ **Higher Tier A %** = More authoritative, fact-checkable sources
- ⚠️ **Higher Tier C %** = Less verified, potentially biased sources
- 📰 **For newsroom use**: Aim for 60%+ Tier A sources

---

#### **2. Source Independence (Unique Domains)**
**What it measures:** How many different websites were used (not just different articles from the same site)

**What to look for:**
- ✅ **More unique domains** = Diverse perspectives, less echo chamber risk
- ⚠️ **Few unique domains** = Limited viewpoint, potential bias
- 📰 **For newsroom use**: Aim for 7+ unique domains for investigative work

**Example:**
- 10 articles from NYT = 1 unique domain ❌
- 1 article each from NYT, WSJ, BBC, Reuters, AP, Guardian, Bloomberg, NPR, Politico, Axios = 10 unique domains ✅

---

#### **3. Total Sources**
**What it measures:** How many articles/documents were retrieved and analyzed

**What to look for:**
- ✅ **More sources** = Broader evidence base, better pattern detection
- ⚠️ **Too few sources** = Incomplete picture, missed contradictions
- 📰 **For newsroom use**: 8-15 sources typical for fact-checking, 20+ for investigations

---

#### **4. Cost & Tokens**
**What it measures:** How much the search cost in API usage

**Breakdown:**
- **Exa Cost** = Search API ($5/1000 searches) + Claude tokens
- **Claude Web Search Cost** = Claude tokens only (no search API fees)

**What to look for:**
- ✅ **Cost-effectiveness**: Is the quality improvement worth the price difference?
- 📰 **For newsroom budgets**: Track cost per investigation in Comparison History

---

### 🔍 **What Makes a "Better" Result?**

A better research result typically has:
1. **✅ Higher Tier A percentage** (60%+ for fact-checking)
2. **✅ More unique domains** (7+ for diverse perspectives)
3. **✅ Sufficient sources** (10+ for comprehensive coverage)
4. **✅ Finds contradictions** (not just confirming one narrative)
5. **✅ Provides actionable insights** (next steps, gaps in coverage)

---

### 💡 **Reading the Comparison**

After running a comparison, scroll down to see:
1. **Left column** (🔷 Exa + Claude): Shows Exa's semantic search results
2. **Right column** (🔶 Claude Web Search): Shows Claude's built-in web search
3. **Compare the scorecards**: Which has better source quality? More diversity?
4. **Read both analyses**: Which found contradictions? Which missed key sources?
5. **Check Comparison History**: Track metrics over time, optimize your workflow
""")

# Sidebar for Exa configuration
with st.sidebar:
    st.header("⚙️ Exa Search Configuration")

    search_type = st.selectbox(
        "Search Type",
        ["auto", "neural", "keyword"],
        help="Neural: Semantic/meaning-based. Keyword: Traditional. Auto: Best of both."
    )

    num_results = st.slider(
        "Number of Results",
        min_value=3,
        max_value=20,
        value=10,
        help="How many sources to retrieve"
    )

    max_characters = st.slider(
        "Max Characters per Source",
        min_value=500,
        max_value=3000,
        value=1500,
        step=100,
        help="Amount of text to extract from each source"
    )

    st.markdown("---")

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        use_autoprompt = st.checkbox(
            "Use Autoprompt",
            value=True,
            help="Let Exa optimize your query"
        )

        include_domains = st.text_area(
            "Include Domains (one per line)",
            placeholder="fda.gov\ncdc.gov\npubmed.gov",
            help="Only search these domains (leave empty for all)"
        )

        exclude_domains = st.text_area(
            "Exclude Domains (one per line)",
            placeholder="example.com\nspam.com",
            help="Never search these domains"
        )

        start_published_date = st.date_input(
            "Start Date",
            value=None,
            help="Only articles published after this date"
        )

    st.markdown("---")
    st.caption("💡 **Tip**: Try semantic search for cross-domain patterns, keyword for specific terms")

# Main search interface
st.markdown("### 🎯 Enter Your Research Question")

# Example Prompts Section
# Category selector (outside expander so it's always visible)
query_category = st.selectbox(
    "Query Category (for pattern analysis)",
    [
        "Fact-Checking & Verification",
        "Source Diversification",
        "Timeline Construction",
        "Document Analysis",
        "Expert Source Finding",
        "Follow-the-Money",
        "Comparative Reporting",
        "Trend Analysis",
        "Investigative Patterns"
    ],
    help="Categorize your query to identify which types Exa+Claude vs Claude WS excel at"
)

with st.expander("📰 Example Newsroom Prompts", expanded=False):

    example_prompts = {
        "Fact-Checking & Verification": [
            "Verify claims that Meta laid off 15% of its workforce today - find official sources",
            "Cross-reference reports about the Federal Reserve emergency meeting - what do official sources say?",
            "Fact-check viral claims that California banned gas stoves - what does the actual legislation say?"
        ],
        "Source Diversification": [
            "Find perspectives on the rail workers strike from labor unions, railroad companies, and government regulators",
            "Get viewpoints on the TikTok ban from civil liberties groups, national security experts, and tech policy researchers",
            "Compare how conservative and progressive outlets are covering the abortion pill court decision"
        ],
        "Timeline Construction": [
            "Build a timeline of Boeing 737 MAX safety concerns from first reports to current FAA statements",
            "Track the evolution of SVB collapse from first warning signs to FDIC takeover",
            "Map the progression of the Norfolk Southern train derailment story from incident to EPA response"
        ],
        "Document Analysis": [
            "Find SEC filings and investor disclosures about FTX in the months before bankruptcy",
            "Locate FDA approval documents and clinical trial data for the new Alzheimer's drug",
            "Find EPA reports on East Palestine water quality after the train derailment"
        ],
        "Expert Source Finding": [
            "Find academic researchers who have published on ChatGPT's impact on education",
            "Identify economists who predicted the 2023 banking crisis in previous papers",
            "Locate climate scientists who have expertise in methane emissions from agriculture"
        ],
        "Follow-the-Money": [
            "Track political donations from crypto executives to members of Congress",
            "Find lobbying disclosures from pharmaceutical companies on insulin pricing legislation",
            "Trace dark money groups funding ads in the 2024 presidential race"
        ],
        "Comparative Reporting": [
            "How does US drug pricing compare to Canada, UK, and Germany for the same medications?",
            "Compare police use-of-force policies across major US cities",
            "Compare AI safety frameworks proposed by EU, UK, and US regulators"
        ],
        "Trend Analysis": [
            "Are mass shootings increasing? Find FBI, CDC, and academic research data",
            "Track trends in remote work policies among Fortune 500 companies 2020-2024",
            "Track corporate diversity hiring trends since George Floyd protests"
        ],
        "Investigative Patterns": [
            "Find patterns in nursing home violations across states before and after COVID",
            "Identify companies with repeat OSHA violations in warehouse safety",
            "Find patterns in police misconduct settlements by department and type of complaint"
        ]
    }

    selected_prompt = st.selectbox(
        "Select an example",
        example_prompts[query_category]
    )

    if st.button("Use this prompt", key="use_example_prompt"):
        st.session_state.selected_example_prompt = selected_prompt
        st.rerun()

# Use example prompt if selected, otherwise use default
default_query = "A tech company hid research showing their algorithm harms teens. Find historical examples from OTHER industries where companies concealed internal research about product harm."

# If an example prompt was selected, update the query text
if 'selected_example_prompt' in st.session_state:
    st.session_state.query_textarea = st.session_state.selected_example_prompt
    del st.session_state.selected_example_prompt
    st.rerun()

# Initialize with default if not present
if 'query_textarea' not in st.session_state:
    st.session_state.query_textarea = default_query

query = st.text_area(
    "Research Query",
    value=st.session_state.query_textarea,
    height=120,
    help="This query will be sent to both approaches",
    key="query_textarea"
)

# Run comparison button
if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a research query")
    else:
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

                initial_prompt = f"""Research question: "{query}"

Use the exa_search tool to find relevant information. You can call it multiple times with different queries if needed.

FORMAT YOUR ANALYSIS LIKE THIS:

## 1. KEY FINDINGS

For each major finding, use this EXACT structure:

**Claim:** [State the factual claim in one sentence]
**Quote:** "[Exact quote from source that supports this claim, max 30 words]"
**Source:** [URL]

Example:
**Claim:** The tobacco industry concealed lung cancer research for decades.
**Quote:** "industry documents show executives knew of cancer link in 1950s but actively suppressed findings"
**Source:** https://pubmed.ncbi.nlm.nih.gov/tobacco-research

**Claim:** Internal memos revealed coordinated disinformation campaigns.
**Quote:** "we must be careful not to let the public know what we discovered about health risks"
**Source:** https://industry-archive.org/memos-1952

[If making an inference]: **Note:** This pattern appears common across industries. [INFERENCE - not directly stated in sources]

## 2. SOURCE QUALITY ASSESSMENT
[Brief assessment of source reliability]

## 3. GAPS OR CONTRADICTIONS
[What's missing or conflicting]

## 4. RECOMMENDED NEXT STEPS
[Suggested follow-up research]

CRITICAL RULES:
- Every factual claim MUST have: **Claim:** + **Quote:** + **Source:**
- Quotes must be EXACT words from the source
- Always include the full URL
- If you make an inference not directly from sources, mark it [INFERENCE]
- Use the claim/quote/source structure for ALL findings"""

                messages = [{"role": "user", "content": initial_prompt}]
                all_sources = []
                tool_use_count = 0

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

                # Show sources
                if all_sources:
                    with st.expander(f"📦 Sources Retrieved ({len(all_sources)})", expanded=True):
                        for i, source in enumerate(all_sources, 1):
                            st.markdown(f"**{i}. {source['title']}**")
                            st.caption(f"🔗 {source['url']}")
                            st.caption(f"📅 Published: {source.get('published', 'N/A')}")
                            if source['excerpt']:
                                st.text(source['excerpt'][:150] + "...")
                            st.markdown("---")

                    # Extract response content first
                    response_content = ""
                    for block in response.content:
                        if block.type == "text":
                            response_content += block.text

                    # Calculate and display retrieval metrics
                    retrieval_metrics = calculate_retrieval_metrics(
                        all_sources,
                        response.usage.input_tokens,
                        response.usage.output_tokens
                    )

                    # Add traceability to metrics
                    traceability = calculate_evidence_traceability(response_content, all_sources)
                    retrieval_metrics['traceability_rate'] = traceability['traceability_rate']
                    retrieval_metrics['supported_claims'] = traceability['supported_claims']
                    retrieval_metrics['inferences'] = traceability['inferences']
                    retrieval_metrics['unsupported_claims'] = traceability['unsupported_claims']
                    retrieval_metrics['tier_a_claims'] = traceability['tier_a_claims']
                    retrieval_metrics['tier_b_claims'] = traceability['tier_b_claims']
                    retrieval_metrics['tier_c_claims'] = traceability['tier_c_claims']

                    # Add timing to metrics
                    retrieval_metrics['time_seconds'] = total_time

                    display_retrieval_scorecard(retrieval_metrics, "Exa", tool_use_count, response_content, all_sources)

                    # Store Exa metrics in session state for logging
                    st.session_state.exa_metrics = retrieval_metrics
                    st.session_state.exa_search_count = tool_use_count
                    st.session_state.exa_analysis_text = response_content
                    st.session_state.exa_sources = all_sources
                    st.session_state.exa_search_queries = search_queries

                # Update status to complete
                status_container.update(
                    label=f"✅ Exa + Claude Search Complete ({tool_use_count} searches, {total_time:.1f}s)",
                    state="complete",
                    expanded=False
                )

                # Display final response
                st.markdown("#### 📊 Analysis Result")
                st.markdown(response_content)

                # Display traceability prominently after analysis
                st.markdown("---")
                traceability_display = calculate_evidence_traceability(response_content)
                st.markdown("#### 🔍 Evidence Traceability")

                col_supp, col_inf, col_unsupp = st.columns(3)
                with col_supp:
                    st.success(f"**✅ Supported:** {traceability_display['supported_claims']}")
                    st.caption("Has quote + source")
                with col_inf:
                    st.info(f"**🔶 Inferences:** {traceability_display['inferences']}")
                    st.caption("Analyst interpretation")
                with col_unsupp:
                    st.error(f"**⚠️ Unsupported:** {traceability_display['unsupported_claims']}")
                    st.caption("Claims without evidence")

                # Traceability interpretation
                if traceability_display['total_claims'] > 0:
                    trace_rate = traceability_display['traceability_rate']
                    st.info(f"📊 **Traceability Rate: {trace_rate}%** ({traceability_display['supported_claims']}/{traceability_display['total_claims']} claims)")

                    # Warning if unsupported claims exist
                    if traceability_display['unsupported_claims'] > 0:
                        st.warning(f"⚠️ **{traceability_display['unsupported_claims']} unsupported claim(s)** need verification or evidence")

                # Metrics
                st.caption(f"🔢 Tokens: {response.usage.input_tokens + response.usage.output_tokens:,}")

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

                    prompt = f"""Research question: "{query}"

Use web search to find relevant information. Limit yourself to searching for approximately {num_results} sources to match the comparison fairness.

FORMAT YOUR ANALYSIS LIKE THIS:

## 1. KEY FINDINGS

For each major finding, use this EXACT structure:

**Claim:** [State the factual claim in one sentence]
**Quote:** "[Exact quote from source that supports this claim, max 30 words]"
**Source:** [URL]

Example:
**Claim:** The tobacco industry concealed lung cancer research for decades.
**Quote:** "industry documents show executives knew of cancer link in 1950s but actively suppressed findings"
**Source:** https://pubmed.ncbi.nlm.nih.gov/tobacco-research

**Claim:** Internal memos revealed coordinated disinformation campaigns.
**Quote:** "we must be careful not to let the public know what we discovered about health risks"
**Source:** https://industry-archive.org/memos-1952

[If making an inference]: **Note:** This pattern appears common across industries. [INFERENCE - not directly stated in sources]

## 2. SOURCE QUALITY ASSESSMENT
[Brief assessment of source reliability]

## 3. GAPS OR CONTRADICTIONS
[What's missing or conflicting]

## 4. RECOMMENDED NEXT STEPS
[Suggested follow-up research]

CRITICAL RULES:
- Every factual claim MUST have: **Claim:** + **Quote:** + **Source:**
- Quotes must be EXACT words from the source
- Always include the full URL
- If you make an inference not directly from sources, mark it [INFERENCE]
- Use the claim/quote/source structure for ALL findings"""

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

                    # Show sources if available
                    if sources:
                        with st.expander(f"📦 Sources Found ({len(sources)})", expanded=True):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                st.caption(f"🔗 {source['url']}")
                                st.caption(f"📅 Age: {source['page_age']}")
                                st.markdown("---")

                        # Calculate and display retrieval metrics
                        retrieval_metrics = calculate_retrieval_metrics(
                            sources,
                            response.usage.input_tokens,
                            response.usage.output_tokens
                        )

                        # Add traceability to metrics
                        traceability = calculate_evidence_traceability(response_content, sources)
                        retrieval_metrics['traceability_rate'] = traceability['traceability_rate']
                        retrieval_metrics['supported_claims'] = traceability['supported_claims']
                        retrieval_metrics['inferences'] = traceability['inferences']
                        retrieval_metrics['unsupported_claims'] = traceability['unsupported_claims']
                        retrieval_metrics['tier_a_claims'] = traceability['tier_a_claims']
                        retrieval_metrics['tier_b_claims'] = traceability['tier_b_claims']
                        retrieval_metrics['tier_c_claims'] = traceability['tier_c_claims']

                        # Add timing to metrics
                        retrieval_metrics['time_seconds'] = total_time

                        display_retrieval_scorecard(retrieval_metrics, "Claude Web Search", 0, response_content, sources)

                        # Store Claude metrics in session state for logging
                        st.session_state.claude_metrics = retrieval_metrics
                        st.session_state.claude_analysis_text = response_content
                        st.session_state.claude_sources = sources

                    # Display result
                    st.markdown("#### 📊 Analysis Result")
                    st.markdown(response_content)

                    # Display traceability prominently after analysis
                    st.markdown("---")
                    traceability_display = calculate_evidence_traceability(response_content)
                    st.markdown("#### 🔍 Evidence Traceability")

                    col_supp, col_inf, col_unsupp = st.columns(3)
                    with col_supp:
                        st.success(f"**✅ Supported:** {traceability_display['supported_claims']}")
                        st.caption("Has quote + source")
                    with col_inf:
                        st.info(f"**🔶 Inferences:** {traceability_display['inferences']}")
                        st.caption("Analyst interpretation")
                    with col_unsupp:
                        st.error(f"**⚠️ Unsupported:** {traceability_display['unsupported_claims']}")
                        st.caption("Claims without evidence")

                    # Warning if unsupported claims exist
                    if traceability_display['unsupported_claims'] > 0:
                        st.warning(f"⚠️ **{traceability_display['unsupported_claims']} unsupported claim(s)** need verification or evidence")

                    # Traceability interpretation
                    if traceability_display['total_claims'] > 0:
                        trace_rate = traceability_display['traceability_rate']
                        st.info(f"📊 **Traceability Rate: {trace_rate}%**")

                    # Metrics
                    st.caption(f"🔢 Tokens: {response.usage.input_tokens + response.usage.output_tokens:,}")

                except Exception as e:
                    st.error(f"❌ Claude web search failed: {str(e)}")

        # --- LOG COMPARISON METRICS (After both columns complete) ---
        # Log metrics if we have data from both approaches
        if 'exa_metrics' in st.session_state and 'claude_metrics' in st.session_state:
            log_comparison(
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
                exa_search_queries=st.session_state.get('exa_search_queries')
            )
            st.success("✅ Comparison metrics logged to CSV")
            st.info("💡 Scroll to bottom and click '📊 Load Comparison History' to view all logged comparisons")
            # Set flag to show updated stats
            st.session_state.comparison_logged = True

        elif 'exa_metrics' in st.session_state:
            # Log with empty Claude metrics if Claude failed
            empty_metrics = {'quality_mix': {'A': 0, 'B': 0, 'C': 0}, 'unique_domains': 0, 'total_sources': 0, 'tokens': 0, 'cost_usd': 0.0}
            log_comparison(
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
                exa_search_queries=st.session_state.get('exa_search_queries')
            )
            st.warning("⚠️ Logged Exa metrics only (Claude search failed)")
            # Set flag to show updated stats
            st.session_state.comparison_logged = True

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
                # Check if we have sources from Exa session state
                if 'exa_sources' not in st.session_state or not st.session_state.exa_sources:
                    st.warning("⚠️ Run the comparison first to generate Exa sources")
                else:
                    article_prompt = f"""You are a professional journalist writing an article based on research.

Research Question: "{query}"

Sources Retrieved:
{json.dumps(st.session_state.exa_sources, indent=2)}

Write a comprehensive news article (500-800 words) that:
1. Has a compelling headline
2. Follows inverted pyramid structure
3. Cites sources with inline links [source title](url)
4. Maintains journalistic objectivity
5. Includes relevant context and background
6. Ends with implications or what's next

Format as markdown with:
- # Headline
- **Byline**: Research compiled with Exa + Claude
- Article body with proper paragraphs
- Inline source citations"""

                    exa_article_response = anthropic.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=2500,
                        messages=[{"role": "user", "content": article_prompt}]
                    )

                    exa_article = exa_article_response.content[0].text

                    st.markdown(exa_article)

                    # Metrics
                    st.caption(f"📊 Sources used: {len(st.session_state.exa_sources)}")
                    st.caption(f"🔢 Tokens: {exa_article_response.usage.input_tokens + exa_article_response.usage.output_tokens:,}")

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
                article_prompt = f"""You are a professional journalist writing an article.

Research Question: "{query}"

Use web search to gather comprehensive information, then write a news article (500-800 words) that:
1. Has a compelling headline
2. Follows inverted pyramid structure
3. Cites sources with inline links [source title](url)
4. Maintains journalistic objectivity
5. Includes relevant context and background
6. Ends with implications or what's next

Format as markdown with:
- # Headline
- **Byline**: Research compiled with Claude Web Search
- Article body with proper paragraphs
- Inline source citations

Search thoroughly to match the depth of reporting."""

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

# --- METRICS HISTORY SECTION ---
st.markdown("---")
st.markdown("## 📈 Comparison History")

# Button to load history on demand (doesn't cause page refresh)
if st.button("📊 Load Comparison History", type="secondary"):
    st.session_state.show_history = True

# Show history if button was clicked or if just logged a comparison
if st.session_state.get('show_history', False) or st.session_state.get('comparison_logged', False):
    # Fetch fresh stats and detailed history
    stats = get_total_stats()
    history = get_detailed_history()

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
            st.metric("Traceability", f"{stats['avg_exa_traceability']:.1f}%")
            st.metric("Inference Rate", f"{stats['avg_exa_inference_rate']:.1f}%")
            st.metric("Unsupported Rate", f"{stats['avg_exa_unsupported_rate']:.1f}%")

        with col2:
            st.markdown("#### Claude Web Search")
            st.metric("Total Cost", f"${stats['total_claude_cost']:.4f}")
            st.metric("Avg Cost per Run", f"${stats['avg_claude_cost']:.4f}")
            st.metric("Traceability", f"{stats['avg_claude_traceability']:.1f}%")
            st.metric("Inference Rate", f"{stats['avg_claude_inference_rate']:.1f}%")
            st.metric("Unsupported Rate", f"{stats['avg_claude_unsupported_rate']:.1f}%")

        st.caption(f"📁 Metrics saved to: `{stats['csv_path']}`")

        # Show when last updated
        if st.session_state.get('comparison_logged', False):
            st.success("✅ Updated with latest comparison")

        st.markdown("---")

        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison History")
        st.caption("🟢 Green = better | 🔴 Red = worse | ↑ Higher is better: Tier A%, Domains%, Traceability%, Tier A/B Claims% | ↓ Lower is better: Tier C Claims%, Inference%, Unsupported%, Tokens, Cost, Time")

        # Load saved result viewer
        st.markdown("---")
        st.markdown("#### 🔍 View Saved Result")
        st.caption("Load the full analysis text and sources from any previous comparison")

        # Create list of result options with timestamp and query
        result_options = {}
        for row in history:
            result_id = row.get('result_id', '')
            if result_id:
                timestamp = row['timestamp'][:16]  # YYYY-MM-DD HH:MM
                query = row['query'][:50] + "..." if len(row['query']) > 50 else row['query']
                label = f"{timestamp} - {query}"
                result_options[label] = result_id

        if result_options:
            selected_label = st.selectbox("Select a comparison to view:", [""] + list(result_options.keys()))

            if selected_label:
                result_id = result_options[selected_label]
                result_data = load_result(result_id)

                if result_data:
                    st.success(f"✅ Loaded result: {result_id}")

                    # Display in two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🔍 Exa + Claude Results")
                        st.markdown(f"**Query:** {result_data['query']}")
                        st.markdown(f"**Searches:** {result_data['exa']['search_count']}")
                        st.markdown(f"**Time:** {result_data['exa']['time_seconds']:.1f}s")

                        with st.expander("📝 Analysis", expanded=True):
                            st.markdown(result_data['exa']['analysis_text'])

                        with st.expander(f"📦 Sources ({len(result_data['exa']['sources'])})"):
                            for i, source in enumerate(result_data['exa']['sources'], 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                st.caption(f"🔗 {source['url']}")
                                if source.get('excerpt'):
                                    st.text(source['excerpt'][:150] + "...")
                                st.markdown("---")

                        with st.expander(f"🔍 Search Queries ({len(result_data['exa']['search_queries'])})"):
                            for query in result_data['exa']['search_queries']:
                                st.markdown(f"- {query}")

                    with col2:
                        st.markdown("### 🌐 Claude Web Search Results")
                        st.markdown(f"**Query:** {result_data['query']}")
                        st.markdown(f"**Time:** {result_data['claude']['time_seconds']:.1f}s")

                        with st.expander("📝 Analysis", expanded=True):
                            st.markdown(result_data['claude']['analysis_text'])

                        with st.expander(f"📦 Sources ({len(result_data['claude']['sources'])})"):
                            for i, source in enumerate(result_data['claude']['sources'], 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                st.caption(f"🔗 {source['url']}")
                                st.markdown("---")

                    # Display metrics comparison
                    st.markdown("---")
                    st.markdown("#### 📊 Metrics Comparison")
                    met_col1, met_col2, met_col3 = st.columns(3)

                    with met_col1:
                        st.metric("Exa Traceability", f"{result_data['metrics']['exa_traceability']:.1f}%")
                        st.metric("Exa Tier A %", f"{result_data['metrics']['exa_tier_a_pct']:.1f}%")
                        st.metric("Exa Cost", f"${result_data['metrics']['exa_cost']:.4f}")

                    with met_col2:
                        st.metric("Claude Traceability", f"{result_data['metrics']['claude_traceability']:.1f}%")
                        st.metric("Claude Tier A %", f"{result_data['metrics']['claude_tier_a_pct']:.1f}%")
                        st.metric("Claude Cost", f"${result_data['metrics']['claude_cost']:.4f}")

                    with met_col3:
                        trace_diff = result_data['metrics']['exa_traceability'] - result_data['metrics']['claude_traceability']
                        tier_a_diff = result_data['metrics']['exa_tier_a_pct'] - result_data['metrics']['claude_tier_a_pct']
                        cost_diff = result_data['metrics']['exa_cost'] - result_data['metrics']['claude_cost']

                        st.metric("Trace Δ", f"{trace_diff:+.1f}%")
                        st.metric("Tier A Δ", f"{tier_a_diff:+.1f}%")
                        st.metric("Cost Δ", f"${cost_diff:+.4f}")

                    # Add claims analysis button
                    st.markdown("---")
                    if st.button("🔬 Analyze Claims Quality", key=f"analyze_{result_id}", type="primary"):
                        with st.spinner("🔍 Extracting and analyzing claims..."):
                            from claims_analyzer import generate_claims_report, extract_structured_claims, analyze_claim_quality

                            # Prepare result data
                            exa_data = {
                                'analysis_text': result_data['exa']['analysis_text'],
                                'sources': result_data['exa']['sources']
                            }
                            claude_data = {
                                'analysis_text': result_data['claude']['analysis_text'],
                                'sources': result_data['claude']['sources']
                            }

                            # Generate report
                            report = generate_claims_report(exa_data, claude_data, result_data['query'])

                            st.success("✅ Claims analysis complete")

                            # Display claims breakdown
                            st.markdown("## 📋 Claims Breakdown")

                            col_exa, col_claude = st.columns(2)

                            with col_exa:
                                st.markdown("### 🔍 Exa + Claude Claims")
                                exa_q = report['exa_quality']
                                st.metric("Total Claims", exa_q['total_claims'])

                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric("✅ Supported", exa_q['supported_count'], f"{exa_q['supported_pct']:.0f}%")
                                with subcol2:
                                    st.metric("🔶 Inferences", exa_q['inference_count'], f"{exa_q['inference_pct']:.0f}%")
                                with subcol3:
                                    st.metric("⚠️ Unsupported", exa_q['unsupported_count'], f"{exa_q['unsupported_pct']:.0f}%")

                                # Tier breakdown for supported
                                if exa_q['supported_count'] > 0:
                                    st.markdown("**Supported Claims - Source Quality:**")
                                    tier_col1, tier_col2, tier_col3 = st.columns(3)
                                    with tier_col1:
                                        st.metric("🥇 Tier A", exa_q['tier_breakdown']['A'], f"{exa_q['tier_breakdown']['A_pct']:.0f}%")
                                    with tier_col2:
                                        st.metric("🥈 Tier B", exa_q['tier_breakdown']['B'], f"{exa_q['tier_breakdown']['B_pct']:.0f}%")
                                    with tier_col3:
                                        st.metric("🥉 Tier C", exa_q['tier_breakdown']['C'], f"{exa_q['tier_breakdown']['C_pct']:.0f}%")

                                # Issues
                                if exa_q['issues']:
                                    st.markdown("**⚠️ Quality Issues:**")
                                    for issue in exa_q['issues']:
                                        if issue['severity'] == 'high':
                                            st.error(f"🔴 {issue['message']}")
                                        else:
                                            st.warning(f"🟡 {issue['message']}")

                            with col_claude:
                                st.markdown("### 🌐 Claude Web Search Claims")
                                claude_q = report['claude_quality']
                                st.metric("Total Claims", claude_q['total_claims'])

                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric("✅ Supported", claude_q['supported_count'], f"{claude_q['supported_pct']:.0f}%")
                                with subcol2:
                                    st.metric("🔶 Inferences", claude_q['inference_count'], f"{claude_q['inference_pct']:.0f}%")
                                with subcol3:
                                    st.metric("⚠️ Unsupported", claude_q['unsupported_count'], f"{claude_q['unsupported_pct']:.0f}%")

                                # Tier breakdown for supported
                                if claude_q['supported_count'] > 0:
                                    st.markdown("**Supported Claims - Source Quality:**")
                                    tier_col1, tier_col2, tier_col3 = st.columns(3)
                                    with tier_col1:
                                        st.metric("🥇 Tier A", claude_q['tier_breakdown']['A'], f"{claude_q['tier_breakdown']['A_pct']:.0f}%")
                                    with tier_col2:
                                        st.metric("🥈 Tier B", claude_q['tier_breakdown']['B'], f"{claude_q['tier_breakdown']['B_pct']:.0f}%")
                                    with tier_col3:
                                        st.metric("🥉 Tier C", claude_q['tier_breakdown']['C'], f"{claude_q['tier_breakdown']['C_pct']:.0f}%")

                                # Issues
                                if claude_q['issues']:
                                    st.markdown("**⚠️ Quality Issues:**")
                                    for issue in claude_q['issues']:
                                        if issue['severity'] == 'high':
                                            st.error(f"🔴 {issue['message']}")
                                        else:
                                            st.warning(f"🟡 {issue['message']}")

                            # Relevance Analysis
                            st.markdown("---")
                            st.markdown("## 🎯 Relevance to Query")
                            st.caption(f"How well does each approach answer the actual question?")

                            if report['exa_relevance'] and report['claude_relevance']:
                                rel_col1, rel_col2 = st.columns(2)

                                with rel_col1:
                                    st.markdown("### 🔍 Exa + Claude Relevance")
                                    exa_rel = report['exa_relevance']
                                    st.metric("Average Relevance", f"{exa_rel['avg_relevance']}/10")

                                    rel_subcol1, rel_subcol2, rel_subcol3 = st.columns(3)
                                    with rel_subcol1:
                                        st.metric("🎯 Direct Answers", exa_rel['directly_answers_query'], help="Claims scoring 9-10")
                                    with rel_subcol2:
                                        st.metric("📊 Tangential", exa_rel['tangentially_relevant'], help="Claims scoring 4-6")
                                    with rel_subcol3:
                                        st.metric("❌ Off-topic", exa_rel['off_topic'], help="Claims scoring 0-3")

                                    # Issues
                                    if exa_rel['off_topic'] > 0:
                                        st.warning(f"⚠️ {exa_rel['off_topic']} claim(s) are off-topic or barely relevant")
                                    if exa_rel['directly_answers_query'] == 0:
                                        st.error("🔴 No claims directly answer the query")

                                with rel_col2:
                                    st.markdown("### 🌐 Claude Web Search Relevance")
                                    claude_rel = report['claude_relevance']
                                    st.metric("Average Relevance", f"{claude_rel['avg_relevance']}/10")

                                    rel_subcol1, rel_subcol2, rel_subcol3 = st.columns(3)
                                    with rel_subcol1:
                                        st.metric("🎯 Direct Answers", claude_rel['directly_answers_query'], help="Claims scoring 9-10")
                                    with rel_subcol2:
                                        st.metric("📊 Tangential", claude_rel['tangentially_relevant'], help="Claims scoring 4-6")
                                    with rel_subcol3:
                                        st.metric("❌ Off-topic", claude_rel['off_topic'], help="Claims scoring 0-3")

                                    # Issues
                                    if claude_rel['off_topic'] > 0:
                                        st.warning(f"⚠️ {claude_rel['off_topic']} claim(s) are off-topic or barely relevant")
                                    if claude_rel['directly_answers_query'] == 0:
                                        st.error("🔴 No claims directly answer the query")

                                # Comparison
                                st.markdown("---")
                                rel_diff = exa_rel['avg_relevance'] - claude_rel['avg_relevance']
                                direct_diff = exa_rel['directly_answers_query'] - claude_rel['directly_answers_query']

                                if abs(rel_diff) < 0.5:
                                    st.info(f"📊 **Similar relevance:** Both approaches average ~{exa_rel['avg_relevance']:.1f}/10")
                                elif rel_diff > 0:
                                    st.success(f"✅ **Exa more relevant:** {rel_diff:+.1f} point advantage ({exa_rel['avg_relevance']:.1f} vs {claude_rel['avg_relevance']:.1f})")
                                else:
                                    st.success(f"✅ **Claude more relevant:** {abs(rel_diff):.1f} point advantage ({claude_rel['avg_relevance']:.1f} vs {exa_rel['avg_relevance']:.1f})")

                                if direct_diff > 0:
                                    st.info(f"🎯 Exa has {direct_diff} more direct answer(s) to the query")
                                elif direct_diff < 0:
                                    st.info(f"🎯 Claude has {abs(direct_diff)} more direct answer(s) to the query")

                                st.caption(f"Token usage: {exa_rel.get('tokens', 0) + claude_rel.get('tokens', 0):,}")

                            # Coverage comparison
                            st.markdown("---")
                            st.markdown("## 🔄 Coverage Comparison")
                            st.caption(f"Token usage: {report['coverage_comparison'].get('tokens', 0):,}")
                            st.markdown(report['coverage_comparison']['comparison_text'])

                            # Detailed claims lists
                            st.markdown("---")
                            st.markdown("## 📝 Detailed Claims Inspection")

                            tab1, tab2 = st.tabs(["🔍 Exa + Claude Claims", "🌐 Claude Web Search Claims"])

                            with tab1:
                                exa_claims = report['exa_claims']
                                exa_rel = report.get('exa_relevance')

                                # Create lookup for relevance scores
                                relevance_lookup = {}
                                if exa_rel and exa_rel['relevance_scores']:
                                    for score_data in exa_rel['relevance_scores']:
                                        relevance_lookup[score_data['claim']] = score_data

                                if exa_claims['supported_claims']:
                                    st.markdown(f"### ✅ Supported Claims ({len(exa_claims['supported_claims'])})")
                                    for i, claim in enumerate(exa_claims['supported_claims'], 1):
                                        # Get relevance score
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        rel_emoji = ""
                                        if rel_data:
                                            score = rel_data['score']
                                            if score >= 9:
                                                rel_emoji = "🎯"
                                            elif score >= 7:
                                                rel_emoji = "✅"
                                            elif score >= 4:
                                                rel_emoji = "📊"
                                            else:
                                                rel_emoji = "⚠️"

                                        title = f"Claim {i} - {claim['tier']} Source"
                                        if rel_data:
                                            title += f" - {rel_emoji} {rel_data['score']}/10"

                                        with st.expander(title):
                                            st.markdown(f"**Claim:** {claim['claim']}")
                                            st.markdown(f"**Quote:** \"{claim['quote']}\"")
                                            st.markdown(f"**Source:** [{claim['source_title']}]({claim['url']})")
                                            st.caption(f"Tier: {claim['tier']}")
                                            if rel_data:
                                                st.caption(f"**Relevance:** {rel_data['score']}/10 - {rel_data['reasoning']}")

                                if exa_claims['inference_claims']:
                                    st.markdown(f"### 🔶 Inference Claims ({len(exa_claims['inference_claims'])})")
                                    for i, claim in enumerate(exa_claims['inference_claims'], 1):
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        display_text = f"{i}. {claim['claim']}"
                                        if rel_data:
                                            display_text += f" ({rel_data['score']}/10)"
                                        st.info(display_text)
                                        if rel_data:
                                            st.caption(f"↳ {rel_data['reasoning']}")

                                if exa_claims['unsupported_claims']:
                                    st.markdown(f"### ⚠️ Unsupported Claims ({len(exa_claims['unsupported_claims'])})")
                                    for i, claim in enumerate(exa_claims['unsupported_claims'], 1):
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        display_text = f"{i}. {claim['claim']}"
                                        if rel_data:
                                            display_text += f" ({rel_data['score']}/10)"
                                        st.error(display_text)
                                        if rel_data:
                                            st.caption(f"↳ {rel_data['reasoning']}")

                            with tab2:
                                claude_claims = report['claude_claims']
                                claude_rel = report.get('claude_relevance')

                                # Create lookup for relevance scores
                                relevance_lookup = {}
                                if claude_rel and claude_rel['relevance_scores']:
                                    for score_data in claude_rel['relevance_scores']:
                                        relevance_lookup[score_data['claim']] = score_data

                                if claude_claims['supported_claims']:
                                    st.markdown(f"### ✅ Supported Claims ({len(claude_claims['supported_claims'])})")
                                    for i, claim in enumerate(claude_claims['supported_claims'], 1):
                                        # Get relevance score
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        rel_emoji = ""
                                        if rel_data:
                                            score = rel_data['score']
                                            if score >= 9:
                                                rel_emoji = "🎯"
                                            elif score >= 7:
                                                rel_emoji = "✅"
                                            elif score >= 4:
                                                rel_emoji = "📊"
                                            else:
                                                rel_emoji = "⚠️"

                                        title = f"Claim {i} - {claim['tier']} Source"
                                        if rel_data:
                                            title += f" - {rel_emoji} {rel_data['score']}/10"

                                        with st.expander(title):
                                            st.markdown(f"**Claim:** {claim['claim']}")
                                            st.markdown(f"**Quote:** \"{claim['quote']}\"")
                                            st.markdown(f"**Source:** [{claim['source_title']}]({claim['url']})")
                                            st.caption(f"Tier: {claim['tier']}")
                                            if rel_data:
                                                st.caption(f"**Relevance:** {rel_data['score']}/10 - {rel_data['reasoning']}")

                                if claude_claims['inference_claims']:
                                    st.markdown(f"### 🔶 Inference Claims ({len(claude_claims['inference_claims'])})")
                                    for i, claim in enumerate(claude_claims['inference_claims'], 1):
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        display_text = f"{i}. {claim['claim']}"
                                        if rel_data:
                                            display_text += f" ({rel_data['score']}/10)"
                                        st.info(display_text)
                                        if rel_data:
                                            st.caption(f"↳ {rel_data['reasoning']}")

                                if claude_claims['unsupported_claims']:
                                    st.markdown(f"### ⚠️ Unsupported Claims ({len(claude_claims['unsupported_claims'])})")
                                    for i, claim in enumerate(claude_claims['unsupported_claims'], 1):
                                        rel_data = relevance_lookup.get(claim['claim'])
                                        display_text = f"{i}. {claim['claim']}"
                                        if rel_data:
                                            display_text += f" ({rel_data['score']}/10)"
                                        st.error(display_text)
                                        if rel_data:
                                            st.caption(f"↳ {rel_data['reasoning']}")

                else:
                    st.error(f"❌ Could not load result: {result_id}")

        st.markdown("---")

        # Prepare data for display
        import pandas as pd

        # Helper function to safely parse float from potentially corrupted CSV data
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling corrupted data"""
            if value is None or value == '':
                return default
            try:
                # If it's already a number, return it
                if isinstance(value, (int, float)):
                    return float(value)
                # Try to convert string to float
                # If corrupted (contains non-numeric chars), extract first number
                str_val = str(value)
                # Try direct conversion first
                return float(str_val)
            except (ValueError, TypeError):
                # Data is corrupted, try to extract first valid float
                try:
                    import re
                    match = re.match(r'([0-9]+\.?[0-9]*)', str_val)
                    if match:
                        return float(match.group(1))
                except:
                    pass
                return default

        # Create display dataframe with ALL core metrics
        display_data = []
        for row in history:
            # Calculate percentages
            exa_sources = int(safe_float(row.get('exa_sources', 0)))
            claude_sources = int(safe_float(row.get('claude_sources', 0)))

            exa_tier_a_pct = (int(safe_float(row.get('exa_tier_a', 0))) / exa_sources * 100) if exa_sources > 0 else 0
            claude_tier_a_pct = (int(safe_float(row.get('claude_tier_a', 0))) / claude_sources * 100) if claude_sources > 0 else 0

            exa_domain_pct = (int(safe_float(row.get('exa_domains', 0))) / exa_sources * 100) if exa_sources > 0 else 0
            claude_domain_pct = (int(safe_float(row.get('claude_domains', 0))) / claude_sources * 100) if claude_sources > 0 else 0

            exa_trace = safe_float(row.get('exa_traceability', 0))
            claude_trace = safe_float(row.get('claude_traceability', 0))

            # Calculate inference and unsupported rates
            exa_supported = int(safe_float(row.get('exa_supported', 0)))
            exa_inferences = int(safe_float(row.get('exa_inferences', 0)))
            exa_unsupported = int(safe_float(row.get('exa_unsupported', 0)))
            exa_total = exa_supported + exa_inferences + exa_unsupported

            exa_inference_pct = (exa_inferences / exa_total * 100) if exa_total > 0 else 0
            exa_unsupported_pct = (exa_unsupported / exa_total * 100) if exa_total > 0 else 0

            claude_supported = int(safe_float(row.get('claude_supported', 0)))
            claude_inferences = int(safe_float(row.get('claude_inferences', 0)))
            claude_unsupported = int(safe_float(row.get('claude_unsupported', 0)))
            claude_total = claude_supported + claude_inferences + claude_unsupported

            claude_inference_pct = (claude_inferences / claude_total * 100) if claude_total > 0 else 0
            claude_unsupported_pct = (claude_unsupported / claude_total * 100) if claude_total > 0 else 0

            # Calculate tier breakdown for supported claims
            exa_tier_a_claims = int(safe_float(row.get('exa_tier_a_claims', 0)))
            exa_tier_b_claims = int(safe_float(row.get('exa_tier_b_claims', 0)))
            exa_tier_c_claims = int(safe_float(row.get('exa_tier_c_claims', 0)))

            claude_tier_a_claims = int(safe_float(row.get('claude_tier_a_claims', 0)))
            claude_tier_b_claims = int(safe_float(row.get('claude_tier_b_claims', 0)))
            claude_tier_c_claims = int(safe_float(row.get('claude_tier_c_claims', 0)))

            # Calculate percentages of supported claims backed by each tier
            exa_tier_a_claim_pct = (exa_tier_a_claims / exa_supported * 100) if exa_supported > 0 else 0
            exa_tier_b_claim_pct = (exa_tier_b_claims / exa_supported * 100) if exa_supported > 0 else 0
            exa_tier_c_claim_pct = (exa_tier_c_claims / exa_supported * 100) if exa_supported > 0 else 0

            claude_tier_a_claim_pct = (claude_tier_a_claims / claude_supported * 100) if claude_supported > 0 else 0
            claude_tier_b_claim_pct = (claude_tier_b_claims / claude_supported * 100) if claude_supported > 0 else 0
            claude_tier_c_claim_pct = (claude_tier_c_claims / claude_supported * 100) if claude_supported > 0 else 0

            display_data.append({
                'Timestamp': row['timestamp'][:10],  # Just date
                'Category': row.get('query_category', '') or 'Uncategorized',
                'Query': row['query'],  # Full query, no truncation
                'Exa Search Type': row.get('search_type', '') or 'auto',
                'Exa+Claude Tier A %': exa_tier_a_pct,
                'Claude WS Tier A %': claude_tier_a_pct,
                'Exa+Claude Domains %': exa_domain_pct,
                'Claude WS Domains %': claude_domain_pct,
                'Exa+Claude Trace %': exa_trace,
                'Claude WS Trace %': claude_trace,
                'Exa+Claude Tier A Claims %': exa_tier_a_claim_pct,
                'Claude WS Tier A Claims %': claude_tier_a_claim_pct,
                'Exa+Claude Tier B Claims %': exa_tier_b_claim_pct,
                'Claude WS Tier B Claims %': claude_tier_b_claim_pct,
                'Exa+Claude Tier C Claims %': exa_tier_c_claim_pct,
                'Claude WS Tier C Claims %': claude_tier_c_claim_pct,
                'Exa+Claude Inference %': exa_inference_pct,
                'Claude WS Inference %': claude_inference_pct,
                'Exa+Claude Unsupported %': exa_unsupported_pct,
                'Claude WS Unsupported %': claude_unsupported_pct,
                'Exa+Claude Tokens': int(safe_float(row.get('exa_tokens', 0))),
                'Claude WS Tokens': int(safe_float(row.get('claude_tokens', 0))),
                'Exa+Claude Cost': safe_float(row.get('exa_cost', 0)),
                'Claude WS Cost': safe_float(row.get('claude_cost', 0)),
                'Exa+Claude Time (s)': safe_float(row.get('exa_time_seconds', 0)),
                'Claude WS Time (s)': safe_float(row.get('claude_time_seconds', 0)),
            })

        df = pd.DataFrame(display_data)

        # Add a "Run #" column to show repeated queries
        query_counts = {}
        run_numbers = []
        for query in df['Query']:
            if query not in query_counts:
                query_counts[query] = 0
            query_counts[query] += 1
            run_numbers.append(query_counts[query])

        df.insert(3, 'Run #', run_numbers)

        # Sort by Query first, then by Timestamp to group repeated queries together
        df = df.sort_values(by=['Query', 'Timestamp'], ascending=[True, False])
        df = df.reset_index(drop=True)

        # Add visual separator indicator for query groups
        group_indicators = []
        prev_query = None
        for query in df['Query']:
            if query != prev_query and prev_query is not None:
                group_indicators.append('🔹 New Query')
            else:
                group_indicators.append('')
            prev_query = query

        df.insert(0, '', group_indicators)

        # Function to color-code cells (green = better, red = worse)
        def highlight_comparison(row):
            styles = [''] * len(row)
            # Column order: (0), Timestamp(1), Category(2), Run #(3), Query(4), Exa Search Type(5),
            #               Exa+Claude Tier A %(6), Claude WS Tier A %(7),
            #               Exa+Claude Domains %(8), Claude WS Domains %(9),
            #               Exa+Claude Trace %(10), Claude WS Trace %(11),
            #               Exa+Claude Tier A Claims %(12), Claude WS Tier A Claims %(13),
            #               Exa+Claude Tier B Claims %(14), Claude WS Tier B Claims %(15),
            #               Exa+Claude Tier C Claims %(16), Claude WS Tier C Claims %(17),
            #               Exa+Claude Inference %(18), Claude WS Inference %(19),
            #               Exa+Claude Unsupported %(20), Claude WS Unsupported %(21),
            #               Exa+Claude Tokens(22), Claude WS Tokens(23),
            #               Exa+Claude Cost(24), Claude WS Cost(25),
            #               Exa+Claude Time(26), Claude WS Time(27)

            # Tier A % (higher is better)
            if row['Exa+Claude Tier A %'] > row['Claude WS Tier A %']:
                styles[6] = 'background-color: #d4edda'  # green
                styles[7] = 'background-color: #f8d7da'  # red
            elif row['Exa+Claude Tier A %'] < row['Claude WS Tier A %']:
                styles[6] = 'background-color: #f8d7da'
                styles[7] = 'background-color: #d4edda'

            # Domains % (higher is better)
            if row['Exa+Claude Domains %'] > row['Claude WS Domains %']:
                styles[8] = 'background-color: #d4edda'
                styles[9] = 'background-color: #f8d7da'
            elif row['Exa+Claude Domains %'] < row['Claude WS Domains %']:
                styles[8] = 'background-color: #f8d7da'
                styles[9] = 'background-color: #d4edda'

            # Traceability % (higher is better)
            if row['Exa+Claude Trace %'] > row['Claude WS Trace %']:
                styles[10] = 'background-color: #d4edda'
                styles[11] = 'background-color: #f8d7da'
            elif row['Exa+Claude Trace %'] < row['Claude WS Trace %']:
                styles[10] = 'background-color: #f8d7da'
                styles[11] = 'background-color: #d4edda'

            # Tier A Claims % (higher is better - more claims backed by Tier A sources)
            if row['Exa+Claude Tier A Claims %'] > row['Claude WS Tier A Claims %']:
                styles[12] = 'background-color: #d4edda'
                styles[13] = 'background-color: #f8d7da'
            elif row['Exa+Claude Tier A Claims %'] < row['Claude WS Tier A Claims %']:
                styles[12] = 'background-color: #f8d7da'
                styles[13] = 'background-color: #d4edda'

            # Tier B Claims % (higher is better)
            if row['Exa+Claude Tier B Claims %'] > row['Claude WS Tier B Claims %']:
                styles[14] = 'background-color: #d4edda'
                styles[15] = 'background-color: #f8d7da'
            elif row['Exa+Claude Tier B Claims %'] < row['Claude WS Tier B Claims %']:
                styles[14] = 'background-color: #f8d7da'
                styles[15] = 'background-color: #d4edda'

            # Tier C Claims % (lower is better - fewer claims backed by Tier C sources)
            if row['Exa+Claude Tier C Claims %'] < row['Claude WS Tier C Claims %']:
                styles[16] = 'background-color: #d4edda'
                styles[17] = 'background-color: #f8d7da'
            elif row['Exa+Claude Tier C Claims %'] > row['Claude WS Tier C Claims %']:
                styles[16] = 'background-color: #f8d7da'
                styles[17] = 'background-color: #d4edda'

            # Inference % (lower is better - less analyst interpretation)
            if row['Exa+Claude Inference %'] < row['Claude WS Inference %']:
                styles[18] = 'background-color: #d4edda'
                styles[19] = 'background-color: #f8d7da'
            elif row['Exa+Claude Inference %'] > row['Claude WS Inference %']:
                styles[18] = 'background-color: #f8d7da'
                styles[19] = 'background-color: #d4edda'

            # Unsupported % (lower is better - fewer claims without evidence)
            if row['Exa+Claude Unsupported %'] < row['Claude WS Unsupported %']:
                styles[20] = 'background-color: #d4edda'
                styles[21] = 'background-color: #f8d7da'
            elif row['Exa+Claude Unsupported %'] > row['Claude WS Unsupported %']:
                styles[20] = 'background-color: #f8d7da'
                styles[21] = 'background-color: #d4edda'

            # Tokens (lower is better - more efficient)
            if row['Exa+Claude Tokens'] < row['Claude WS Tokens']:
                styles[22] = 'background-color: #d4edda'
                styles[23] = 'background-color: #f8d7da'
            elif row['Exa+Claude Tokens'] > row['Claude WS Tokens']:
                styles[22] = 'background-color: #f8d7da'
                styles[23] = 'background-color: #d4edda'

            # Cost (lower is better - more cost-effective)
            if row['Exa+Claude Cost'] < row['Claude WS Cost']:
                styles[24] = 'background-color: #d4edda'
                styles[25] = 'background-color: #f8d7da'
            elif row['Exa+Claude Cost'] > row['Claude WS Cost']:
                styles[24] = 'background-color: #f8d7da'
                styles[25] = 'background-color: #d4edda'

            # Time (lower is better - faster)
            if row['Exa+Claude Time (s)'] < row['Claude WS Time (s)']:
                styles[26] = 'background-color: #d4edda'
                styles[27] = 'background-color: #f8d7da'
            elif row['Exa+Claude Time (s)'] > row['Claude WS Time (s)']:
                styles[26] = 'background-color: #f8d7da'
                styles[27] = 'background-color: #d4edda'

            return styles

        # Apply styling
        styled_df = df.style.apply(highlight_comparison, axis=1).format({
            'Exa+Claude Tier A %': '{:.0f}%',
            'Claude WS Tier A %': '{:.0f}%',
            'Exa+Claude Domains %': '{:.0f}%',
            'Claude WS Domains %': '{:.0f}%',
            'Exa+Claude Trace %': '{:.0f}%',
            'Claude WS Trace %': '{:.0f}%',
            'Exa+Claude Tier A Claims %': '{:.0f}%',
            'Claude WS Tier A Claims %': '{:.0f}%',
            'Exa+Claude Tier B Claims %': '{:.0f}%',
            'Claude WS Tier B Claims %': '{:.0f}%',
            'Exa+Claude Tier C Claims %': '{:.0f}%',
            'Claude WS Tier C Claims %': '{:.0f}%',
            'Exa+Claude Inference %': '{:.0f}%',
            'Claude WS Inference %': '{:.0f}%',
            'Exa+Claude Unsupported %': '{:.0f}%',
            'Claude WS Unsupported %': '{:.0f}%',
            'Exa+Claude Tokens': '{:,}',
            'Claude WS Tokens': '{:,}',
            'Exa+Claude Cost': '${:.4f}',
            'Claude WS Cost': '${:.4f}',
            'Exa+Claude Time (s)': '{:.1f}s',
            'Claude WS Time (s)': '{:.1f}s',
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.caption("🔹 Queries are **grouped together** for easy comparison of repeat runs")
        st.caption("🔢 **Run #** shows which iteration of the same query (useful for repeatability testing)")
        st.caption("📁 Full data available in CSV for pivot tables and advanced analysis")
        st.caption("💡 Use this table to identify which query types each system excels at")

    else:
        st.info("📊 No comparison history yet. Run a comparison above to start logging metrics!")
else:
    st.info("💡 Click 'Load Comparison History' to view your logged comparisons (won't refresh the page)")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Building Your Value Proposition

This tool helps you **generate data-driven insights** for positioning Exa's semantic search.

**How to use it:**
1. Run comparisons across different query types (Scientific, Legal, Follow-the-Money, etc.)
2. Build up your comparison history (10-20 queries minimum)
3. Analyze patterns: Where does Exa+Claude excel? Where does it struggle?
4. Use the CSV export for pivot tables and deeper analysis

**What to look for:**
- Which query categories show higher Tier A% for Exa?
- Which types have better traceability with Exa?
- Where does domain filtering matter vs recency?

Your **Comparison History** becomes your proof - show stakeholders real data, not assumptions.
""")

st.caption("Built with Streamlit • Powered by Exa & Claude")
