"""
Streamlit Web App - Exa Websets vs Claude Web Search Comparison
Showcases Exa's websets feature for vertical-specific search
"""

import streamlit as st
import os
from dotenv import load_dotenv
from exa_py import Exa
from anthropic import Anthropic
from datetime import datetime
import json
import re
from metrics_logger import log_comparison, get_total_stats, get_detailed_history

# Load environment variables
load_dotenv()

# Initialize clients
@st.cache_resource
def get_clients():
    exa = Exa(api_key=os.environ.get("EXA_API_KEY"))
    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return exa, anthropic

exa, anthropic = get_clients()

# --- HELPER FUNCTIONS FOR METRICS (reused from app.py) ---

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

def get_webset_domains(webset_category):
    """
    Get domain list for each webset category
    """
    webset_domains = {
        "Academic Research": [
            "arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
            "researchgate.net", "semanticscholar.org", "nature.com",
            "science.org", "pnas.org", "cell.com", "nejm.org",
            "thelancet.com", "bmj.com", "springer.com", "sciencedirect.com"
        ],
        "News & Media": [
            "nytimes.com", "washingtonpost.com", "wsj.com", "reuters.com",
            "apnews.com", "bbc.com", "theguardian.com", "bloomberg.com",
            "npr.org", "pbs.org", "axios.com", "politico.com",
            "theatlantic.com", "economist.com", "ft.com"
        ],
        "Company Engineering Blogs": [
            "openai.com/blog", "engineering.fb.com", "blog.google",
            "aws.amazon.com/blogs", "netflixtechblog.com", "eng.uber.com",
            "stripe.com/blog", "dropbox.tech", "github.blog",
            "engineering.atspotify.com", "medium.com/airbnb-engineering",
            "engineering.linkedin.com", "slack.engineering"
        ],
        "Government & Policy": [
            "whitehouse.gov", "congress.gov", "supremecourt.gov",
            "sec.gov", "ftc.gov", "fda.gov", "cdc.gov", "nih.gov",
            "nasa.gov", "nsf.gov", "energy.gov", "epa.gov"
        ],
        "Financial & Investor Relations": [
            "sec.gov", "investor.apple.com", "investor.google.com",
            "investor.fb.com", "investor.microsoft.com", "investor.amazon.com",
            "investor.netflix.com", "investor.tesla.com", "ir.nvidia.com",
            "seekingalpha.com", "fool.com", "marketwatch.com"
        ]
    }
    return webset_domains.get(webset_category, [])

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
    page_title="Exa Websets vs Claude Web Search",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Exa Websets vs Claude Web Search")
st.markdown("**Vertical-Specific Search Comparison** - Search within curated domain sets")

# --- COMPARISON CRITERIA SECTION ---
with st.expander("📊 What You're Measuring (Websets Edition)", expanded=True):
    st.markdown("""
This tool tests **vertical-specific search** by comparing Exa's domain-filtered websets against Claude's general web search.

### 🎯 **Key Difference: Domain Filtering**

**Exa Websets (Left):**
- ✅ Searches ONLY within pre-selected domains for each vertical
- ✅ Guaranteed source quality (no random blogs, spam, or low-quality sites)
- ✅ Faster results (smaller search space)

**Claude Web Search (Right):**
- ⚠️ Searches the entire web (all domains)
- ⚠️ May include low-quality sources
- ⚠️ Slower results (larger search space)

---

### 📚 **Webset Categories Explained**

#### **1. Academic Research**
**Domains:** arxiv.org, PubMed, Google Scholar, Nature, Science, NEJM, etc.

**What to look for:**
- ✅ Exa should find **peer-reviewed papers** and **preprints**
- ✅ Claude might miss academic sources or include news about research instead
- 📰 **Use case:** Literature reviews, fact-checking scientific claims, finding expert consensus

---

#### **2. News & Media**
**Domains:** NYT, WSJ, Reuters, AP, BBC, Guardian, Bloomberg, NPR, Politico, etc.

**What to look for:**
- ✅ Exa should find **only journalistic coverage** from major outlets
- ✅ Claude might include blogs, opinion pieces, or aggregators
- 📰 **Use case:** Fact-checking, comparing coverage across outlets, finding original reporting

---

#### **3. Company Engineering Blogs**
**Domains:** OpenAI, Meta Engineering, Netflix Tech Blog, AWS Blogs, Stripe, Uber, etc.

**What to look for:**
- ✅ Exa should find **technical deep-dives** from official company sources
- ✅ Claude might include third-party analysis or outdated info
- 📰 **Use case:** Understanding tech implementations, finding case studies, competitive analysis

---

#### **4. Government & Policy**
**Domains:** .gov only (White House, Congress, SEC, FDA, CDC, EPA, etc.)

**What to look for:**
- ✅ Exa should find **official government data and policy documents**
- ✅ Claude might include news ABOUT government vs. official government sources
- 📰 **Use case:** Policy research, regulatory compliance, official data verification

---

#### **5. Financial & Investor Relations**
**Domains:** SEC filings, investor.* subdomains, Seeking Alpha, etc.

**What to look for:**
- ✅ Exa should find **official company disclosures** and **SEC filings**
- ✅ Claude might include retail investor forums or speculation
- 📰 **Use case:** Investment research, due diligence, financial analysis

---

### 🔍 **Comparison Metrics (Same as General App)**

#### **Source Quality (Tier A/B/C)**
- **Websets Expectation:** Should have **100% Tier A** if domains are configured correctly
- **Claude Expectation:** Mix of Tier A/B/C depending on query

#### **Unique Domains**
- **Websets Expectation:** Limited to domains in the webset (10-15 typical)
- **Claude Expectation:** Potentially more diverse, but lower average quality

#### **Total Sources**
- **Both should retrieve similar numbers** (controlled by num_results slider)

#### **Cost**
- **Websets:** Exa API + Claude tokens
- **Claude Web Search:** Claude tokens only

---

### 📰 **When to Use Websets**

**✅ Use Websets when:**
- You need **guaranteed source quality** (e.g., only peer-reviewed papers)
- Researching within a **specific industry/domain** (e.g., only .gov for policy research)
- **Compliance requirements** (e.g., citing only official sources)
- Want to **exclude noise** (no blogs, forums, spam)
- Building **repeatable workflows** (same domains every time)

**⚠️ Don't use Websets when:**
- Need **very recent/breaking news** (websets might be too narrow)
- Researching **cross-industry patterns** (websets are vertical-specific)
- Want **maximum diversity** (websets intentionally limit domains)

---

### 💡 **Reading the Websets Comparison**

After running a comparison:

1. **Check if Exa stayed within the webset**
   - Look at source URLs in left column
   - All should match the webset domains (e.g., all .gov for Government webset)

2. **Compare source quality**
   - Exa websets should have **higher Tier A %**
   - Claude might have more Tier C sources

3. **Compare relevance**
   - Did Exa find the RIGHT content within the domain constraints?
   - Did Claude find relevant sources that Exa missed (outside the webset)?

4. **Evaluate the tradeoff**
   - **Precision vs. Recall**: Websets = high precision (quality), Claude = high recall (coverage)
   - Which matters more for your use case?

---

### 🎯 **Custom Websets**

Use "Custom Domains" to create your own vertical-specific search:
- **Media monitoring**: Specific news outlets you trust
- **Competitor analysis**: Specific company domains
- **Academic research**: Specific journals or institutions
- **Regulatory compliance**: Specific .gov agencies

**Pro tip:** Save your custom domain lists in a text file for reuse!
""")

# Sidebar for Webset configuration
with st.sidebar:
    st.header("🎯 Webset Configuration")

    webset_option = st.selectbox(
        "Select Webset Category",
        [
            "Academic Research",
            "News & Media",
            "Company Engineering Blogs",
            "Government & Policy",
            "Financial & Investor Relations",
            "Custom Domains"
        ],
        help="Choose a pre-configured domain set for vertical-specific search"
    )

    # Show custom domain input if selected
    if webset_option == "Custom Domains":
        custom_domains = st.text_area(
            "Custom Domains (one per line)",
            placeholder="arxiv.org\npubmed.gov\nnature.com",
            help="Enter domains to create your own webset"
        )
    else:
        # Show the domains for the selected webset
        domains = get_webset_domains(webset_option)
        with st.expander(f"📋 Domains in '{webset_option}' ({len(domains)})"):
            for domain in domains:
                st.caption(f"• {domain}")

    st.markdown("---")

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

    search_type = st.selectbox(
        "Search Type",
        ["auto", "neural", "keyword"],
        help="Neural: Semantic/meaning-based. Keyword: Traditional. Auto: Best of both."
    )

    st.markdown("---")
    st.caption("💡 **Tip**: Websets limit search to specific domains, enabling vertical-specific discovery")

# Main search interface
st.markdown("### 🎯 Enter Your Research Question")

query = st.text_area(
    "Research Query",
    value="What are the latest breakthroughs in large language model training efficiency?",
    height=120,
    help="This query will be sent to both approaches"
)

# Example queries based on webset
if webset_option != "Custom Webset ID":
    with st.expander("💡 Example Queries for This Webset"):
        examples = {
            "Academic Research": [
                "Latest breakthroughs in quantum computing error correction",
                "Recent advances in protein folding prediction",
                "New findings in climate change attribution science"
            ],
            "News & Media": [
                "How are major news outlets covering AI regulation?",
                "Recent investigative journalism on social media algorithms",
                "Breaking coverage of renewable energy policy changes"
            ],
            "Company Engineering Blogs": [
                "How are tech companies implementing LLMs in production?",
                "Engineering approaches to scaling infrastructure",
                "Best practices for ML model deployment"
            ],
            "Government & Policy": [
                "Recent federal AI regulation proposals",
                "Government cybersecurity policy updates",
                "Public health data transparency initiatives"
            ],
            "Financial & Investor Relations": [
                "Tech company Q4 2024 earnings reports",
                "Investor concerns about AI capital expenditure",
                "SEC disclosures on data privacy risks"
            ]
        }
        for example in examples.get(webset_option, []):
            st.markdown(f"- {example}")

# Run comparison button
if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a research query")
    elif webset_option == "Custom Domains" and not custom_domains.strip():
        st.error("Please enter at least one domain for custom webset")
    else:
        # Create two columns for results
        col1, col2 = st.columns(2)

        # --- LEFT COLUMN: Exa Websets ---
        with col1:
            st.markdown("### 🎯 Exa Websets")

            try:
                start_time = datetime.now()

                # Create status container
                status_container = st.status(f"🔍 Exa Websets: {webset_option}", expanded=True)

                with status_container:
                    st.write("🎯 Searching webset domains...")

                # Get domains for the selected webset category
                if webset_option == "Custom Domains":
                    webset_domains = [d.strip() for d in custom_domains.split("\n") if d.strip()]
                else:
                    webset_domains = get_webset_domains(webset_option)

                # Direct Exa search with domain filtering (simulating websets)
                search_params = {
                    "query": query,
                    "type": search_type,
                    "num_results": num_results,
                    "contents": {"text": True}
                }

                # Add domain filtering
                if webset_domains:
                    search_params["include_domains"] = webset_domains

                # Execute search
                results = exa.search(**search_params)

                # Build sources
                all_sources = []
                for r in results.results:
                    source = {
                        "title": r.title,
                        "url": r.url,
                        "published": r.published_date if hasattr(r, 'published_date') else None,
                        "excerpt": r.text[:500] if hasattr(r, 'text') and r.text else ""
                    }
                    all_sources.append(source)

                # Store in session state for article generation
                st.session_state.webset_sources = all_sources

                total_time = (datetime.now() - start_time).total_seconds()

                # Update status to complete
                status_container.update(
                    label=f"✅ Exa Websets Complete ({len(all_sources)} sources, {total_time:.1f}s)",
                    state="complete",
                    expanded=False
                )

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

                # Now analyze with Claude
                analysis_status = st.status("🤖 Analyzing with Claude", expanded=True)
                with analysis_status:
                    st.write("🎯 Generating analysis from sources...")

                analysis_prompt = f"""Research question: "{query}"

Sources from webset '{webset_option}':
{json.dumps(all_sources, indent=2)}

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

                analysis_response = anthropic.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )

                analysis_content = analysis_response.content[0].text

                # Update analysis status to complete
                analysis_status.update(
                    label="✅ Analysis Complete",
                    state="complete",
                    expanded=False
                )

                # Calculate and display retrieval metrics
                retrieval_metrics = calculate_retrieval_metrics(
                    all_sources,
                    analysis_response.usage.input_tokens,
                    analysis_response.usage.output_tokens
                )

                # Add traceability to metrics
                traceability = calculate_evidence_traceability(analysis_content, all_sources)
                retrieval_metrics['traceability_rate'] = traceability['traceability_rate']
                retrieval_metrics['supported_claims'] = traceability['supported_claims']
                retrieval_metrics['inferences'] = traceability['inferences']
                retrieval_metrics['unsupported_claims'] = traceability['unsupported_claims']
                retrieval_metrics['tier_a_claims'] = traceability['tier_a_claims']
                retrieval_metrics['tier_b_claims'] = traceability['tier_b_claims']
                retrieval_metrics['tier_c_claims'] = traceability['tier_c_claims']

                display_retrieval_scorecard(retrieval_metrics, "Exa Websets", 1, analysis_content, all_sources)

                # Store Exa metrics in session state for logging
                st.session_state.exa_metrics = retrieval_metrics
                st.session_state.exa_search_count = 1
                st.session_state.webset_category = webset_option

                # Display analysis
                st.markdown("#### 📊 Analysis Result")
                st.markdown(analysis_content)

                # Display traceability prominently after analysis
                st.markdown("---")
                traceability_display = calculate_evidence_traceability(analysis_content)
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
                    st.info(f"📊 **Traceability Rate: {trace_rate}%** ({traceability_display['supported_claims']}/{traceability_display['total_claims']} claims)")

                # Metrics
                st.caption(f"🔢 Tokens: {analysis_response.usage.input_tokens + analysis_response.usage.output_tokens:,}")

            except Exception as e:
                st.error(f"❌ Exa Websets failed: {str(e)}")

        # --- RIGHT COLUMN: Claude Web Search ---
        with col2:
            st.markdown("### 🔶 Claude Web Search")

            with st.spinner("🤖 Claude searching and analyzing..."):
                try:
                    start_time = datetime.now()

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
                            st.warning("⚠️ Rate limit reached. Waiting 60 seconds...")
                            import time
                            time.sleep(60)
                            response = anthropic.messages.create(
                                model="claude-sonnet-4-5-20250929",
                                max_tokens=4000,
                                tools=tools,
                                messages=[{"role": "user", "content": prompt}]
                            )
                        else:
                            raise

                    total_time = (datetime.now() - start_time).total_seconds()

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

                        display_retrieval_scorecard(retrieval_metrics, "Claude Web Search", 0, response_content, sources)

                        # Store Claude metrics in session state for logging
                        st.session_state.claude_metrics = retrieval_metrics

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
                app_type='websets',
                exa_metrics=st.session_state.exa_metrics,
                claude_metrics=st.session_state.claude_metrics,
                exa_search_count=st.session_state.get('exa_search_count', 1),
                webset_category=st.session_state.get('webset_category'),
                search_type=search_type,
                num_results=num_results,
                query_category=webset_option  # Use webset category as query category
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
                app_type='websets',
                exa_metrics=st.session_state.exa_metrics,
                claude_metrics=empty_metrics,
                exa_search_count=st.session_state.get('exa_search_count', 1),
                webset_category=st.session_state.get('webset_category'),
                search_type=search_type,
                num_results=num_results,
                query_category=webset_option  # Use webset category as query category
            )
            st.warning("⚠️ Logged Exa metrics only (Claude search failed)")
            st.info("💡 Scroll to bottom and click '📊 Load Comparison History' to view all logged comparisons")
            # Set flag to show updated stats
            st.session_state.comparison_logged = True

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
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total Queries", stats['total_queries'])
        with stat_col2:
            st.metric("Total Cost", f"${stats['total_cost']:.2f}")
        with stat_col3:
            st.metric("Avg Exa Sources", f"{stats['avg_exa_sources']:.1f}")
        with stat_col4:
            st.metric("Avg Claude Sources", f"{stats['avg_claude_sources']:.1f}")

        st.caption(f"📁 Metrics saved to: `{stats['csv_path']}`")

        # Show when last updated
        if st.session_state.get('comparison_logged', False):
            st.success("✅ Updated with latest comparison")

        st.markdown("---")

        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison History")
        st.caption("🟢 Green = better | 🔴 Red = worse | ↑ Higher is better: Tier A%, Domains%, Traceability%, Tier A/B Claims% | ↓ Lower is better: Tier C Claims%, Inference%, Unsupported%, Tokens, Cost")

        # Prepare data for display
        import pandas as pd

        # Create display dataframe with ALL core metrics
        display_data = []
        for row in history:
            # Calculate percentages
            exa_sources = int(float(row.get('exa_sources', 0) or 0))
            claude_sources = int(float(row.get('claude_sources', 0) or 0))

            exa_tier_a_pct = (int(float(row.get('exa_tier_a', 0) or 0)) / exa_sources * 100) if exa_sources > 0 else 0
            claude_tier_a_pct = (int(float(row.get('claude_tier_a', 0) or 0)) / claude_sources * 100) if claude_sources > 0 else 0

            exa_domain_pct = (int(float(row.get('exa_domains', 0) or 0)) / exa_sources * 100) if exa_sources > 0 else 0
            claude_domain_pct = (int(float(row.get('claude_domains', 0) or 0)) / claude_sources * 100) if claude_sources > 0 else 0

            exa_trace = float(row.get('exa_traceability', 0) or 0)
            claude_trace = float(row.get('claude_traceability', 0) or 0)

            exa_cost = float(row.get('exa_cost', 0) or 0)
            claude_cost = float(row.get('claude_cost', 0) or 0)

            # Calculate inference and unsupported rates (same as app.py)
            exa_supported = int(float(row.get('exa_supported', 0) or 0))
            exa_inferences = int(float(row.get('exa_inferences', 0) or 0))
            exa_unsupported = int(float(row.get('exa_unsupported', 0) or 0))
            exa_total = exa_supported + exa_inferences + exa_unsupported

            exa_inference_pct = (exa_inferences / exa_total * 100) if exa_total > 0 else 0
            exa_unsupported_pct = (exa_unsupported / exa_total * 100) if exa_total > 0 else 0

            claude_supported = int(float(row.get('claude_supported', 0) or 0))
            claude_inferences = int(float(row.get('claude_inferences', 0) or 0))
            claude_unsupported = int(float(row.get('claude_unsupported', 0) or 0))
            claude_total = claude_supported + claude_inferences + claude_unsupported

            claude_inference_pct = (claude_inferences / claude_total * 100) if claude_total > 0 else 0
            claude_unsupported_pct = (claude_unsupported / claude_total * 100) if claude_total > 0 else 0

            # Calculate tier breakdown for supported claims
            exa_tier_a_claims = int(float(row.get('exa_tier_a_claims', 0) or 0))
            exa_tier_b_claims = int(float(row.get('exa_tier_b_claims', 0) or 0))
            exa_tier_c_claims = int(float(row.get('exa_tier_c_claims', 0) or 0))

            claude_tier_a_claims = int(float(row.get('claude_tier_a_claims', 0) or 0))
            claude_tier_b_claims = int(float(row.get('claude_tier_b_claims', 0) or 0))
            claude_tier_c_claims = int(float(row.get('claude_tier_c_claims', 0) or 0))

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
                'Webset': row['webset_category'] or 'N/A',
                'Exa+Claude Sources': exa_sources,
                'Claude WS Sources': claude_sources,
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
                'Exa+Claude Tokens': int(row['exa_tokens']),
                'Claude WS Tokens': int(row['claude_tokens']),
                'Exa+Claude Cost': exa_cost,
                'Claude WS Cost': claude_cost,
            })

        df = pd.DataFrame(display_data)

        # Function to color-code cells (green = better, red = worse)
        def highlight_comparison(row):
            styles = [''] * len(row)
            # Column order: Timestamp(0), Category(1), Query(2), Webset(3),
            #               Exa+Claude Sources(4), Claude WS Sources(5),
            #               Exa+Claude Tier A %(6), Claude WS Tier A %(7),
            #               Exa+Claude Domains %(8), Claude WS Domains %(9),
            #               Exa+Claude Trace %(10), Claude WS Trace %(11),
            #               Exa+Claude Tier A Claims %(12), Claude WS Tier A Claims %(13),
            #               Exa+Claude Tier B Claims %(14), Claude WS Tier B Claims %(15),
            #               Exa+Claude Tier C Claims %(16), Claude WS Tier C Claims %(17),
            #               Exa+Claude Inference %(18), Claude WS Inference %(19),
            #               Exa+Claude Unsupported %(20), Claude WS Unsupported %(21),
            #               Exa+Claude Tokens(22), Claude WS Tokens(23),
            #               Exa+Claude Cost(24), Claude WS Cost(25)

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

            # Tier A Claims % (higher is better)
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

            # Tier C Claims % (lower is better)
            if row['Exa+Claude Tier C Claims %'] < row['Claude WS Tier C Claims %']:
                styles[16] = 'background-color: #d4edda'
                styles[17] = 'background-color: #f8d7da'
            elif row['Exa+Claude Tier C Claims %'] > row['Claude WS Tier C Claims %']:
                styles[16] = 'background-color: #f8d7da'
                styles[17] = 'background-color: #d4edda'

            # Inference % (lower is better)
            if row['Exa+Claude Inference %'] < row['Claude WS Inference %']:
                styles[18] = 'background-color: #d4edda'
                styles[19] = 'background-color: #f8d7da'
            elif row['Exa+Claude Inference %'] > row['Claude WS Inference %']:
                styles[18] = 'background-color: #f8d7da'
                styles[19] = 'background-color: #d4edda'

            # Unsupported % (lower is better)
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

            # Cost (lower is better)
            if row['Exa+Claude Cost'] < row['Claude WS Cost']:
                styles[24] = 'background-color: #d4edda'
                styles[25] = 'background-color: #f8d7da'
            elif row['Exa+Claude Cost'] > row['Claude WS Cost']:
                styles[24] = 'background-color: #f8d7da'
                styles[25] = 'background-color: #d4edda'

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
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.caption("📁 Full data available in CSV for pivot tables and advanced analysis")
        st.caption("💡 Use this table to identify which query types and websets each system excels at")

    else:
        st.info("📊 No comparison history yet. Run a comparison above to start logging metrics!")
else:
    st.info("💡 Click 'Load Comparison History' to view your logged comparisons (won't refresh the page)")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Benefits of Exa Websets

**Vertical-Specific Discovery:**
- **Academic Research**: Search only arxiv, PubMed, Google Scholar, academic journals
- **Company Insights**: Limit to engineering blogs, investor relations, official announcements
- **Policy Analysis**: Focus on .gov domains, policy think tanks, official documents
- **Financial Data**: Target SEC filings, investor relations, financial news

**Key Advantages:**
1. **Noise Reduction**: Eliminate irrelevant sources by domain filtering
2. **Authority Focus**: Ensure all results come from authoritative sources in your vertical
3. **Reproducibility**: Same webset = same domain scope across searches
4. **Efficiency**: Faster, more targeted results vs. open web search

**When to Use Websets:**
- Researching within a specific industry or domain
- Compliance/audit requirements for source quality
- Academic literature reviews
- Competitive intelligence (company blogs/announcements)
- Policy research requiring official sources
""")

st.markdown("---")
st.caption("""
**Note**: This demo uses Exa's `include_domains` parameter to simulate websets functionality.
For production use, you can create actual [Exa Websets](https://exa.ai/docs/websets/api/get-started) for even more powerful vertical-specific search capabilities.
""")
