# prompts.py
from __future__ import annotations

# 1) Shared format instructions (single source of truth)
ANALYSIS_FORMAT_INSTRUCTIONS = """FORMAT YOUR ANALYSIS LIKE THIS:

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
- Use the claim/quote/source structure for ALL findings
"""


def build_exa_initial_prompt(query: str) -> str:
    return f"""Research question: "{query}"

Use the exa_search tool to find relevant information. You can call it multiple times with different queries if needed.

{ANALYSIS_FORMAT_INSTRUCTIONS}
"""


def build_web_search_prompt(query: str, num_results: int) -> str:
    return f"""Research question: "{query}"

Use web search to find relevant information. Limit yourself to searching for approximately {num_results} sources to match the comparison fairness.

{ANALYSIS_FORMAT_INSTRUCTIONS}
"""


def build_exa_article_prompt(query: str, sources_json: str) -> str:
    return f"""You are a professional journalist writing an article based on research.

Research Question: "{query}"

Sources Retrieved:
{sources_json}

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
- Inline source citations
"""


def build_web_article_prompt(query: str) -> str:
    return f"""You are a professional journalist writing an article.

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

Search thoroughly to match the depth of reporting.
"""


# =====================
# Example Newsroom Prompts (UI Data)
# =====================

QUERY_CATEGORIES = [
    "Fact-Checking & Verification",
    "Source Diversification",
    "Timeline Construction",
    "Document Analysis",
    "Expert Source Finding",
    "Follow-the-Money",
    "Comparative Reporting",
    "Trend Analysis",
    "Investigative Patterns",
]


EXAMPLE_PROMPTS = {
    "Fact-Checking & Verification": [
        "Verify claims that Meta laid off 15% of its workforce today - find official sources",
        "Cross-reference reports about the Federal Reserve emergency meeting - what do official sources say?",
        "Fact-check viral claims that California banned gas stoves - what does the actual legislation say?",
    ],
    "Source Diversification": [
        "Find perspectives on the rail workers strike from labor unions, railroad companies, and government regulators",
        "Get viewpoints on the TikTok ban from civil liberties groups, national security experts, and tech policy researchers",
        "Compare how conservative and progressive outlets are covering the abortion pill court decision",
    ],
    "Timeline Construction": [
        "Build a timeline of Boeing 737 MAX safety concerns from first reports to current FAA statements",
        "Track the evolution of SVB collapse from first warning signs to FDIC takeover",
        "Map the progression of the Norfolk Southern train derailment story from incident to EPA response",
    ],
    "Document Analysis": [
        "Find SEC filings and investor disclosures about FTX in the months before bankruptcy",
        "Locate FDA approval documents and clinical trial data for the new Alzheimer's drug",
        "Find EPA reports on East Palestine water quality after the train derailment",
    ],
    "Expert Source Finding": [
        "Find academic researchers who have published on ChatGPT's impact on education",
        "Identify economists who predicted the 2023 banking crisis in previous papers",
        "Locate climate scientists who have expertise in methane emissions from agriculture",
    ],
    "Follow-the-Money": [
        "Track political donations from crypto executives to members of Congress",
        "Find lobbying disclosures from pharmaceutical companies on insulin pricing legislation",
        "Trace dark money groups funding ads in the 2024 presidential race",
    ],
    "Comparative Reporting": [
        "How does US drug pricing compare to Canada, UK, and Germany for the same medications?",
        "Compare police use-of-force policies across major US cities",
        "Compare AI safety frameworks proposed by EU, UK, and US regulators",
    ],
    "Trend Analysis": [
        "Are mass shootings increasing? Find FBI, CDC, and academic research data",
        "Track trends in remote work policies among Fortune 500 companies 2020-2024",
        "Track corporate diversity hiring trends since George Floyd protests",
    ],
    "Investigative Patterns": [
        "Find patterns in nursing home violations across states before and after COVID",
        "Identify companies with repeat OSHA violations in warehouse safety",
        "Find patterns in police misconduct settlements by department and type of complaint",
    ],
}


def get_example_prompts(category: str) -> list[str]:
    """Return example prompts for a category, with a safe fallback."""
    if category in EXAMPLE_PROMPTS:
        return EXAMPLE_PROMPTS[category]
    return EXAMPLE_PROMPTS[QUERY_CATEGORIES[0]]


measurement_guide_text = """
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
"""
def get_measurement_guide_text():
    return measurement_guide_text