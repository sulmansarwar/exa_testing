"""
Batch Testing Page - Private page for running automated tests
DO NOT SHARE THIS PAGE PUBLICLY - It will consume API credits
"""

import streamlit as st
import os
from dotenv import load_dotenv
from exa_py import Exa
from anthropic import Anthropic
import json
import time
from datetime import datetime
from metrics_logger import log_comparison
import sys

# Add parent directory to path to import from main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, '.env'))

# Import calculate_retrieval_metrics from main app
def calculate_retrieval_metrics(sources, input_tokens, output_tokens):
    """
    Calculate retrieval quality metrics - duplicated from main app
    """
    from app import classify_source_quality

    # Source quality classification
    quality_counts = {'A': 0, 'B': 0, 'C': 0}
    for source in sources:
        tier = classify_source_quality(source['url'])
        quality_counts[tier] += 1

    total_sources = len(sources)
    tier_a_pct = (quality_counts['A'] / total_sources * 100) if total_sources > 0 else 0
    tier_b_pct = (quality_counts['B'] / total_sources * 100) if total_sources > 0 else 0
    tier_c_pct = (quality_counts['C'] / total_sources * 100) if total_sources > 0 else 0

    # Domain diversity
    domains = set()
    for source in sources:
        from urllib.parse import urlparse
        domain = urlparse(source['url']).netloc
        domains.add(domain)

    domain_diversity_pct = (len(domains) / total_sources * 100) if total_sources > 0 else 0

    # Cost calculation (Claude pricing)
    input_cost = (input_tokens / 1_000_000) * 3.00
    output_cost = (output_tokens / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    return {
        'quality_mix': quality_counts,  # Added for log_comparison compatibility
        'total_sources': total_sources,
        'tier_a_pct': tier_a_pct,
        'tier_b_pct': tier_b_pct,
        'tier_c_pct': tier_c_pct,
        'tier_a_count': quality_counts['A'],
        'tier_b_count': quality_counts['B'],
        'tier_c_count': quality_counts['C'],
        'domain_diversity_pct': domain_diversity_pct,
        'unique_domains': len(domains),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'tokens': input_tokens + output_tokens,  # Added for log_comparison compatibility
        'cost_usd': total_cost  # Renamed from total_cost for log_comparison compatibility
    }

# Initialize clients
@st.cache_resource
def get_clients():
    exa = Exa(api_key=os.environ.get("EXA_API_KEY"))
    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return exa, anthropic

exa, anthropic = get_clients()

# Page config
st.set_page_config(
    page_title="Batch Testing - PRIVATE",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Batch Testing - Automated Query Testing")
st.warning("⚠️ **PRIVATE PAGE** - This will consume API credits. Do not share this link publicly.")

st.markdown("---")

# Example prompts dictionary (same as main app)
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

# Count total prompts
total_prompts = sum(len(prompts) for prompts in example_prompts.values())

# Initialize tested queries tracker in session state
if 'tested_queries' not in st.session_state:
    st.session_state.tested_queries = set()

# Count remaining queries
remaining_queries = total_prompts - len(st.session_state.tested_queries)

st.markdown(f"""
### Batch Test Configuration

**Progress:** {len(st.session_state.tested_queries)} of {total_prompts} queries completed ({remaining_queries} remaining)

This will run example prompts through both Exa+Claude and Claude Web Search approaches.
Each result will be logged to your CSV file for analysis.

**Important:**
- Each query runs BOTH approaches (2 API calls per query)
- Claude rate limit requires ~60s between queries
""")

# Mode selector
batch_mode = st.radio(
    "Batch Mode",
    ["Auto (skip tested queries)", "Manual (select specific queries)"],
    help="Auto mode skips already-tested queries. Manual mode lets you select any queries to run/re-run."
)

# Settings
col1, col2 = st.columns(2)

with col1:
    batch_search_type = st.selectbox(
        "Exa Search Type",
        ["auto", "neural", "keyword"],
        help="Search algorithm for Exa searches"
    )

    batch_num_results = st.slider(
        "Results per query",
        min_value=5,
        max_value=25,
        value=10,
        help="Number of sources to retrieve per query"
    )

with col2:
    batch_delay = st.slider(
        "Delay between queries (seconds)",
        min_value=60,
        max_value=120,
        value=65,
        help="Claude API rate limit requires ~60s. Default 65s for safety."
    )

    # Option to run a subset for testing (only in auto mode)
    if batch_mode == "Auto (skip tested queries)":
        test_mode = st.checkbox(
            "Test mode (next 1 untested query)",
            value=True,
            help="Run only next 1 untested query - great for controlled step-by-step testing without long waits"
        )
    else:
        test_mode = False

# Get list of all prompts
all_prompts_list = []
for category, prompts in example_prompts.items():
    for prompt in prompts:
        all_prompts_list.append((category, prompt))

# Handle Auto vs Manual mode
if batch_mode == "Auto (skip tested queries)":
    # Filter out already tested queries
    untested_prompts = [
        (cat, prompt) for cat, prompt in all_prompts_list
        if prompt not in st.session_state.tested_queries
    ]

    # Determine how many queries will run
    if test_mode:
        queries_to_run = min(1, len(untested_prompts))
        queries_to_test = untested_prompts[:1]
    else:
        queries_to_run = len(untested_prompts)
        queries_to_test = untested_prompts

else:  # Manual mode
    st.markdown("---")
    st.markdown("### 🎯 Manual Query Selection")
    st.info("Select specific queries to run. Already-tested queries are marked with ✅")

    # Initialize selected queries in session state
    if 'manual_selected_queries' not in st.session_state:
        st.session_state.manual_selected_queries = []

    # Create selection interface by category
    selected_queries = []

    for category, prompts in example_prompts.items():
        with st.expander(f"**{category}** ({len(prompts)} queries)", expanded=False):
            for idx, prompt in enumerate(prompts):
                # Check if already tested
                tested_marker = "✅" if prompt in st.session_state.tested_queries else "⬜"

                # Create unique key
                query_key = f"{category}_{idx}"

                # Checkbox for selection
                is_selected = st.checkbox(
                    f"{tested_marker} {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                    key=query_key,
                    help=f"Full query: {prompt}"
                )

                if is_selected:
                    selected_queries.append((category, prompt))

    queries_to_test = selected_queries
    queries_to_run = len(selected_queries)

st.warning(f"⏱️ Estimated time: ~{(queries_to_run * batch_delay) / 60:.1f} minutes for {queries_to_run} queries")

# Show which queries will be tested (only for Auto mode)
if batch_mode == "Auto (skip tested queries)":
    st.markdown("---")
    st.markdown("### 📋 Queries to be Tested")

    if queries_to_run == 0:
        st.success("🎉 **All queries completed!** No more queries to test.")
        if st.button("🔄 Reset Progress (start over)"):
            st.session_state.tested_queries = set()
            st.rerun()
    elif test_mode:
        st.info(f"**Test Mode Active** - Next {queries_to_run} untested queries will run:")
        display_prompts = queries_to_test

        for idx, (category, prompt) in enumerate(display_prompts, 1):
            st.markdown(f"**{idx}. {category}**")
            st.markdown(f"   → {prompt}")
    else:
        st.info(f"**Full Batch Mode** - All {len(queries_to_test)} remaining queries")
        with st.expander("📄 View all untested queries", expanded=False):
            current_category = None
            for category, prompt in queries_to_test:
                if category != current_category:
                    if current_category is not None:
                        st.markdown("")
                    st.markdown(f"**{category}**")
                    current_category = category
                st.markdown(f"- {prompt}")
else:  # Manual mode
    if queries_to_run > 0:
        st.markdown("---")
        st.markdown("### 📋 Selected Queries")
        st.info(f"**{queries_to_run} queries selected**")

        with st.expander("📄 View selected queries", expanded=True):
            for idx, (category, prompt) in enumerate(queries_to_test, 1):
                tested_marker = "✅" if prompt in st.session_state.tested_queries else "⬜"
                st.markdown(f"**{idx}. {tested_marker} {category}**")
                st.markdown(f"   → {prompt}")
    else:
        st.warning("⚠️ No queries selected. Please select at least one query above.")

st.markdown("---")
st.markdown("### 📊 Where Results Appear")
st.info("""
**Results are saved immediately to CSV as each query completes.**

After batch testing completes:
1. Go back to the **main app page** (click 'app' in sidebar)
2. Scroll down to **"📊 Detailed Comparison History"** section
3. You'll see all your batch test results with full metrics
4. Click any row to view detailed analysis and sources
""")

st.markdown("---")

# Disable button if no queries to run
button_disabled = (queries_to_run == 0)
button_label = "🚀 Start Batch Testing" if queries_to_run > 0 else "✅ All Queries Complete"

if st.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
    st.session_state.batch_testing = True
    st.session_state.batch_results = []
    st.session_state.batch_config = {
        'search_type': batch_search_type,
        'num_results': batch_num_results,
        'delay': batch_delay,
        'test_mode': test_mode,
        'batch_mode': batch_mode,
        'queries_to_test': queries_to_test  # Store the specific queries to run
    }
    st.rerun()

# Execute batch testing
if st.session_state.get('batch_testing', False):
    st.markdown("---")
    st.markdown("### 🔬 Batch Testing in Progress...")

    config = st.session_state.batch_config

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Live log console
    st.markdown("### 📜 Live Execution Log")

    # Initialize log if not present
    if 'batch_log' not in st.session_state:
        st.session_state.batch_log = []

    # Create a fixed-height container for logs
    log_display = st.empty()

    def add_log(message, level="info"):
        """Add a log entry with timestamp and update display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "progress": "⏳"
        }.get(level, "📝")
        st.session_state.batch_log.append(f"[{timestamp}] {icon} {message}")

        # Update display with latest logs in a fixed container
        log_text = "\n".join(st.session_state.batch_log[-20:])  # Show last 20 entries
        log_display.text_area(
            "Console Output",
            value=log_text,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )

    # Get test cases from stored config
    all_test_cases = []
    for category, prompt in config['queries_to_test']:
        all_test_cases.append({
            'category': category,
            'prompt': prompt
        })

    total_tests = len(all_test_cases)

    if total_tests == 0:
        st.warning("⚠️ No queries to run!")
        st.session_state.batch_testing = False
        st.rerun()

    # Show mode info
    if config['batch_mode'] == "Auto (skip tested queries)":
        if config['test_mode']:
            st.info(f"🧪 Auto mode - Test mode: Running next {len(all_test_cases)} untested queries")
        else:
            st.info(f"🧪 Auto mode - Full batch: Running {len(all_test_cases)} remaining queries")
    else:
        st.info(f"🎯 Manual mode: Running {len(all_test_cases)} selected queries")

    # Initialize results if not present
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

    # Continue from where we left off
    start_idx = len(st.session_state.batch_results)

    add_log(f"Starting batch test: {total_tests} queries to process", "info")

    for idx in range(start_idx, total_tests):
        test_case = all_test_cases[idx]
        current_progress = idx + 1

        status_text.markdown(f"""
        **Progress: {current_progress}/{total_tests}**

        **Category:** {test_case['category']}

        **Query:** {test_case['prompt'][:150]}{'...' if len(test_case['prompt']) > 150 else ''}
        """)

        progress_bar.progress(current_progress / total_tests)

        add_log(f"Query {current_progress}/{total_tests}: {test_case['category']}", "progress")
        add_log(f"  → {test_case['prompt'][:80]}...", "info")

        try:
            test_query = test_case['prompt']
            test_category = test_case['category']
            test_search_type = config['search_type']
            test_num_results = config['num_results']

            # --- EXA + CLAUDE ---
            add_log("Starting Exa+Claude approach...", "info")
            exa_start = time.time()

            exa_tool = {
                "name": "exa_search",
                "description": f"Search the web using Exa's semantic/neural search. Returns structured results with URLs, titles, excerpts, and publish dates. Search type: {test_search_type}. Max results: {test_num_results}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }

            exa_messages = [{
                "role": "user",
                "content": f"""Research this query and provide a comprehensive analysis:

{test_query}

You have access to Exa search to find relevant sources. Use the search tool to gather information, then analyze what you find."""
            }]

            exa_response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                tools=[exa_tool],
                messages=exa_messages
            )

            # Process tool calls
            exa_sources = []
            search_count = 0
            while exa_response.stop_reason == "tool_use":
                tool_use_block = next(block for block in exa_response.content if block.type == "tool_use")
                search_count += 1

                search_query = tool_use_block.input["query"]
                add_log(f"  Exa search #{search_count}: '{search_query[:60]}...'", "info")

                exa_results = exa.search_and_contents(
                    search_query,
                    type=test_search_type,
                    num_results=test_num_results,
                    text={"max_characters": 2000}
                )

                add_log(f"  Found {len(exa_results.results)} sources from Exa", "success")

                for result in exa_results.results:
                    exa_sources.append({
                        'url': result.url,
                        'title': result.title,
                        'excerpt': result.text[:500] if result.text else ''
                    })

                exa_messages.append({"role": "assistant", "content": exa_response.content})
                exa_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": json.dumps([{
                            'url': r.url,
                            'title': r.title,
                            'text': r.text[:2000] if r.text else '',
                            'published_date': r.published_date
                        } for r in exa_results.results])
                    }]
                })

                add_log(f"  Claude analyzing {len(exa_results.results)} sources...", "info")

                exa_response = anthropic.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    tools=[exa_tool],
                    messages=exa_messages
                )

            exa_analysis = next((block.text for block in exa_response.content if hasattr(block, 'text')), '')
            exa_time = time.time() - exa_start
            exa_input_tokens = exa_response.usage.input_tokens
            exa_output_tokens = exa_response.usage.output_tokens

            add_log(f"Exa+Claude complete: {len(exa_sources)} sources, {exa_time:.1f}s, ${exa_input_tokens/1e6*3 + exa_output_tokens/1e6*15:.4f}", "success")

            # --- CLAUDE WEB SEARCH ---
            add_log("Starting Claude Web Search approach...", "info")
            claude_start = time.time()

            claude_response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": f"""Research this query and provide a comprehensive analysis:

{test_query}

Use web search to find relevant sources and cite them in your analysis."""
                }],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }]
            )

            claude_time = time.time() - claude_start
            claude_input_tokens = claude_response.usage.input_tokens
            claude_output_tokens = claude_response.usage.output_tokens

            # Extract Claude sources and analysis
            claude_sources = []
            claude_analysis = ""
            for block in claude_response.content:
                if block.type == "text":
                    claude_analysis += block.text
                elif block.type == "web_search_tool_result":
                    for result in block.content:
                        if hasattr(result, 'url'):
                            claude_sources.append({
                                'url': result.url,
                                'title': result.title if hasattr(result, 'title') else 'N/A',
                                'excerpt': result.snippet if hasattr(result, 'snippet') else ''
                            })

            add_log(f"Claude Web Search complete: {len(claude_sources)} sources, {claude_time:.1f}s, ${claude_input_tokens/1e6*3 + claude_output_tokens/1e6*15:.4f}", "success")

            # Calculate metrics
            add_log("Calculating metrics...", "info")
            exa_metrics = calculate_retrieval_metrics(exa_sources, exa_input_tokens, exa_output_tokens)
            claude_metrics = calculate_retrieval_metrics(claude_sources, claude_input_tokens, claude_output_tokens)

            # Log to CSV immediately
            add_log("Logging results to CSV...", "info")
            log_comparison(
                query=test_query,
                app_type='general',
                query_category=test_category,
                search_type=test_search_type,
                num_results=test_num_results,
                exa_sources=exa_sources,
                exa_analysis_text=exa_analysis,
                exa_metrics=exa_metrics,
                exa_search_count=search_count,
                claude_sources=claude_sources,
                claude_analysis_text=claude_analysis,
                claude_metrics=claude_metrics
            )

            add_log(f"Query {current_progress}/{total_tests} completed successfully! ✓", "success")

            # Store result for summary
            st.session_state.batch_results.append({
                'category': test_category,
                'prompt': test_query,
                'status': 'success',
                'exa_sources': len(exa_sources),
                'claude_sources': len(claude_sources)
            })

            # Mark query as tested (successful completion)
            st.session_state.tested_queries.add(test_query)

        except Exception as e:
            add_log(f"ERROR: {str(e)}", "error")
            st.session_state.batch_results.append({
                'category': test_case['category'],
                'prompt': test_case['prompt'],
                'status': 'error',
                'error': str(e)
            })

            # Don't mark as tested if it failed - allow retry

        # Rate limiting delay
        if idx < total_tests - 1:
            add_log(f"Waiting {config['delay']}s before next query (rate limiting)...", "warning")
            time.sleep(config['delay'])

        # Force rerun to update progress
        if current_progress < total_tests:
            st.rerun()

    # Batch complete
    progress_bar.progress(1.0)
    status_text.markdown("### ✅ Batch Testing Complete!")

    add_log("=" * 50, "info")
    add_log("BATCH TESTING COMPLETE!", "success")
    add_log("=" * 50, "info")

    # Summary
    successful = sum(1 for r in st.session_state.batch_results if r['status'] == 'success')
    failed = sum(1 for r in st.session_state.batch_results if r['status'] == 'error')

    add_log(f"Results: {successful} successful, {failed} failed", "info")

    # Overall progress
    total_completed = len(st.session_state.tested_queries)
    total_all_queries = sum(len(prompts) for prompts in example_prompts.values())
    remaining = total_all_queries - total_completed

    # Mode-specific message
    mode_note = ""
    if config['batch_mode'] == "Manual (select specific queries)":
        mode_note = "\n\n*Note: Manual mode - queries may have been re-run. Progress tracker only counts unique queries.*"

    st.success(f"""
    **This batch completed!**
    - Queries in this batch: {len(st.session_state.batch_results)}
    - Successful: {successful}
    - Errors: {failed}

    **Overall progress: {total_completed}/{total_all_queries} unique queries completed ({remaining} remaining)**{mode_note}

    All results have been logged to CSV. View them on the main app's Comparison History section.
    """)

    # Show errors if any
    errors = [r for r in st.session_state.batch_results if r['status'] == 'error']
    if errors:
        with st.expander("⚠️ Errors Encountered", expanded=True):
            for err in errors:
                st.error(f"""
                **Category:** {err['category']}

                **Query:** {err['prompt'][:200]}...

                **Error:** {err.get('error', 'Unknown error')}
                """)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if remaining > 0:
            if st.button("▶️ Continue Testing (next batch)", use_container_width=True, type="primary"):
                st.session_state.batch_testing = False
                if 'batch_results' in st.session_state:
                    del st.session_state.batch_results
                if 'batch_config' in st.session_state:
                    del st.session_state.batch_config
                st.rerun()
        else:
            st.info("✅ All queries completed!")

    with col2:
        if st.button("🔄 Reset Progress (start over)", use_container_width=True):
            st.session_state.tested_queries = set()
            st.session_state.batch_testing = False
            if 'batch_results' in st.session_state:
                del st.session_state.batch_results
            if 'batch_config' in st.session_state:
                del st.session_state.batch_config
            if 'batch_log' in st.session_state:
                del st.session_state.batch_log
            st.rerun()

    # Clear logs button
    if st.button("🗑️ Clear Logs", use_container_width=True):
        st.session_state.batch_log = []
        st.rerun()
