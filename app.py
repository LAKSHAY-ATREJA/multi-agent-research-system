import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero p { font-size: 1.1rem; opacity: 0.8; margin: 0.8rem 0 0 0; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 0.8rem 0.2rem 0 0.2rem;
}

.agent-card {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    border: 1px solid #e8e8e8;
    transition: all 0.3s ease;
}
.agent-waiting { background: #f8f9fa; border-color: #dee2e6; opacity: 0.6; }
.agent-running {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-color: #f39c12;
    box-shadow: 0 4px 15px rgba(243,156,18,0.2);
    animation: pulse 1.5s infinite;
}
.agent-done {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-color: #27ae60;
    box-shadow: 0 4px 15px rgba(39,174,96,0.15);
}
.agent-error {
    background: #fef2f2;
    border-color: #e74c3c;
    box-shadow: 0 4px 15px rgba(231,76,60,0.15);
}
@keyframes pulse {
    0% { box-shadow: 0 4px 15px rgba(243,156,18,0.2); }
    50% { box-shadow: 0 4px 25px rgba(243,156,18,0.5); }
    100% { box-shadow: 0 4px 15px rgba(243,156,18,0.2); }
}

.agent-name { font-weight: 700; font-size: 1rem; }
.agent-role { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }
.agent-output {
    background: white;
    border-radius: 8px;
    padding: 0.8rem;
    margin-top: 0.8rem;
    font-size: 0.85rem;
    color: #444;
    border: 1px solid rgba(0,0,0,0.08);
    max-height: 600px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.metric-box {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e8e8e8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #302b63; }
.metric-lbl { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }

.final-report {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #e8e8e8;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    line-height: 1.7;
}

.history-item {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.4rem 0;
    border-left: 3px solid #667eea;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

AGENTS = [
    {
        "id": "planner",
        "name": "Planner",
        "role": "Breaks topic into structured research questions",
        "color": "#e74c3c",
        "prompt": (
            "You are a Research Planner AI. Analyse this topic and create a comprehensive, detailed research plan.\n\n"
            "Topic: {topic}\n\n"
            "Output EXACTLY the following structure. Each research question MUST include a 2-3 sentence context "
            "explaining why this question matters and what angle to investigate:\n\n"
            "RESEARCH PLAN:\n"
            "1. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "2. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "3. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "4. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "5. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "6. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "7. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "8. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "9. [Specific research question]\n"
            "   Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "10. [Specific research question]\n"
            "    Context: [2-3 sentences explaining why this question is important and what specific aspects to explore]\n\n"
            "KEY FOCUS AREAS:\n"
            "- [Area 1 with explanation]\n"
            "- [Area 2 with explanation]\n"
            "- [Area 3 with explanation]\n"
            "- [Area 4 with explanation]\n"
            "- [Area 5 with explanation]\n\n"
            "RESEARCH APPROACH:\n"
            "[4-5 sentences on how to investigate this topic effectively, what sources and methods to use, "
            "and what interdisciplinary angles to consider]\n\n"
            "EXPECTED OUTCOMES:\n"
            "[3-4 sentences describing what the research should ultimately reveal or achieve]"
        ),
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "role": "Investigates each question with detailed findings",
        "color": "#e67e22",
        "prompt": (
            "You are a Domain Researcher AI. Investigate each question from the research plan with exceptional depth and detail.\n\n"
            "Topic: {topic}\n"
            "Research Plan: {planner}\n\n"
            "IMPORTANT: Each question MUST receive a minimum of 500 words of detailed findings. Include specific facts, "
            "statistics, real-world examples, case studies, expert perspectives, and supporting evidence. "
            "Do not be superficial — go deep into every question.\n\n"
            "For each research question provide comprehensive findings. Format as:\n\n"
            "RESEARCH FINDINGS:\n\n"
            "Q1: [question]\n"
            "Findings: [Minimum 500 words. Cover background context, current state, key facts and statistics, "
            "real-world examples or case studies, different perspectives, challenges, and implications. "
            "Use specific data points and concrete examples wherever possible.]\n\n"
            "Q2: [question]\n"
            "Findings: [Minimum 500 words with the same depth — facts, examples, evidence, multiple angles.]\n\n"
            "Q3: [question]\n"
            "Findings: [Minimum 500 words with the same depth — facts, examples, evidence, multiple angles.]\n\n"
            "Q4: [question]\n"
            "Findings: [Minimum 500 words with the same depth — facts, examples, evidence, multiple angles.]\n\n"
            "Q5: [question]\n"
            "Findings: [Minimum 500 words with the same depth — facts, examples, evidence, multiple angles.]\n\n"
            "Q6: [question]\n"
            "Findings: [Minimum 500 words with the same depth — facts, examples, evidence, multiple angles.]\n\n"
            "SYNTHESIS:\n"
            "[3-4 paragraphs drawing connections across all the findings, identifying common themes, "
            "surprising discoveries, and areas that require further investigation.]"
        ),
    },
    {
        "id": "analyst",
        "name": "Analyst",
        "role": "Identifies patterns, trends and insights",
        "color": "#d4ac0d",
        "prompt": (
            "You are a Data Analyst AI. Analyse all research findings and extract deep, comprehensive insights.\n\n"
            "Topic: {topic}\n"
            "Research Findings: {researcher}\n\n"
            "IMPORTANT: Your total analysis MUST be a minimum of 600 words. Every section must be substantive, "
            "evidence-backed, and analytically rigorous. Avoid superficial observations — dig into root causes, "
            "second-order effects, and non-obvious connections.\n\n"
            "Provide:\n"
            "ANALYSIS REPORT:\n\n"
            "KEY PATTERNS IDENTIFIED:\n"
            "- [Pattern 1]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n"
            "- [Pattern 2]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n"
            "- [Pattern 3]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n"
            "- [Pattern 4]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n"
            "- [Pattern 5]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n"
            "- [Pattern 6]: [3-5 sentences explaining the pattern, its evidence across the findings, and its significance]\n\n"
            "CRITICAL INSIGHTS:\n"
            "1. [Insight title]: [2-3 paragraphs exploring why this matters, its implications, and what it means for the future]\n"
            "2. [Insight title]: [2-3 paragraphs exploring why this matters, its implications, and what it means for the future]\n"
            "3. [Insight title]: [2-3 paragraphs exploring why this matters, its implications, and what it means for the future]\n"
            "4. [Insight title]: [2-3 paragraphs exploring why this matters, its implications, and what it means for the future]\n\n"
            "TREND ANALYSIS:\n"
            "[Full paragraph (minimum 150 words) on current trends, trajectory, and where things are headed over the next 3-5 years. "
            "Include specific drivers and inhibitors of each trend.]\n\n"
            "COMPARATIVE ANALYSIS:\n"
            "[2-3 paragraphs comparing different approaches, schools of thought, or solutions identified in the research. "
            "What are the trade-offs? Which approaches show the most promise?]\n\n"
            "CONTRADICTIONS OR GAPS:\n"
            "- [Contradiction/gap 1 with explanation of why it exists and what it means]\n"
            "- [Contradiction/gap 2 with explanation of why it exists and what it means]\n"
            "- [Contradiction/gap 3 with explanation of why it exists and what it means]\n\n"
            "STRATEGIC IMPLICATIONS:\n"
            "[2-3 paragraphs on what decision-makers, practitioners, or researchers should do given these findings]\n\n"
            "CONFIDENCE ASSESSMENT: [High/Medium/Low] -- [3-4 sentences explaining the confidence level, "
            "what would increase or decrease it, and what additional data would be most valuable]"
        ),
    },
    {
        "id": "writer",
        "name": "Writer",
        "role": "Synthesises everything into a polished report",
        "color": "#27ae60",
        "prompt": (
            "You are a Professional Report Writer AI. Create an exceptionally comprehensive, publication-quality research report.\n\n"
            "Topic: {topic}\n"
            "Research Plan: {planner}\n"
            "Findings: {researcher}\n"
            "Analysis: {analyst}\n\n"
            "IMPORTANT: This report MUST be a minimum of 3000-4000 words. Every section must be fully developed with "
            "rich content, detailed explanations, concrete examples, and substantive analysis. Do not use placeholders "
            "or brief summaries — write complete, polished prose throughout. Use all available findings and analysis.\n\n"
            "Write a complete professional report using this structure:\n\n"
            "# {topic}\n\n"
            "## Executive Summary\n"
            "[3-4 substantial paragraphs covering: the scope and purpose of this research, the most critical findings, "
            "key analytical insights, and the primary recommendations or conclusions. This section alone should be 250-300 words.]\n\n"
            "## Background and Context\n"
            "[2-3 full paragraphs covering: historical background, why this topic matters now, the current landscape, "
            "key stakeholders affected, and the broader significance. Minimum 300 words.]\n\n"
            "## Research Scope and Methodology\n"
            "[1-2 paragraphs explaining what questions were investigated, the research approach taken, "
            "and the framework used to analyse the findings. Minimum 150 words.]\n\n"
            "## Key Findings\n"
            "[8-10 major findings, each presented as a subheading with 2-3 sentences of explanation and supporting evidence. "
            "Minimum 400 words total for this section.]\n\n"
            "## In-Depth Analysis\n"
            "### Patterns and Trends\n"
            "[2-3 full paragraphs analysing the dominant patterns identified. Minimum 250 words.]\n\n"
            "### Critical Insights\n"
            "[2-3 full paragraphs on the most important non-obvious insights. Minimum 250 words.]\n\n"
            "### Comparative Perspectives\n"
            "[1-2 paragraphs comparing different viewpoints, approaches, or schools of thought. Minimum 200 words.]\n\n"
            "### Challenges and Limitations\n"
            "[1-2 paragraphs on key challenges, obstacles, and limitations identified. Minimum 150 words.]\n\n"
            "## Implications and Applications\n"
            "### For Practitioners\n"
            "[1-2 paragraphs on practical implications for those working in this area. Minimum 150 words.]\n\n"
            "### For Decision-Makers\n"
            "[1-2 paragraphs on strategic implications for leaders and decision-makers. Minimum 150 words.]\n\n"
            "### For Researchers\n"
            "[1-2 paragraphs on implications for future research and knowledge gaps. Minimum 150 words.]\n\n"
            "## Future Outlook\n"
            "[2-3 paragraphs on where this topic is heading, emerging developments to watch, and what the next 3-5 years "
            "may look like. Minimum 250 words.]\n\n"
            "## Recommendations\n"
            "[6-8 specific, actionable recommendations based on the research. Each recommendation should have a brief "
            "rationale. Minimum 300 words total.]\n\n"
            "## Conclusion\n"
            "[2-3 paragraphs that synthesise the entire report into a coherent narrative conclusion. Restate the most "
            "important insights and their significance. End with a forward-looking statement. Minimum 200 words.]\n\n"
            "## References and Further Reading\n"
            "[List 8-10 types of sources or specific works that would be valuable for deeper research on this topic.]\n\n"
            "---\n"
            "*Report generated by Multi-Agent Research System*"
        ),
    },
    {
        "id": "critic",
        "name": "Critic",
        "role": "Reviews quality and scores the report",
        "color": "#2980b9",
        "prompt": (
            "You are a Quality Control AI. Conduct a rigorous, detailed review of this research report.\n\n"
            "Report: {writer}\n"
            "Original Topic: {topic}\n\n"
            "IMPORTANT: Your quality review MUST be a minimum of 400 words. Each dimension score must be accompanied "
            "by a detailed explanation of 3-5 sentences. Be specific — cite actual passages from the report, "
            "identify exact weaknesses, and provide actionable feedback.\n\n"
            "QUALITY REVIEW:\n\n"
            "DIMENSION SCORES (each out of 10):\n"
            "- Completeness: [X/10] -- [3-5 sentences: What does the report cover well? What important aspects "
            "of the topic are missing or underdeveloped? How well does it address the original research questions?]\n\n"
            "- Accuracy & Evidence: [X/10] -- [3-5 sentences: How well are claims supported by evidence? "
            "Are there unsupported assertions? Does the evidence used appear credible and relevant?]\n\n"
            "- Clarity & Writing: [X/10] -- [3-5 sentences: How clear and readable is the writing? "
            "Are there sections that are confusing, overly technical, or too vague? What is the overall writing quality?]\n\n"
            "- Structure & Flow: [X/10] -- [3-5 sentences: How well is the report organised? "
            "Does it flow logically from section to section? Are headings used effectively? Does the structure serve the content?]\n\n"
            "- Analytical Depth: [X/10] -- [3-5 sentences: How deep is the analysis? Does the report go beyond "
            "surface observations? Are the insights genuinely valuable and non-obvious?]\n\n"
            "- Practical Value: [X/10] -- [3-5 sentences: How useful are the recommendations? "
            "Can practitioners actually act on them? Are the implications clearly explained?]\n\n"
            "OVERALL SCORE: [X/10]\n\n"
            "DETAILED STRENGTHS:\n"
            "- [Strength 1]: [2-3 sentences explaining what works well and why it adds value]\n"
            "- [Strength 2]: [2-3 sentences explaining what works well and why it adds value]\n"
            "- [Strength 3]: [2-3 sentences explaining what works well and why it adds value]\n"
            "- [Strength 4]: [2-3 sentences explaining what works well and why it adds value]\n\n"
            "DETAILED WEAKNESSES:\n"
            "- [Weakness 1]: [2-3 sentences explaining what is weak, why it matters, and how to fix it]\n"
            "- [Weakness 2]: [2-3 sentences explaining what is weak, why it matters, and how to fix it]\n"
            "- [Weakness 3]: [2-3 sentences explaining what is weak, why it matters, and how to fix it]\n\n"
            "MISSING ELEMENTS:\n"
            "- [Missing element 1]: [What it is, why it should be included, and how it would improve the report]\n"
            "- [Missing element 2]: [What it is, why it should be included, and how it would improve the report]\n"
            "- [Missing element 3]: [What it is, why it should be included, and how it would improve the report]\n\n"
            "SPECIFIC RECOMMENDATIONS FOR IMPROVEMENT:\n"
            "1. [Concrete, actionable recommendation with explanation]\n"
            "2. [Concrete, actionable recommendation with explanation]\n"
            "3. [Concrete, actionable recommendation with explanation]\n\n"
            "VERDICT: [Approved for Publication / Needs Minor Revision / Needs Major Revision] -- "
            "[2-3 sentences explaining the verdict, the primary reason, and the most critical next step]"
        ),
    },
]

# Mapping from agent id to which prior outputs it needs as context
AGENT_INPUTS: dict[str, list[str]] = {
    "planner": [],
    "researcher": ["planner"],
    "analyst": ["researcher"],
    "writer": ["planner", "researcher", "analyst"],
    "critic": ["writer"],
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "outputs" not in st.session_state:
    st.session_state.outputs = {}
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "history" not in st.session_state:
    st.session_state.history = []
if "total_reports" not in st.session_state:
    st.session_state.total_reports = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_llm(api_key: str) -> ChatGroq:
    """Return a configured ChatGroq instance."""
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
        groq_api_key=api_key,
        max_tokens=4000,
    )


def run_agent(agent: dict, inputs: dict, llm: ChatGroq) -> str:
    """
    Format the agent prompt with the given inputs and invoke the LLM.

    Raises ValueError for missing template variables and RuntimeError for
    any other LLM / network failures.
    """
    try:
        prompt = agent["prompt"].format(**inputs)
    except KeyError as exc:
        raise ValueError(f"Agent '{agent['id']}' prompt missing variable: {exc}") from exc

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Agent '{agent['id']}' LLM call failed: {exc}") from exc


def extract_score(critic_text: str) -> int:
    """
    Parse the overall quality score from the Critic agent's output.
    Falls back to 7 if no score can be found.
    """
    # Primary pattern: "OVERALL SCORE: 8/10"
    match = re.search(r"OVERALL\s+SCORE\s*:\s*(\d+)\s*/\s*10", critic_text, re.IGNORECASE)
    if match:
        return max(1, min(10, int(match.group(1))))

    # Secondary pattern: look for a standalone number near the label
    for line in critic_text.splitlines():
        if "OVERALL SCORE" in line.upper():
            numbers = re.findall(r"\b(\d{1,2})\b", line)
            for n in numbers:
                val = int(n)
                if 1 <= val <= 10:
                    return val

    return 7  # sensible default


def escape_html(text: str) -> str:
    """Minimal HTML escaping for safe injection into markdown HTML blocks."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# UI – hero
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Multi-Agent Research System</h1>
    <p>Five specialised AI agents collaborate in sequence to produce comprehensive, quality-scored research reports</p>
    <div>
        <span class="badge">Planner</span>
        <span class="badge">Researcher</span>
        <span class="badge">Analyst</span>
        <span class="badge">Writer</span>
        <span class="badge">Critic</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI – sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    _env_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=_env_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com. Can also be set via GROQ_API_KEY in .env.",
    )

    st.divider()
    st.header("Agent Pipeline")
    for agent in AGENTS:
        st.markdown(f"**{agent['name']}** — {agent['role']}")

    st.divider()
    st.header("Session Stats")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Reports", st.session_state.total_reports)
    col_s2.metric("History", len(st.session_state.history))

    if st.session_state.history:
        st.divider()
        st.header("Recent Research")
        for item in reversed(st.session_state.history[-5:]):
            score_label = "High" if item["score"] >= 8 else "Mid" if item["score"] >= 6 else "Low"
            topic_preview = item["topic"][:50] + ("..." if len(item["topic"]) > 50 else "")
            st.markdown(
                f'<div class="history-item">'
                f'<strong>{item["score"]}/10</strong> ({score_label})<br>'
                f"{escape_html(topic_preview)}<br>"
                f'<small style="color:#888">{item["time"]}</small>'
                f"</div>",
                unsafe_allow_html=True,
            )

    if st.session_state.final_report:
        st.divider()
        st.download_button(
            "Download Report",
            st.session_state.final_report,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# UI – main input
# ---------------------------------------------------------------------------

if not api_key:
    st.info(
        "Enter your free Groq API key in the sidebar to get started. "
        "Get one at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

with st.form("research_form", enter_to_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. 'The impact of multi-agent AI systems on enterprise document processing'",
            label_visibility="collapsed",
        )
    with col_btn:
        go = st.form_submit_button("Research", use_container_width=True, type="primary")

st.caption(
    "Example topics: 'Future of RAG systems in enterprise AI' or "
    "'Comparing LLM architectures for production systems'"
)

# ---------------------------------------------------------------------------
# UI – pipeline execution
# ---------------------------------------------------------------------------

if go and topic.strip():
    st.session_state.outputs = {}
    st.session_state.final_report = None

    st.divider()
    st.subheader("Live Agent Pipeline")

    containers = {a["id"]: st.empty() for a in AGENTS}
    llm = build_llm(api_key)
    inputs: dict[str, str] = {"topic": topic.strip()}
    pipeline_failed = False

    for i, agent in enumerate(AGENTS):
        # Render current status of all cards
        for j, a in enumerate(AGENTS):
            out_preview = escape_html(st.session_state.outputs.get(a["id"], ""))
            if j < i:
                containers[a["id"]].markdown(
                    f'<div class="agent-card agent-done">'
                    f'<div class="agent-name">Done: {a["name"]}</div>'
                    f'<div class="agent-role">{a["role"]}</div>'
                    f'<div class="agent-output">{out_preview}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif j == i:
                containers[a["id"]].markdown(
                    f'<div class="agent-card agent-running">'
                    f'<div class="agent-name">{a["name"]} -- Working...</div>'
                    f'<div class="agent-role">{a["role"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                containers[a["id"]].markdown(
                    f'<div class="agent-card agent-waiting">'
                    f'<div class="agent-name">{a["name"]}</div>'
                    f'<div class="agent-role">{a["role"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Inject required prior outputs into inputs dict
        for dep in AGENT_INPUTS.get(agent["id"], []):
            inputs[dep] = st.session_state.outputs.get(dep, "")

        try:
            output = run_agent(agent, inputs, llm)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Agent '{agent['name']}' failed: {exc}")
            containers[agent["id"]].markdown(
                f'<div class="agent-card agent-error">'
                f'<div class="agent-name" style="color:#e74c3c;">Failed: {agent["name"]}</div>'
                f'<div class="agent-role">{escape_html(str(exc))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            pipeline_failed = True
            break

        st.session_state.outputs[agent["id"]] = output
        out_preview = escape_html(output)
        containers[agent["id"]].markdown(
            f'<div class="agent-card agent-done">'
            f'<div class="agent-name">Done: {agent["name"]}</div>'
            f'<div class="agent-role">{agent["role"]}</div>'
            f'<div class="agent-output">{out_preview}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if not pipeline_failed:
        final_report = st.session_state.outputs.get("writer", "")
        critic_output = st.session_state.outputs.get("critic", "")
        score = extract_score(critic_output)

        st.session_state.final_report = final_report
        st.session_state.total_reports += 1
        st.session_state.history.append(
            {
                "topic": topic.strip(),
                "score": score,
                "time": datetime.now().strftime("%H:%M"),
            }
        )

        st.divider()

        # Metrics row
        col_m1, col_m2, col_m3 = st.columns(3)
        score_color = "#27ae60" if score >= 8 else "#f39c12" if score >= 6 else "#e74c3c"
        word_count = len(final_report.split())

        with col_m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val" style="color:{score_color}">{score}/10</div>'
                f'<div class="metric-lbl">Quality Score</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                '<div class="metric-box">'
                '<div class="metric-val">5</div>'
                '<div class="metric-lbl">Agents Used</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{word_count:,}</div>'
                f'<div class="metric-lbl">Words Generated</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Tabbed output
        st.subheader("Final Research Report")
        tab_report, tab_review, tab_all = st.tabs(["Report", "Quality Review", "All Agent Outputs"])

        with tab_report:
            st.markdown(final_report, unsafe_allow_html=False)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Download as Markdown",
                    final_report,
                    file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "Download as Text",
                    final_report,
                    file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with tab_review:
            st.markdown(critic_output if critic_output else "_No critic output available._")

        with tab_all:
            for agent in AGENTS:
                with st.expander(f"{agent['name']} -- Full Output"):
                    agent_out = st.session_state.outputs.get(agent["id"], "Not run yet.")
                    st.text(agent_out)
