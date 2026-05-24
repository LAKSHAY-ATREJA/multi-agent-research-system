import os
import streamlit as st
from langchain_groq import ChatGroq
from datetime import datetime

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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

.agent-pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.agent-pill {
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: white;
}
.pill-1 { background: #e74c3c; }
.pill-2 { background: #e67e22; }
.pill-3 { background: #f1c40f; color: #333; }
.pill-4 { background: #27ae60; }
.pill-5 { background: #2980b9; }
.arrow { color: #666; font-size: 1.2rem; }

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
    max-height: 120px;
    overflow-y: auto;
}

.score-display {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 16px;
    color: white;
    margin: 1rem 0;
}
.score-number { font-size: 4rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }

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

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e8e8e8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #302b63; }
.metric-lbl { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Agents config ──────────────────────────────────────────────
AGENTS = [
    {
        "id": "planner",
        "name": "🗺️ Planner",
        "role": "Breaks topic into structured research questions",
        "color": "#e74c3c",
        "prompt": """You are a Research Planner AI. Analyse this topic and create a structured research plan.

Topic: {topic}

Output EXACTLY:
RESEARCH PLAN:
1. [Specific research question]
2. [Specific research question]
3. [Specific research question]
4. [Specific research question]
5. [Specific research question]

KEY FOCUS AREAS:
- [Area 1]
- [Area 2]
- [Area 3]

RESEARCH APPROACH:
[2-3 sentences on how to investigate this topic effectively]"""
    },
    {
        "id": "researcher",
        "name": "🔬 Researcher",
        "role": "Investigates each question with detailed findings",
        "color": "#e67e22",
        "prompt": """You are a Domain Researcher AI. Investigate each question from the research plan thoroughly.

Topic: {topic}
Research Plan: {planner}

For each research question provide detailed findings. Format as:

RESEARCH FINDINGS:

Q1: [question]
Findings: [detailed findings with facts, context, examples]

Q2: [question]  
Findings: [detailed findings]

Q3: [question]
Findings: [detailed findings]

Q4: [question]
Findings: [detailed findings]

Q5: [question]
Findings: [detailed findings]"""
    },
    {
        "id": "analyst",
        "name": "📊 Analyst",
        "role": "Identifies patterns, trends and insights",
        "color": "#f1c40f",
        "prompt": """You are a Data Analyst AI. Analyse all research findings and extract key insights.

Topic: {topic}
Research Findings: {researcher}

Provide:
ANALYSIS REPORT:

KEY PATTERNS IDENTIFIED:
- [Pattern 1 with evidence]
- [Pattern 2 with evidence]
- [Pattern 3 with evidence]
- [Pattern 4 with evidence]

CRITICAL INSIGHTS:
1. [Insight + why it matters]
2. [Insight + why it matters]
3. [Insight + why it matters]

TREND ANALYSIS:
[2-3 sentences on trends]

CONTRADICTIONS OR GAPS:
- [Any conflicting information or missing data]

CONFIDENCE ASSESSMENT: [High/Medium/Low] — [reason]"""
    },
    {
        "id": "writer",
        "name": "✍️ Writer",
        "role": "Synthesises everything into a polished report",
        "color": "#27ae60",
        "prompt": """You are a Professional Report Writer AI. Create a comprehensive research report.

Topic: {topic}
Research Plan: {planner}
Findings: {researcher}
Analysis: {analyst}

Write a complete professional report:

# {topic}

## Executive Summary
[3-4 sentence overview of the most important conclusions]

## Background & Context
[Why this topic matters, current landscape]

## Key Findings
[5-6 major findings with supporting evidence, use bullet points]

## In-Depth Analysis
[3-4 paragraphs analysing the findings and their implications]

## Implications & Applications
[What this means in practice, who should care and why]

## Conclusion
[Clear, actionable conclusions — what does this all mean?]

---
*Report generated by Multi-Agent Research System*"""
    },
    {
        "id": "critic",
        "name": "🔍 Critic",
        "role": "Reviews quality and scores the report",
        "color": "#2980b9",
        "prompt": """You are a Quality Control AI. Rigorously review this research report.

Report: {writer}
Original Topic: {topic}

Evaluate critically on 5 dimensions:

QUALITY REVIEW:

DIMENSION SCORES (each out of 10):
- Completeness: [X/10] — [brief reason]
- Accuracy & Evidence: [X/10] — [brief reason]
- Clarity & Writing: [X/10] — [brief reason]
- Structure & Flow: [X/10] — [brief reason]
- Analytical Depth: [X/10] — [brief reason]

OVERALL SCORE: [X/10]

STRENGTHS:
- [Specific strength]
- [Specific strength]
- [Specific strength]

WEAKNESSES:
- [Specific weakness]
- [Specific weakness]

MISSING ELEMENTS:
- [What could have been included]

VERDICT: [Approved for Publication / Needs Revision] — [one sentence reason]"""
    }
]

# ── Session state ─────────────────────────────────────────────
if "outputs" not in st.session_state:
    st.session_state.outputs = {}
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "history" not in st.session_state:
    st.session_state.history = []
if "total_reports" not in st.session_state:
    st.session_state.total_reports = 0


def run_agent(agent, inputs, api_key):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.4,
        groq_api_key=api_key,
        max_tokens=1500
    )
    prompt = agent["prompt"].format(**inputs)
    return llm.invoke(prompt).content


def extract_score(critic_text):
    for line in critic_text.split("\n"):
        if "OVERALL SCORE:" in line:
            try:
                return int(line.split(":")[-1].strip().split("/")[0].strip())
            except:
                pass
    return 7


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🤖 Multi-Agent Research System</h1>
    <p>5 specialised AI agents collaborate in sequence to produce comprehensive, quality-scored research reports</p>
    <div>
        <span class="badge">🗺️ Planner</span>
        <span class="badge">🔬 Researcher</span>
        <span class="badge">📊 Analyst</span>
        <span class="badge">✍️ Writer</span>
        <span class="badge">🔍 Critic</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Get a free key at console.groq.com")

    st.divider()
    st.header("🏗️ Agent Pipeline")
    for agent in AGENTS:
        st.markdown(f"**{agent['name']}** — {agent['role']}")

    st.divider()
    st.header("📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Reports", st.session_state.total_reports)
    col2.metric("History", len(st.session_state.history))

    if st.session_state.history:
        st.divider()
        st.header("📚 Recent Research")
        for item in reversed(st.session_state.history[-5:]):
            score_emoji = "🟢" if item["score"] >= 8 else "🟡" if item["score"] >= 6 else "🔴"
            st.markdown(f"""
<div class="history-item">
    {score_emoji} <strong>{item['score']}/10</strong><br>
    {item['topic'][:45]}...<br>
    <small style="color:#888">{item['time']}</small>
</div>""", unsafe_allow_html=True)

    if st.session_state.final_report:
        st.divider()
        st.download_button(
            "💾 Download Report",
            st.session_state.final_report,
            file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# ── Main ──────────────────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your free Groq API key in the sidebar to begin. Get one at [console.groq.com](https://console.groq.com)")
    st.stop()

col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. 'The impact of multi-agent AI systems on enterprise document processing'",
        label_visibility="collapsed"
    )
with col2:
    go = st.button("🚀 Research", use_container_width=True, type="primary")

# Example topics
st.caption("💡 Try: *'Future of RAG systems in enterprise AI'* or *'Comparing LLM architectures for production systems'*")

if go and topic.strip():
    st.session_state.outputs = {}
    st.session_state.final_report = None

    st.divider()
    st.subheader("⚡ Live Agent Pipeline")

    containers = {a["id"]: st.empty() for a in AGENTS}
    inputs = {"topic": topic}

    for i, agent in enumerate(AGENTS):

        # Render all cards
        for j, a in enumerate(AGENTS):
            if j < i:
                out = st.session_state.outputs.get(a["id"], "")
                containers[a["id"]].markdown(f"""
<div class="agent-card agent-done">
    <div class="agent-name">✅ {a['name']}</div>
    <div class="agent-role">{a['role']}</div>
    <div class="agent-output">{out[:400]}...</div>
</div>""", unsafe_allow_html=True)
            elif j == i:
                containers[a["id"]].markdown(f"""
<div class="agent-card agent-running">
    <div class="agent-name">⚡ {a['name']} — Working...</div>
    <div class="agent-role">{a['role']}</div>
</div>""", unsafe_allow_html=True)
            else:
                containers[a["id"]].markdown(f"""
<div class="agent-card agent-waiting">
    <div class="agent-name">⏳ {a['name']}</div>
    <div class="agent-role">{a['role']}</div>
</div>""", unsafe_allow_html=True)

        # Build inputs
        if agent["id"] == "researcher":
            inputs["planner"] = st.session_state.outputs.get("planner", "")
        elif agent["id"] == "analyst":
            inputs["planner"] = st.session_state.outputs.get("planner", "")
            inputs["researcher"] = st.session_state.outputs.get("researcher", "")
        elif agent["id"] == "writer":
            inputs["planner"] = st.session_state.outputs.get("planner", "")
            inputs["researcher"] = st.session_state.outputs.get("researcher", "")
            inputs["analyst"] = st.session_state.outputs.get("analyst", "")
        elif agent["id"] == "critic":
            inputs["writer"] = st.session_state.outputs.get("writer", "")

        output = run_agent(agent, inputs, api_key)
        st.session_state.outputs[agent["id"]] = output

        containers[agent["id"]].markdown(f"""
<div class="agent-card agent-done">
    <div class="agent-name">✅ {agent['name']}</div>
    <div class="agent-role">{agent['role']}</div>
    <div class="agent-output">{output[:400]}...</div>
</div>""", unsafe_allow_html=True)

    # Final report
    st.session_state.final_report = st.session_state.outputs.get("writer", "")
    score = extract_score(st.session_state.outputs.get("critic", ""))
    st.session_state.total_reports += 1
    st.session_state.history.append({
        "topic": topic,
        "score": score,
        "time": datetime.now().strftime("%H:%M")
    })

    st.divider()

    # Score + metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#27ae60" if score >= 8 else "#f39c12" if score >= 6 else "#e74c3c"
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{score}/10</div><div class="metric-lbl">Quality Score</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">5</div><div class="metric-lbl">Agents Used</div></div>', unsafe_allow_html=True)
    with col3:
        word_count = len(st.session_state.final_report.split())
        st.markdown(f'<div class="metric-box"><div class="metric-val">{word_count}</div><div class="metric-lbl">Words Generated</div></div>', unsafe_allow_html=True)

    st.subheader("📋 Final Research Report")
    tab1, tab2, tab3 = st.tabs(["📄 Report", "🔍 Quality Review", "🔬 All Agent Outputs"])

    with tab1:
        st.markdown(f'<div class="final-report">{st.session_state.final_report.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("💾 Download as Markdown", st.session_state.final_report,
                               file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.md", use_container_width=True)
        with col2:
            st.download_button("📋 Download as Text", st.session_state.final_report,
                               file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", use_container_width=True)

    with tab2:
        critic_out = st.session_state.outputs.get("critic", "")
        st.markdown(f'<div class="final-report">{critic_out.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    with tab3:
        for agent in AGENTS:
            with st.expander(f"{agent['name']} — Full Output"):
                st.text(st.session_state.outputs.get(agent["id"], "Not run yet"))
