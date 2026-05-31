# 🤖 Multi-Agent Research System

> A production-grade multi-agent AI system where 5 specialised agents collaborate in sequence to produce comprehensive, quality-scored research reports.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

## 🚀 Live Demo

**[👉 Try it live here](https://your-app.streamlit.app)** ← Update after deployment

---

## 🏗️ Agent Architecture

```
User Input (Research Topic)
         │
         ▼
┌─────────────────┐
│  🗺️  PLANNER    │  Breaks topic into 5 structured research questions
│   Agent #1      │  + identifies key focus areas + defines approach
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  🔬 RESEARCHER  │  Investigates each question with detailed findings
│   Agent #2      │  + facts, context, examples, evidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  📊  ANALYST    │  Identifies patterns, trends, critical insights
│   Agent #3      │  + confidence assessment + gap analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ✍️   WRITER    │  Synthesises everything into a polished report
│   Agent #4      │  + executive summary + structured sections
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  🔍   CRITIC    │  Reviews quality across 5 dimensions
│   Agent #5      │  + scores 1-10 + strengths + improvements
└────────┬────────┘
         │
         ▼
  📋 Final Report + Quality Score + Download
```

## ✨ Features

- **Live Pipeline Visualisation** — Watch each agent activate in real-time
- **5 Specialised Agents** — Each with a distinct role and expert prompt
- **Quality Scoring** — Automated 10-point review by the Critic agent
- **Research History** — Track all sessions with scores in sidebar
- **Export Reports** — Download as Markdown or plain text
- **Full Transparency** — View raw output from every agent

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (Llama3-8b-8192) |
| Agent Orchestration | Custom sequential pipeline |
| Prompt Engineering | Role-based expert prompts |
| Language | Python 3.9+ |

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/LAKSHAY-ATREJA/multi-agent-research-system
cd multi-agent-research-system

# Install
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

## 🌐 Deploy to Streamlit Cloud (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set `app.py` as the main file
5. Deploy — live in 2 minutes

## 💡 Example Topics

- *"The impact of RAG systems on enterprise document processing"*
- *"Comparing transformer architectures for production NLP"*
- *"Current state of autonomous AI agents in 2026"*
- *"Best practices for LLM deployment in regulated industries"*

## 🔑 Why Multi-Agent vs Single LLM?

| | Single LLM | Multi-Agent System |
|---|---|---|
| Specialisation | Generic | Each agent is an expert |
| Quality control | None | Critic agent reviews output |
| Transparency | Black box | Full pipeline visibility |
| Output quality | Variable | Consistently structured |

---

Built by [Lakshay Atreja](https://linkedin.com/in/lakshay-atreja) | [GitHub](https://github.com/LAKSHAY-ATREJA)
