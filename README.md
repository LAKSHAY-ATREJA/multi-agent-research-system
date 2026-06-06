# Multi-Agent Research System

A Streamlit application where five specialised AI agents collaborate in sequence to produce comprehensive, quality-scored research reports on any topic. Each agent has a distinct role: planning, researching, analysing, writing, and quality-reviewing. All agents run on Groq's Llama 3 inference API, which is free to use.

## How it works

The pipeline is strictly sequential — each agent receives the outputs of all prior agents as context:

1. Planner: decomposes the topic into five structured research questions and defines a research approach
2. Researcher: investigates each question with detailed findings, facts, context, and examples
3. Analyst: identifies patterns, trends, critical insights, contradictions, and assigns a confidence level
4. Writer: synthesises all prior outputs into a structured research report with executive summary, key findings, and conclusions
5. Critic: reviews the report across five quality dimensions and assigns an overall score out of 10

The final output is a downloadable Markdown or plain-text research report with a quality score.

## Live demo

Deploy your own instance for free on Streamlit Community Cloud (see deployment section below).

## Features

- Live pipeline visualisation showing each agent's status in real time
- Per-agent output panel for full transparency into the reasoning chain
- Quality scoring by the Critic agent across completeness, accuracy, clarity, structure, and analytical depth
- Session history tracking all research topics and scores
- Download reports as Markdown or plain text

## Requirements

- Python 3.9 or later
- A free Groq API key from https://console.groq.com

## Local setup

```bash
git clone https://github.com/LAKSHAY-ATREJA/multi-agent-research-system.git
cd multi-agent-research-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

streamlit run app.py
```

The app opens at http://localhost:8501. Enter your Groq API key in the sidebar (pre-filled if set via .env) and type a research topic to begin.

## Environment variables

| Variable     | Required | Description                              |
|--------------|----------|------------------------------------------|
| GROQ_API_KEY | Yes      | Your Groq API key from console.groq.com  |

## Deployment to Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click "New app", select this repository, and set the main file to `app.py`
4. Under "Advanced settings", add GROQ_API_KEY as a secret
5. Click "Deploy" — the app is live in about two minutes

## Example topics

- The impact of RAG systems on enterprise document processing
- Comparing transformer architectures for production NLP in 2025
- Current state and limitations of autonomous AI agents
- Best practices for LLM deployment in regulated industries
- How vector databases are changing the data infrastructure landscape

## Tech stack

| Component          | Technology                    |
|--------------------|-------------------------------|
| Frontend           | Streamlit                     |
| LLM inference      | Groq (Llama 3 8B)             |
| Agent orchestration| Custom sequential pipeline    |
| Language           | Python 3.9+                   |

## Project structure

```
app.py              Streamlit application with agent definitions and pipeline
requirements.txt    Python dependencies
.env.example        Template for required environment variables
```

## License

MIT. Built by Lakshay Atreja.
