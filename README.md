# Multi-Agent Research System

A Streamlit web application where five specialised AI agents collaborate in a fixed sequential pipeline to produce comprehensive, quality-scored research reports on any topic. Each agent has a distinct responsibility: planning, researching, analysing, writing, and quality-reviewing. All inference runs on Groq's free API using Llama 3 8B.

## How it works

The pipeline is strictly sequential. Every agent receives the outputs of all prior agents as context before it generates its own output.

1. Planner -- decomposes the topic into five structured research questions and defines a research approach
2. Researcher -- investigates each question in depth, providing facts, context, and examples
3. Analyst -- identifies patterns, trends, critical insights, contradictions, and assigns a confidence level
4. Writer -- synthesises all prior outputs into a structured report with an executive summary, key findings, and conclusions
5. Critic -- reviews the report across five quality dimensions (completeness, accuracy, clarity, structure, analytical depth) and assigns an overall score out of 10

The final output is a downloadable Markdown or plain-text report with a quality score.

## Features

- Live pipeline visualisation showing each agent's running, done, or failed status in real time
- Per-agent output panel for full transparency into the reasoning chain
- Quality scoring by the Critic agent with per-dimension breakdown
- Session history panel listing recent topics and scores
- Download the final report as Markdown or plain text
- Command-line demo script that runs the full pipeline without a browser

## Requirements

- Python 3.9 or later
- A free Groq API key from https://console.groq.com

## Installation

```bash
git clone https://github.com/LAKSHAY-ATREJA/multi-agent-research-system.git
cd multi-agent-research-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Local setup

Copy the environment template and fill in your key:

```bash
cp .env.example .env
# Open .env and set GROQ_API_KEY=gsk_...
```

Then launch the web application:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Enter your Groq API key in the sidebar (it is pre-filled automatically if set in .env) and type a research topic to start the pipeline.

## Running the demo

The demo script runs the full pipeline from the command line and saves a Markdown report to the current directory. No browser is required.

```bash
python demo.py
```

To use a custom topic:

```bash
python demo.py "Best practices for deploying large language models in regulated industries"
```

The script prints a trimmed preview of each agent's output as it runs, then saves the complete report to a timestamped file such as `report_Best_practices_for_deploying_20250623_143021.md`.

## Environment variables

| Variable     | Required | Description                             |
|--------------|----------|-----------------------------------------|
| GROQ_API_KEY | Yes      | Your Groq API key from console.groq.com |

The key can be set in a `.env` file in the project root or exported directly in the shell. The Streamlit sidebar also accepts it at runtime if you prefer not to use a file.

## Example output

Running the pipeline on the topic "The current state and future potential of retrieval-augmented generation (RAG) in enterprise AI" produces output similar to the following:

Planner output (excerpt):

```
RESEARCH PLAN:
1. What are the core architectural components of a production RAG system?
2. How do retrieval strategies (dense vs sparse vs hybrid) affect answer quality?
3. What are the main failure modes of RAG in enterprise deployments?
4. How do organisations measure and improve RAG accuracy at scale?
5. What developments in the next two years are most likely to change RAG architecture?
```

Critic scoring (excerpt):

```
DIMENSION SCORES:
- Completeness: 8/10
- Accuracy & Evidence: 7/10
- Clarity & Writing: 9/10
- Structure & Flow: 9/10
- Analytical Depth: 7/10

OVERALL SCORE: 8/10
VERDICT: Approved for Publication
```

## Example topics

- The impact of RAG systems on enterprise document processing
- Comparing transformer architectures for production NLP
- Current state and limitations of autonomous AI agents
- Best practices for LLM deployment in regulated industries
- How vector databases are changing the data infrastructure landscape

## Deployment

### Streamlit Community Cloud (recommended, free)

1. Fork this repository to your GitHub account
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click "New app", select this repository, and set the main file to `app.py`
4. Under "Advanced settings", add `GROQ_API_KEY` as a secret
5. Click "Deploy" -- the app is live in about two minutes

### Render (free tier)

A `render.yaml` is included in the repository. Connect your GitHub account at https://render.com, click "New Web Service", select this repository, and Render will read the configuration automatically. Add `GROQ_API_KEY` in the environment variables panel before deploying.

## Project structure

```
app.py              Streamlit application with agent definitions and pipeline logic
demo.py             Command-line demo that runs the full pipeline without a browser
requirements.txt    Python dependencies
.env.example        Template for required environment variables
render.yaml         Render deployment configuration
```

## Tech stack

| Component           | Technology                 |
|---------------------|----------------------------|
| Frontend            | Streamlit                  |
| LLM inference       | Groq (Llama 3 8B)          |
| LLM client          | LangChain Groq integration |
| Agent orchestration | Custom sequential pipeline |
| Language            | Python 3.9+                |

## License

MIT. Built by Lakshay Atreja.
