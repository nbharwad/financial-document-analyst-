# Multi-Agent Financial Document Analyst

A LangGraph-powered multi-agent system for analyzing financial documents (10-K, 10-Q, earnings reports).

## Architecture

```
User Question
     │
     ▼
Supervisor Agent
     │
     ├──► Financial Research Agent   ──► Revenue, Earnings, P&L
     │
     ├──► Risk Analysis Agent        ──► Risk Factors, Debt, Liabilities
     │
     └──► Market Sentiment Agent     ──► Outlook, Guidance, Analyst Tone
                    │
                    ▼
           Final Synthesized Answer
```

## Tech Stack

- **LangGraph** — Multi-agent orchestration
- **LangChain** — LLM framework
- **FAISS** — Vector similarity search
- **OpenAI** — GPT-4o for analysis
- **FastAPI** — REST API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Add sample PDFs to data/sample_reports/

# Build FAISS index
python tools/retriever.py

# Run the API
python app.py
```

## API Endpoints

- `GET /` — API info
- `GET /health` — Health check
- `POST /analyze` — Analyze financial question

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue and what are the risk factors?"}'
```

## Project Structure

```
financial-document-analyst/
├── agents/
│   ├── supervisor.py          # Routes to correct agent
│   ├── financial_research.py  # Revenue, earnings, P&L
│   ├── risk_analysis.py       # Risk factors, debt
│   └── market_sentiment.py   # Outlook, guidance
├── tools/
│   ├── retriever.py           # FAISS retrieval
│   └── parser.py              # PDF parser
├── data/
│   └── sample_reports/        # Sample PDFs
├── graph.py                   # LangGraph definition
├── app.py                    # FastAPI app
└── requirements.txt
```

## Quick Start