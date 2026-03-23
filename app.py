from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from graph import graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Multi-Agent Financial Document Analyst",
    description="AI-powered financial document analysis using LangGraph multi-agent system",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    question: str
    selected_agents: Optional[list] = None
    supervisor_reason: Optional[str] = None
    financial_research: Optional[str] = None
    risk_analysis: Optional[str] = None
    market_sentiment: Optional[str] = None
    final_answer: str


@app.get("/")
def read_root():
    return {
        "message": "Multi-Agent Financial Document Analyst API",
        "endpoints": {
            "/analyze": "POST - Analyze a financial question",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_question(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = graph.invoke({"question": request.question})
        return AnalysisResponse(
            question=result.get("question", ""),
            selected_agents=result.get("selected_agents"),
            supervisor_reason=result.get("supervisor_reason"),
            financial_research=result.get("financial_research"),
            risk_analysis=result.get("risk_analysis"),
            market_sentiment=result.get("market_sentiment"),
            final_answer=result.get("final_answer", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)