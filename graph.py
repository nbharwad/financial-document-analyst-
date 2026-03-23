from langgraph.graph import StateGraph, END
from typing import TypedDict, List


class AnalystState(TypedDict):
    question: str
    selected_agents: List[str]
    supervisor_reason: str
    financial_research: str
    risk_analysis: str
    market_sentiment: str
    final_answer: str


def create_graph():
    from agents.supervisor import supervisor_node
    from agents.financial_research import financial_research_node
    from agents.risk_analysis import risk_analysis_node
    from agents.market_sentiment import market_sentiment_node
    
    def run_agents_then_synthesize(state: AnalystState) -> dict:
        agents = state.get("selected_agents", [])
        result = {}
        
        if "financial_research" in agents:
            result.update(financial_research_node(state))
        if "risk_analysis" in agents:
            result.update(risk_analysis_node(state))
        if "market_sentiment" in agents:
            result.update(market_sentiment_node(state))
        
        return result
    
    graph = StateGraph(AnalystState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("run_agents", run_agents_then_synthesize)
    graph.add_node("synthesize", synthesizer_node)
    
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "run_agents")
    graph.add_edge("run_agents", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


def synthesizer_node(state: AnalystState) -> dict:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    
    synthesis_prompt = ChatPromptTemplate.from_template(
        """You are a Financial Analysis Synthesizer. Combine the analyses from multiple specialist agents into a cohesive, comprehensive answer.

Question: {question}

Financial Research Analysis:
{financial_research}

Risk Analysis:
{risk_analysis}

Market Sentiment:
{market_sentiment}

Provide a final synthesized answer that:
- Integrates all relevant insights
- Addresses the user's question directly
- Uses proper formatting with headers
- Highlights key findings and numbers"""
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    chain = synthesis_prompt | llm
    
    response = chain.invoke({
        "question": state["question"],
        "financial_research": state.get("financial_research", "No analysis available"),
        "risk_analysis": state.get("risk_analysis", "No analysis available"),
        "market_sentiment": state.get("market_sentiment", "No analysis available")
    })
    
    return {"final_answer": response.content}


graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke({"question": "What was the revenue and what are the risk factors?"})
    print(result.get("final_answer", "No answer"))