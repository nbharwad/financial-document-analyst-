from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools.retriever import RetrieverTool


supervisor_prompt = ChatPromptTemplate.from_template(
    """You are a Supervisor Agent for a multi-agent financial document analysis system.

Your job is to analyze the user's question and determine which specialist agents should be invoked to provide a comprehensive answer.

Available agents:
1. financial_research - For questions about revenue, earnings, profit/loss, financial metrics
2. risk_analysis - For questions about risk factors, debt, liabilities, concerns
3. market_sentiment - For questions about outlook, guidance, analyst tone, future expectations

User Question: {question}

Based on the question, respond with which agents should be called. You can call multiple agents if the question requires diverse analysis.

Respond in the following format:
AGENTS: agent1,agent2 (comma-separated list of agent names)
REASON: Brief explanation of why these agents were chosen"""
)


def create_supervisor():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    chain = supervisor_prompt | llm
    
    def supervisor(state: dict) -> dict:
        question = state["question"]
        response = chain.invoke({"question": question})
        content = str(response.content)
        
        agents = []
        content_lower = content.lower()
        if "financial_research" in content_lower:
            agents.append("financial_research")
        if "risk_analysis" in content_lower:
            agents.append("risk_analysis")
        if "market_sentiment" in content_lower:
            agents.append("market_sentiment")
        
        if not agents:
            agents = ["financial_research"]
        
        return {
            **state,
            "selected_agents": agents,
            "supervisor_reason": content
        }
    
    return supervisor


supervisor_node = create_supervisor()


if __name__ == "__main__":
    state = {"question": "What was the revenue and what are the risk factors?"}
    result = supervisor_node(state)
    print(f"Selected agents: {result['selected_agents']}")
    print(f"Reason: {result['supervisor_reason']}")