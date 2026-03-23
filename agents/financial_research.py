from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools.retriever import RetrieverTool
from typing import Any


financial_research_prompt = ChatPromptTemplate.from_template(
    """You are a Financial Research Analyst specializing in analyzing revenue, earnings, and profit/loss statements from financial documents.

Given the user's question about financial performance, retrieve relevant document sections and provide a detailed analysis.

User Question: {question}

Retrieved Context:
{context}

Provide analysis covering:
- Revenue performance and trends
- Earnings per share (EPS)
- Profit/Loss details
- Key financial metrics
- Year-over-year comparisons if available"""
)


def create_financial_research_agent():
    retriever = RetrieverTool()
    
    def financial_research_agent(state: Any) -> dict:
        question = state["question"]
        docs = retriever.retrieve(question, k=5)
        context = "\n\n".join([d["content"][:2000] for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        chain = financial_research_prompt | llm
        
        response = chain.invoke({"question": question, "context": context})
        
        return {
            **state,
            "financial_research": response.content
        }
    
    return financial_research_agent


financial_research_node = create_financial_research_agent()


if __name__ == "__main__":
    state = {"question": "What was the company's revenue and earnings for 2023?"}
    result = financial_research_node(state)
    print(result.get("financial_research", "No response"))