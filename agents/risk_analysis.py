from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools.retriever import RetrieverTool


risk_analysis_prompt = ChatPromptTemplate.from_template(
    """You are a Risk Analysis Analyst specializing in identifying financial risks, debt obligations, and liability concerns from financial documents.

Given the user's question about risk factors, retrieve relevant document sections and provide a detailed risk assessment.

User Question: {question}

Retrieved Context:
{context}

Provide analysis covering:
- Debt levels and leverage ratios
- Risk factors disclosed
- Legal liabilities
- Market risks (currency, interest rate)
- Operational risks
- Liquidity concerns
- Any risk mitigation strategies mentioned"""
)


def create_risk_analysis_agent():
    retriever = RetrieverTool()
    
    def risk_analysis_agent(state: dict) -> dict:
        question = state["question"]
        docs = retriever.retrieve(question, k=5)
        context = "\n\n".join([d["content"][:2000] for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        chain = risk_analysis_prompt | llm
        
        response = chain.invoke({"question": question, "context": context})
        
        return {
            **state,
            "risk_analysis": response.content
        }
    
    return risk_analysis_agent


risk_analysis_node = create_risk_analysis_agent()


if __name__ == "__main__":
    state = {"question": "What are the main risk factors and debt obligations?"}
    result = risk_analysis_node(state)
    print(result.get("risk_analysis", "No response"))