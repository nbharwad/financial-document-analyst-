from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools.retriever import RetrieverTool


market_sentiment_prompt = ChatPromptTemplate.from_template(
    """You are a Market Sentiment Analyst specializing in analyzing company outlook, guidance, and analyst tone from financial documents.

Given the user's question about market sentiment or outlook, retrieve relevant document sections and provide a detailed sentiment analysis.

User Question: {question}

Retrieved Context:
{context}

Provide analysis covering:
- Forward-looking guidance
- Management outlook and tone
- Analyst ratings and recommendations
- Market sentiment indicators
- Confidence levels in future performance
- Any guidance revisions or updates mentioned"""
)


def create_market_sentiment_agent():
    retriever = RetrieverTool()
    
    def market_sentiment_agent(state: dict) -> dict:
        question = state["question"]
        docs = retriever.retrieve(question, k=5)
        context = "\n\n".join([d["content"][:2000] for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        chain = market_sentiment_prompt | llm
        
        response = chain.invoke({"question": question, "context": context})
        
        return {
            **state,
            "market_sentiment": response.content
        }
    
    return market_sentiment_agent


market_sentiment_node = create_market_sentiment_agent()


if __name__ == "__main__":
    state = {"question": "What is the company's outlook and guidance for next year?"}
    result = market_sentiment_node(state)
    print(result.get("market_sentiment", "No response"))