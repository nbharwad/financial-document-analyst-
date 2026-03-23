import os
import pickle
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class RetrieverTool:
    def __init__(self, index_path: str = "data/faiss_index"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_path = Path(index_path)
        self.vectorstore = None
    
    def create_index(self, documents: List[Dict]) -> None:
        docs = [
            Document(
                page_content=doc["content"],
                metadata={"id": doc["id"], "file_name": doc["file_name"]}
            )
            for doc in documents
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self._save_index()
    
    def _save_index(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path / "faiss.pkl", "wb") as f:
            pickle.dump(self.vectorstore, f)
    
    def load_index(self) -> None:
        with open(self.index_path / "faiss.pkl", "rb") as f:
            self.vectorstore = pickle.load(f)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.vectorstore is None:
            self.load_index()
        results = self.vectorstore.similarity_search(query, k=k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]


if __name__ == "__main__":
    from parser import PDFParser
    
    parser = PDFParser()
    documents = parser.parse_all_pdfs()
    
    retriever = RetrieverTool()
    retriever.create_index(documents)
    print("FAISS index created")
    
    results = retriever.retrieve("revenue and earnings", k=3)
    print(f"Retrieved {len(results)} documents")