import os
from pathlib import Path
from typing import List
from pypdf import PdfReader


class PDFParser:
    def __init__(self, data_dir: str = "data/sample_reports"):
        self.data_dir = Path(data_dir)
    
    def parse_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def parse_all_pdfs(self) -> List[dict]:
        documents = []
        for pdf_file in self.data_dir.glob("*.pdf"):
            text = self.parse_pdf(str(pdf_file))
            documents.append({
                "id": pdf_file.stem,
                "file_name": pdf_file.name,
                "content": text
            })
        return documents


if __name__ == "__main__":
    parser = PDFParser()
    docs = parser.parse_all_pdfs()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc['file_name']}: {len(doc['content'])} chars")