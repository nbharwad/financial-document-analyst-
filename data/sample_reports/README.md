# Sample Financial Reports

Place your sample 10-K, 10-Q, or earnings reports (PDF files) in this folder.

The parser will automatically:
1. Read all `.pdf` files in this directory
2. Extract text from each page
3. Create a document with id, filename, and full text content

## Example files to add:
- `aapl_10k_2023.pdf` - Apple 2023 10-K
- `msft_10q_q3_2023.pdf` - Microsoft Q3 2023 10-Q
- `jpm_earnings_q4.pdf` - JPMorgan earnings call transcript

Run the retriever to build the FAISS index:
```bash
python tools/retriever.py
```