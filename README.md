# E&TC AI Assistant

RAG study assistant: upload PDF manuals/notes, then ask questions, summarize, make viva Qs, quizzes, etc.

## Run
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key                  # or paste it in the sidebar
    streamlit run app.py

Optional: `export GEMINI_MODEL=<model name>` to change the model (default gemini-2.5-flash).

## How it works
PDF -> PyMuPDF text per page -> 1200-char overlapping chunks -> ChromaDB (local embeddings, saved in ./chroma_db)
-> top-8 chunks retrieved per query -> sent to Gemini with page citations.
First run downloads a small embedding model (~80 MB).

## Ideas to extend
- Render PDF pages to images (page.get_pixmap()) and send them to Gemini for diagram-heavy or scanned pages.
- Add a "Sources used" expander showing retrieved chunks.
- Export viva Q&A / quiz to PDF or Anki.
