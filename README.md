# HireRight_Hackathon25
**My submission for HireRight Hackathon 2025**

*This project allows you to upload a PDF and ask questions about its content using Microsoft Phi3 Model via Ollama. The application processes PDFs, extracts text, indexes them, and retrieves relevant context to generate answers.*

How It Works
Upload a PDF: Use the UI to upload a document.
Processing: The app extracts text and chunks it for indexing.
Ask Questions: Enter a question in the chat box.
Get Answers: The system retrieves relevant text and responds concisely.

**Dependencies**
streamlit (for UI)
ollama, Phi3 (for LLM inference)
pdfplumber (for PDF extraction)
pytest
langchain (for text processing)
