# Multi-PDF RAG Chatbot
[🚀 Live Demo](https://ai-pdf-chatbot-57mphwqnrcecestwfcgbjq.streamlit.app/)
 **Live Website:** [Try the AI PDF Chatbot](https://multipdfaichatbot.lovable.app/)
An AI-powered chatbot that allows users to upload multiple PDF documents and ask questions about their content.

## Features

- Upload multiple PDF documents
- PDF text extraction
- Text chunking using LangChain
- Gemini embeddings
- FAISS vector database
- Semantic similarity search
- Gemini-powered question answering
- RAG-based responses
- Conversational document Q&A
- Streamlit interface

## Architecture

PDF Documents
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Gemini Embeddings
        ↓
FAISS Vector Database
        ↓
Similarity Search
        ↓
Relevant Context
        ↓
Gemini LLM
        ↓
Answer

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- PyPDF2
- Generative AI
- RAG

## Installation


Go to the project directory:

cd multi-pdf-rag-chatbot

Create a virtual environment:

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create a `.env` file:

GOOGLE_API_KEY=your_api_key

Run the application:

streamlit run app.py
