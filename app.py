import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Load environment variables
load_dotenv()

load_dotenv()

# Get API key from .env locally or Streamlit Secrets when deployed
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()

st.write("API key loaded:", bool(GOOGLE_API_KEY))

# ---------------------------------------------------
# Extract text from multiple PDFs
# ---------------------------------------------------

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()

            if page_text:
                text += (
                    f"\n\nSource: {pdf.name} | Page: {page_number}\n"
                    f"{page_text}"
                )

    return text


# ---------------------------------------------------
# Split text into chunks
# ---------------------------------------------------

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ---------------------------------------------------
# Create FAISS vector store
# ---------------------------------------------------

def get_vector_store(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ---------------------------------------------------
# Gemini conversational chain
# ---------------------------------------------------

def get_conversational_chain():

    prompt_template = """
Answer the question using ONLY the provided context.

If the answer is not available in the context, say:

"The answer is not available in the uploaded documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = prompt | model

    return chain


# ---------------------------------------------------
# Ask question
# ---------------------------------------------------

def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(
        user_question,
        k=5
    )

    chain = get_conversational_chain()

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = chain.invoke(
        {
            "context": context,
            "question": user_question
        }
    )

    st.write("### Answer")
    st.write(response.content)


# ---------------------------------------------------
# Main application
# ---------------------------------------------------

def main():

    st.set_page_config(
        page_title="Chat With Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Chat With Multiple PDFs")
    st.write(
        "Upload multiple PDF documents and ask questions about them using Gemini."
    )

    # Sidebar
    with st.sidebar:

        st.header("📂 Upload Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if pdf_docs:

            st.write(
                f"**{len(pdf_docs)} PDF(s) selected**"
            )

            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")

        if st.button("🚀 Process PDFs"):

            if pdf_docs:

                with st.spinner(
                    "Processing PDFs and creating embeddings..."
                ):

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error(
                            "No readable text was found in the uploaded PDFs."
                        )
                        st.stop()

                    text_chunks = get_text_chunks(raw_text)

                    get_vector_store(text_chunks)

                    st.success(
                        f"Successfully processed {len(pdf_docs)} PDF(s)!"
                    )

            else:

                st.warning(
                    "Please upload at least one PDF."
                )

    # Question area

    st.subheader("💬 Ask a Question")

    user_question = st.text_input(
        "Ask something about your uploaded PDFs"
    )

    if user_question:

        if os.path.exists("faiss_index"):

            with st.spinner("Searching documents and generating answer..."):
                user_input(user_question)

        else:

            st.warning(
                "Please upload and process your PDFs first."
            )


if __name__ == "__main__":
    main()