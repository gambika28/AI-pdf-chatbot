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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chat With Multiple PDFs",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET GOOGLE API KEY
# ============================================================

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not configured. "
        "Please add it to your .env file or Streamlit Secrets."
    )
    st.stop()


# ============================================================
# EXTRACT TEXT FROM MULTIPLE PDFS
# ============================================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page_number, page in enumerate(
                pdf_reader.pages,
                start=1
            ):

                page_text = page.extract_text()

                if page_text:

                    text += (
                        f"\n\n"
                        f"Source: {pdf.name} | Page: {page_number}\n"
                        f"{page_text}"
                    )

        except Exception as e:

            st.warning(
                f"Could not read {pdf.name}: {str(e)}"
            )

    return text


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

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

    return vector_store


# ============================================================
# GEMINI CONVERSATIONAL CHAIN
# ============================================================

def get_conversational_chain():

    prompt_template = """
You are an AI assistant that answers questions about uploaded PDF documents.

Use ONLY the information available in the context.

Rules:

1. Answer the user's question clearly and directly.
2. Do not repeat the question.
3. Do not mention "provided context".
4. Do not make up information.
5. Do not use information outside the uploaded documents.
6. Keep the answer relevant to the question.
7. Use bullet points when useful.
8. If the answer cannot be found in the documents, say:

"Answer is not available in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question"
        ]
    )

    chain = prompt | model

    return chain


# ============================================================
# EXTRACT ONLY TEXT FROM GEMINI RESPONSE
# ============================================================

def extract_clean_answer(response):

    content = response.content

    # Gemini can return content as a list
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text = block.get("text", "")

                    if text:
                        text_parts.append(text)

            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(text_parts).strip()

    # If Gemini returns a normal string
    if isinstance(content, str):

        return content.strip()

    return str(content)


# ============================================================
# ASK QUESTION
# ============================================================

def user_input(user_question):

    try:

        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )


        # ----------------------------------------------------
        # LOAD FAISS DATABASE
        # ----------------------------------------------------

        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )


        # ----------------------------------------------------
        # RETRIEVE TOP 4 RELEVANT CHUNKS
        # ----------------------------------------------------

        docs = new_db.similarity_search(
            user_question,
            k=4
        )


        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )


        # ----------------------------------------------------
        # CREATE GEMINI CHAIN
        # ----------------------------------------------------

        chain = get_conversational_chain()


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        response = chain.invoke(
            {
                "context": context,
                "question": user_question
            }
        )


        # ----------------------------------------------------
        # EXTRACT ONLY THE ACTUAL ANSWER
        # ----------------------------------------------------

        answer = extract_clean_answer(
            response
        )


        # ----------------------------------------------------
        # DISPLAY ONLY THE ANSWER
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 Answer"
        )

        st.markdown(
            answer
        )


    except Exception as e:

        st.error(
            "❌ An error occurred while generating the answer."
        )

        st.error(
            str(e)
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "📚 Chat With Multiple PDFs"
    )

    st.write(
        "Upload multiple PDF documents and ask questions "
        "about them using Gemini."
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "📂 Upload Documents"
        )


        # ----------------------------------------------------
        # MULTIPLE PDF UPLOAD
        # ----------------------------------------------------

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )


        # ----------------------------------------------------
        # SHOW SELECTED PDF NAMES
        # ----------------------------------------------------

        if pdf_docs:

            st.write(
                f"**{len(pdf_docs)} PDF(s) selected**"
            )

            for pdf in pdf_docs:

                st.write(
                    f"📄 {pdf.name}"
                )


        # ----------------------------------------------------
        # PROCESS BUTTON
        # ----------------------------------------------------

        if st.button(
            "🚀 Process PDFs",
            use_container_width=True
        ):

            if pdf_docs:

                with st.spinner(
                    "Processing PDFs and creating embeddings..."
                ):

                    # Extract text
                    raw_text = get_pdf_text(
                        pdf_docs
                    )


                    # Check whether PDFs contain text
                    if not raw_text.strip():

                        st.error(
                            "No readable text was found "
                            "in the uploaded PDFs."
                        )

                        st.stop()


                    # Split into chunks
                    text_chunks = get_text_chunks(
                        raw_text
                    )


                    # Create FAISS vector store
                    get_vector_store(
                        text_chunks
                    )


                    # Success message
                    st.success(
                        f"Successfully processed "
                        f"{len(pdf_docs)} PDF(s)!"
                    )


                    st.info(
                        f"📦 Created {len(text_chunks)} text chunks."
                    )


            else:

                st.warning(
                    "Please upload at least one PDF."
                )


    # ========================================================
    # QUESTION AREA
    # ========================================================

    st.subheader(
        "💬 Ask a Question"
    )


    user_question = st.text_input(
        "Ask something about your uploaded PDFs"
    )


    # ========================================================
    # ANSWER
    # ========================================================

    if user_question:

        if os.path.exists(
            "faiss_index"
        ):

            with st.spinner(
                "🔍 Searching documents and generating answer..."
            ):

                user_input(
                    user_question
                )

        else:

            st.warning(
                "⚠️ Please upload and process your PDFs first."
            )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()