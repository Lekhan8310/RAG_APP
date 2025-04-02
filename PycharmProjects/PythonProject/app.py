import os
import uuid
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Directory to store vector stores
VECTOR_STORE_DIR = Path("vectorstores")
VECTOR_STORE_DIR.mkdir(exist_ok=True)


# Load SentenceTransformer embeddings model
@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Process PDF and create FAISS vector store
def process_pdf_and_create_vector_store(pdf_file, user_id):
    try:
        with st.spinner("Extracting text..."):
            # Extract text from PDF
            reader = PdfReader(pdf_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])

            if not text.strip():
                st.error("No text could be extracted from the PDF.")
                return None

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = splitter.split_text(text)
            st.info(f"Extracted {len(chunks)} text chunks from the PDF.")

        # Create FAISS vector store with embeddings
        with st.spinner("Creating vector store..."):
            embeddings = load_embeddings()
            if not embeddings:
                return None

            vector_store = FAISS.from_texts(chunks, embeddings)

            # Save vector store to disk with a unique name automatically
            safe_name = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in pdf_file.name)
            folder_name = f"{safe_name}_{user_id}_faiss"
            save_path = VECTOR_STORE_DIR / folder_name

            vector_store.save_local(str(save_path))
            st.success(f"Vector store saved automatically at: {folder_name}")
            return str(save_path)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


# List existing vector stores for a user
def list_user_vector_stores(user_id):
    return [
        vs for vs in VECTOR_STORE_DIR.glob(f"*_{user_id}_faiss")
    ]


# Streamlit application
def main():
    st.title("FAISS Vector Store Manager (Automatic Save)")

    # Generate unique user ID for session isolation
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    user_id = st.session_state.user_id

    # Upload PDF section
    st.header("Upload a PDF to Create a Vector Store")
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file:
        process_pdf_and_create_vector_store(pdf_file, user_id)

    # List existing vector stores for the user (no download buttons)
    st.header("Your Saved Vector Stores")
    user_vector_stores = list_user_vector_stores(user_id)

    if not user_vector_stores:
        st.info("You have no saved vector stores yet.")
    else:
        for idx, vs_path in enumerate(user_vector_stores):
            st.write(f"{idx + 1}. **{vs_path.name}**")
            st.write(f"Stored at: {vs_path}")


if __name__ == "__main__":
    main()
