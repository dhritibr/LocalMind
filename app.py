import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. IMPORTS: Using the correct module paths ---
from langchain_community.embeddings import HuggingFaceEmbeddings  # <-- FREE Embeddings
from langchain_chroma import Chroma                                # <-- ChromaDB
from langchain_google_genai import ChatGoogleGenerativeAI         # <-- Gemini LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document # For type hinting

# --- CONFIGURATION & GEMINI API KEY SETUP ---
load_dotenv()
# We keep the Gemini API key configured here for the Chat LLM later on.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Directory to save the persistent Chroma index
# UPDATED: Renamed for clarity as requested
CHROMA_PERSIST_DIR = "chroma_db"


def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    """Splits raw text into chunks."""
    # Using a large chunk size (10000) is generally NOT recommended for RAG
    # as it clogs the context window. Consider using 1000 or 4000 max.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates the vector store using HuggingFace Embeddings (free, local)
    and ChromaDB (persistent).
    """
    # ⚠️ SWITCHED TO FREE HUGGINGFACE EMBEDDINGS to avoid quota error
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # --- CHROMADB IMPLEMENTATION (Creation) ---
    # Chroma.from_texts handles embedding and persistence automatically.
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="pdf_knowledge_base" # Assign a name to the collection
    )
    st.success("Indexing Done with ChromaDB! You can now ask questions.")


def get_conversational_chain():
    """Defines the prompt template and loads the RAG QA chain (using Gemini LLM)."""

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize Gemini LLM (Uses your key, but only for final answer generation)
    # FIX: Changed model to the modern, supported chat model.
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                  temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load the QA Chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """
    Loads the ChromaDB index, retrieves relevant documents, and generates the response.
    """
    # Initialize the SAME Embeddings function used during creation (HuggingFace)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # --- CHROMADB IMPLEMENTATION (Loading) ---
    # Load the existing ChromaDB collection from the persist directory
    try:
        new_db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="pdf_knowledge_base"
        )
    except Exception as e:
        st.error("Error loading ChromaDB. Did you click 'Submit & Process' first?")
        st.error(f"Details: {e}")
        return

    # Perform similarity search (retrieval)
    docs = new_db.similarity_search(user_question)

    # Get the QA chain configured with the LLM
    chain = get_conversational_chain()

    # Run the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # st.write displays content in the main panel
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("LocalMind RAG")
    st.header("LocalMind: Private Document Q&A")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if not GEMINI_API_KEY:
                st.error("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
                return

            if pdf_docs:
                with st.spinner("Processing and Indexing... (This may be slow on the first run)"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Indexing Complete! You can now ask questions in the main chat.")
            else:
                st.warning("Please upload PDF files first.")


if __name__ == "__main__":
    main()