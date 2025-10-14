# 🧠LocalMind: Private Document Q&A

LocalMind is a powerful and private question-answering application built with Streamlit. It allows you to have conversations with your PDF documents locally. 💬 By leveraging the capabilities of Google's Gemini Large Language Model, HuggingFace embeddings, and the ChromaDB vector store, you can securely ask questions about your documents without your data ever leaving your local machine.

## Features ✨

*   **Privacy-Focused** 🛡️: All processing and question-answering happens locally. Your documents and queries are not sent to any external servers.
*   **Upload Multiple PDFs** 📂: Easily build a knowledge base from one or more PDF files.
*   **Persistent Vector Store** 💾: Uses ChromaDB to save a persistent index of your document embeddings, avoiding the need to re-process files every time you run the application.
*   **High-Quality Answers** 🤖: Powered by Google's cutting-edge `gemini-2.5-flash` model to provide detailed and contextually accurate answers.
*   **Cost-Effective** 💸: Utilizes the free and efficient `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face for generating text embeddings locally.
*   **Intuitive User Interface** 🖥️: A clean and simple web interface powered by Streamlit for a seamless user experience.

## How It Works ⚙️

This application is built on the principles of Retrieval-Augmented Generation (RAG), which involves the following steps:

1.  **📄 PDF Text Extraction**: The application first extracts the text content from your uploaded PDF files using the `PyPDF2` library.
2.  **✂️ Text Chunking**: The extracted text is divided into smaller, more manageable chunks. This helps the language model to process the information more effectively.
3.  **🧠 Embedding Generation**: Each text chunk is converted into a numerical vector, known as an embedding, which captures its semantic meaning.
4.  **🗄️ Vector Storage**: The text chunks and their embeddings are stored in a ChromaDB vector store on your local machine for efficient searching.
5.  **🔍 Query and Retrieval**: When you ask a question, it is also converted into an embedding. The application then searches the ChromaDB to find the most semantically similar text chunks from your documents.
6.  **💡 Answer Generation**: The retrieved text chunks (the "context") and your original question are provided to the Gemini language model, which then generates a comprehensive answer.

## Getting Started 🚀

### Prerequisites

*   Python 3.7 or higher 
*   A Google API Key for using the Gemini model 

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    You will need to create a `requirements.txt` file with the following content:
    ```
    streamlit
    PyPDF2
    langchain-community
    langchain-chroma
    langchain-google-genai
    langchain-text-splitters
    langchain
    python-dotenv
    google-generativeai
    sentence-transformers
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root of your project directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

### Running the Application ▶️

1.  **Execute the following command in your terminal:**
    ```bash
    streamlit run your_script_name.py
    ```

2.  **Open your web browser and navigate to the local URL provided by Streamlit (e.g., `http://localhost:8501`).**

## How to Use 📖

1.  **📤 Upload Your PDFs**: In the sidebar, use the file uploader to select the PDF files you want to include in your knowledge base.
2.  **🔄 Process the Documents**: Click the "Submit & Process" button. The application will then extract the text, generate embeddings, and create the ChromaDB vector store. A success message will appear upon completion.
3.  **❓ Ask Your Questions**: In the main chat interface, type your questions in the input box and press Enter. LocalMind will then retrieve the relevant information from your documents and generate a detailed answer.
