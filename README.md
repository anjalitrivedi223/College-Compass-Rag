## 📚 College Compass RAG System

This project is a question-and-answer system that allows you to "chat" with your college documents. It uses a Retrieval-Augmented Generation (RAG) model to find the most relevant information from your PDFs (like the academic calendar, annual reports, and by-laws) and generate a natural language answer.

---

#🧭 How It Works

The project works in two main stages:

1. Ingestion: The embed_and_store.py script reads all the PDF documents, splits them into small chunks, and converts them into vector embeddings. These embeddings are then stored in a local vector database.
2. Q&A:When you ask a question, the ask_compass.py script searches the vector database for the most relevant document chunks. These chunks are then passed (along with your question) to a Large Language Model (LLM) to generate a final, accurate answer.


#✨ Features

* Multi-Document Analysis: Capable of processing and embedding information from multiple PDF files in the project directory.
* Vector Database: Utilizes **ChromaDB** to efficiently store and retrieve document embeddings.
* Gemini Integration: Leverages the **`gemini-2.5-flash`** model via the `langchain-google-genai` library for generation.
* Multiple Interfaces: Supports both a **Command-Line Interface (CLI)** for quick queries and a feature-rich **Streamlit Web App** for a better user experience.
* Source Citation: Provides the source document and page number for every answer, ensuring transparency and verifiability.


# ⚙️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  A valid **Gemini API Key**.


# 🚀 Setup and Installation

Follow these steps to get your project running locally.

# 1. Clone the Repository


# Assuming your code is in a git repository
git clone <your-repository-url>
cd <your-project-directory>


# 2\. Install Dependencies

The project relies on several key libraries, including `langchain` components, `chromadb`, `streamlit`, and `pypdf`.

```bash
pip install -r requirements.txt
# Alternatively, if you don't have a requirements.txt:
pip install langchain-community langchain-core langchain-google-genai streamlit chromadb sentence-transformers pypdf langchain-text-splitters
```

# 3\. Place Documents

Place all the PDF files you want to query (e.g., `academic_calender.pdf`, `Annual-Report.pdf`, `BYE-LAWS.pdf`) directly into the project's root directory.

# 4\. Set Environment Variable

You must set your Gemini API key as an environment variable for the application to function.

  * Linux/macOS:**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
  * Windows (Command Prompt):**
    ```bash
    set GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
  * Windows (PowerShell):**
    ```powershell
    $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    *(Note: The scripts check for `GEMINI_API_KEY`)*

-----

# 🏃 How to Run

The RAG process is a two-step operation: embedding and Querying.

# Step 1: Create the Vector Database

You must run the embedding script first to process your PDF documents and store them in the ChromaDB. This step will create a local folder named `chroma_db`.

```bash
python embed_and_store.py
```

**Expected Output:**
The script will print messages about loading documents, removing the old database (if it exists), creating the embeddings model, and finally confirming the number of vectors saved.

# Step 2: Ask Questions

You have two options for querying the database: the CLI or the Streamlit web application.

# A. Streamlit Web App (Recommended)

Run the web interface:

```bash
streamlit run app.py
```

This will open a local web application in your browser where you can type questions and see the answers with source citations.

# B. Command-Line Interface (CLI)

Run the CLI script for a terminal-based experience:

```bash
python ask_compass.py
```

The script will prompt you: `College Compass (Ask a question based on your PDF):`

-----

### 📁 Project Structure

```
college-compass-rag/

├── chroma_db/                 
│   ├── data_level0.bin
│   ├── header.bin
│   └── length.bin
                                # ⬅️ Source PDF
├── Annual-Report.pdf           # ⬅️ Source PDF
├── BYE-LAWS.pdf                # ⬅️ Source PDF
├── embed_and_store.py         
├── ask_compass.py             
├── app.py
etc                 
