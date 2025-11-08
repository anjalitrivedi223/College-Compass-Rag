# embed_and_store.py (Final complete and robust code for ALL PDFs)
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter 

# --- Configuration ---
DATA_PATH = "." 
CHROMA_PATH = "chroma_db"

def prepare_chunks():
    """Loads ALL PDF documents from the directory and splits them."""
    print("--- 1. Preparing Chunks (Using PDF-Only Loader) ---")
    try:
        # Use DirectoryLoader configured for PDFs only
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader, # Use the dedicated PyPDFLoader
            silent_errors=True 
        )
        
        documents = loader.load()

        if not documents:
            print("ERROR: No PDF documents found or loaded successfully.")
            print(f"Current directory path: {os.getcwd()}") 
            return []
        
        print(f"Loaded {len(documents)} document objects from PDF files.")
        
        # Split Documents 
        text_splitter = CharacterTextSplitter(
            separator="\n\n", 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks.")
        
        return chunks
        
    except ImportError as ie:
        # This catches errors for missing pypdf or langchain-text-splitters
        print(f"CRITICAL ERROR: {ie}. Please ensure all required packages are installed.")
        return []
    except Exception as e:
        print(f"An error occurred during chunk preparation: {e}")
        return []

# --- NEW/MISSING FUNCTION ---
def create_and_save_db(chunks):
    """Converts Chunks into Embeddings and saves them to the Vector Database."""
    if not chunks:
        print("No chunks to process. Exiting.")
        return

    print("--- 2. Creating Embeddings Model ---")
    # Make sure this model is installed: pip install sentence-transformers
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Remove the existing DB folder to ensure a fresh dataset is used
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing DB at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    print("--- 3. Storing Embeddings in ChromaDB ---")
    # Create the Vector Database and persist it to disk
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"✅ Successfully saved {db._collection.count()} vectors to {CHROMA_PATH}!")
    print("Vector Database is now ready.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- SCRIPT STARTING ---") # Confirmation that execution began
    chunks = prepare_chunks()
    create_and_save_db(chunks)