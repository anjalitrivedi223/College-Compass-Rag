# app.py (Streamlit Web Interface for RAG)

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 

# --- Configuration ---
CHROMA_PATH = "chroma_db"
MODEL_NAME = "gemini-2.5-flash" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------- RAG Logic -------------------
PROMPT_TEMPLATE = """
Answer the user's question based ONLY on the following context.
If you cannot find the answer in the context, clearly state that you don't know, 
and DO NOT make up an answer.

Context:
{context}

---

Question: {question}
"""

def get_rag_response(question):
    # API Key check
    if "GEMINI_API_KEY" not in os.environ:
        # Since we already checked in Streamlit, this is a fallback error
        return "API Key Error: GEMINI_API_KEY is missing.", None
    
    # Load Embeddings and DB
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Load LLM
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=os.environ["GEMINI_API_KEY"])

    # Retrieval: Find top 3 relevant chunks
    results = db.similarity_search_with_score(question, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    
    # Generation: Create and invoke the prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context_text, question=question)
    
    response = llm.invoke(formatted_prompt)
    
    return response.content, results

# ------------------- Streamlit Interface -------------------

st.title("📚 College Compass RAG System")
st.subheader("Ask questions about the college policy/syllabus!")

# Input field for user question
user_question = st.text_input("Enter your question here:", key="question_input")

if user_question:
    # Button logic (runs when user hits Enter or clicks the button)
    with st.spinner("Searching and generating answer..."):
        try:
            # Check for API Key again before running main logic
            if "GEMINI_API_KEY" not in os.environ:
                 st.error("🔴 GEMINI_API_KEY environment variable is not set. Please set it in your terminal before running the app.")
            else:
                answer, sources = get_rag_response(user_question)
                
                # Display Answer
                st.success("✅ Answer:")
                st.markdown(f"**{answer}**")
                
                # Display Sources
                st.markdown("---")
                st.subheader("🔍 Source Documents Used:")
                for doc, score in sources:
                    source_info = doc.metadata.get('source', 'Unknown Source')
                    page_info = doc.metadata.get('page', 'Unknown Page')
                    st.write(f"- **Source:** `{source_info}` (Page: {page_info}) (Score: {score:.4f})")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your Gemini API key validity.")