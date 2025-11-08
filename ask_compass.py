# ask_compass.py (Final Clean Version for Gemini)
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 

# --- Configuration ---
CHROMA_PATH = "chroma_db"

# This prompt template guides the AI on how to answer.
PROMPT_TEMPLATE = """
Answer the user's question based ONLY on the following context.
If you cannot find the answer in the context, clearly state that you don't know, 
and DO NOT make up an answer.

Context:
{context}

---

Question: {question}
"""

def main():
    # 1. API Key check
    if not os.environ.get("GEMINI_API_KEY"): 
        print("🔴 ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set your API key using: $env:GEMINI_API_KEY='YOUR_KEY_HERE'")
        return
        
    # 2. Load Embeddings Model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Load Vector Database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 4. Load LLM (Using Google Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 5. Get User Question
    question = input("\nCollege Compass (Ask a question based on your PDF): ")
    
    # 6. Retrieval: Find top 3 relevant chunks from the DB
    print("\n--- Retrieving relevant documents... ---")
    results = db.similarity_search_with_score(question, k=3)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    
    # 7. Generation: Create the final prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context_text, question=question)
    
    # 8. Send prompt to LLM and get the answer
    print("--- Sending prompt to LLM... ---")
    response = llm.invoke(formatted_prompt)
    
    # 9. Display answer and source
    print("\n\n✅ COMPASS ANSWER:")
    print(response.content)
    
    print("\n--- Source Documents Used ---")
    for doc, score in results:
        source_info = doc.metadata.get('source', 'Unknown Source')
        page_info = doc.metadata.get('page', 'Unknown Page')
        print(f"Source: {source_info} (Page: {page_info}) (Score: {score:.4f})")

if __name__ == "__main__":
    main()