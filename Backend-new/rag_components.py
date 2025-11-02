import torch
import chromadb
import os
from dotenv import load_dotenv

# --- Imports for Ollama Cloud LLM ---
from langchain_ollama.chat_models import ChatOllama

# --- Imports for RAG ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuration ---
load_dotenv()

# ChromaDB and Embedding Config
DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ollama Cloud LLM Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = "https://ollama.com"
LLM_MODEL_ID = "gpt-oss:120b"

# --- 2. Global Variables to hold loaded models ---
llm = None
embeddings = None
chroma_client = None

def load_models():
    """
    Loads the LLM, Embedding Model, and Chroma Client into global variables.
    This is called once when the FastAPI app starts.
    """
    global llm, embeddings, chroma_client

    print("--- Loading RAG models ---")
    
    # --- Load LLM ---
    print(f"Connecting to LLM: {LLM_MODEL_ID}...")
    if not OLLAMA_API_KEY:
        print("FATAL Error: OLLAMA_API_KEY environment variable is not set.")
        exit()
    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL_ID,
            headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'},
            temperature=0.7,
        )
        print("Successfully connected to Ollama cloud LLM.")
    except Exception as e:
        print(f"FATAL Error connecting to Ollama cloud LLM: {e}")
        exit()

    # --- Load Embedding Model ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded.")
    except Exception as e:
        print(f"FATAL Error loading embedding model: {e}")
        exit()

    # --- Connect to ChromaDB ---
    # This is "get or create" and is safe. It just ensures the client
    # is ready and the directory exists.
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        print(f"Connected to ChromaDB at: {DB_PATH}")
    except Exception as e:
        print(f"FATAL Error connecting to ChromaDB: {e}")
        exit()
        
    print("--- All RAG models loaded successfully ---")


def get_rag_chain_for_collection(collection_name: str):
    """
    Dynamically creates a RAG chain for a specific collection.
    It reuses the globally loaded models.
    """
    global llm, embeddings, chroma_client

    if not all([llm, embeddings, chroma_client]):
        print("Error: Models are not loaded. Call load_models() first.")
        return None

    print(f"Attempting to build RAG chain for collection: {collection_name}")
    
    # --- MODIFICATION: Check if the collection exists *before* using it ---
    try:
        # Get a list of all collection names
        collection_names = [c.name for c in chroma_client.list_collections()]
        
        if collection_name not in collection_names:
            print(f"Warning: Collection '{collection_name}' does not exist yet.")
            print(f"Available collections: {collection_names}")
            # This is expected if the background task hasn't finished.
            # Returning None will trigger the "still processing" message in main.py
            return None
            
    except Exception as e:
        print(f"Error while checking for collection '{collection_name}': {e}")
        return None
    # --- END MODIFICATION ---

    print(f"Collection '{collection_name}' found. Building retriever...")
    
    try:
        # 1. Create a VectorStore for the specific collection
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        # 2. Create a retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Retrieve top 5 chunks
        )
    except Exception as e:
        print(f"Error creating retriever for collection '{collection_name}': {e}")
        return None

    # 3. Define the RAG prompt template
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer based on the context, just say that you don't know.
    Keep the answer concise and helpful.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Helper function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build and return the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain