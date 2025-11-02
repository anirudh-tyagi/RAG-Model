import torch
import chromadb
import os
from dotenv import load_dotenv

# --- Imports for Ollama Cloud LLM ---
from langchain_ollama.chat_models import ChatOllama

# --- Imports from your original script for RAG ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuration ---

# Load environment variables from .env file
load_dotenv()

# ChromaDB and Embedding Config (Unchanged)
DB_PATH = "chroma_db"
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW: Ollama Cloud LLM Configuration ---
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = "https://ollama.com" # As per your example
LLM_MODEL_ID = "gpt-oss:120b" # The model you specified

# --- 2. Connect to Ollama Cloud LLM ---

print(f"Connecting to LLM: {LLM_MODEL_ID} at {OLLAMA_BASE_URL}...")

if not OLLAMA_API_KEY:
    print("Error: OLLAMA_API_KEY environment variable is not set.")
    print("Please create a .env file with this variable.")
    exit()

try:
    # Pass the API key in the headers dictionary
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL_ID,
        headers={
            'Authorization': f'Bearer {OLLAMA_API_KEY}'
        },
        temperature=0.7,
        # Add other parameters like num_ctx if needed
    )
    
    print("Successfully connected to Ollama cloud LLM.")

except Exception as e:
    print(f"Error connecting to Ollama cloud LLM: {e}")
    exit()

# --- 3. Connect to ChromaDB and Define Retriever ---
# (This section is identical to your original code)

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
print(f"Embeddings will run on: {DEVICE}")

try:
    # Load the BGE embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Embedding model loaded.")

    # Connect to your persistent ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)
    print(f"Connected to ChromaDB at: {DB_PATH}")

    # Get your specific collection
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Create a retriever object
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve the top 5 most relevant chunks
    )

    print(f"Retriever created for collection: {COLLECTION_NAME}")

except Exception as e:
    print(f"Error connecting to ChromaDB or loading embedding model: {e}")
    print(f"Please ensure '{DB_PATH}' exists and contains a collection named '{COLLECTION_NAME}'.")
    exit()

# --- 4. Define the RAG Chain ---
# (This section is identical to your original code)

print("Building RAG pipeline...")

# This is the prompt template that instructs the LLM
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

# Helper function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG pipeline is ready. You can now ask questions.")

# --- 5. Start Chat Loop ---
# (This section is identical to your original code)

if __name__ == "__main__":
    while True:
        try:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            if not query.strip():
                continue

            print("Searching your docs and generating answer...")
            
            # 1. Invoke the chain
            answer = rag_chain.invoke(query)
            
            # 2. Print the result
            print("\nANSWER:\n", answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

    print("\nChat finished.")