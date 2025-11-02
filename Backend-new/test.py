import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from langchain_community.llms import HuggingFacePipeline #this is deprecated  
from langchain_huggingface import HuggingFacePipeline
#from langchain_community.embeddings import HuggingFaceEmbeddings #this is deprecated  
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma #this is deprecated  
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuration ---

# Use your specified paths and model names
DB_PATH = "chroma_db"
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Using Qwen1.5-7B-Chat as a high-performance model available on Hugging Face.
# You can swap this for "Qwen/Qwen3-8B" if it becomes available.
LLM_MODEL_ID = "Qwen/Qwen3-8B"

# Determine device for embedding model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Local LLM (Qwen) ---

print(f"Loading LLM: {LLM_MODEL_ID}...")
print("This may take a few minutes and require significant GPU RAM...")

try:
    # Load the tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    # Load the model
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        dtype="auto",  # Use bfloat16 if available
        device_map="auto"    # Automatically uses GPU
    )

    print("LLM loaded successfully.")

    # Create a text-generation pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    # Wrap the pipeline in a LangChain-compatible object
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

except Exception as e:
    print(f"Error loading LLM: {e}")
    print("Please ensure you have enough VRAM and 'transformers', 'torch', and 'accelerate' are installed.")
    exit()

# --- 3. Connect to ChromaDB and Define Retriever ---

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
print(f"Embeddings will run on: {DEVICE}")

try:
    # Load the BGE embedding model
    # We specify model_kwargs to use the GPU (cuda)
    # and encode_kwargs to normalize embeddings (standard for BGE)
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