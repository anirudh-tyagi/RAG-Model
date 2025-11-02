import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import time
import sys
import torch

# --- 1. Configuration (now from command-line) ---

if len(sys.argv) < 3:
    print("Error: Missing arguments.")
    print("Usage: python Emmbed.py <path_to_markdown_file> <collection_name>")
    sys.exit(1)

# Data Configuration
MARKDOWN_FILE = sys.argv[1] # The path to your markdown file
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ChromaDB Configuration
CHROMA_PATH = "./chroma_db"  # Directory to store the persistent database
COLLECTION_NAME = sys.argv[2] # A dynamic collection name (e.g., the file stem)

# Embedding Model Configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# -----------------------------------------------


# --- 2. Load Embedding Model ---
print(f"Loading embedding model: {MODEL_NAME}...")
# Use 'cuda' if you have a GPU, otherwise 'cpu'
model = SentenceTransformer(MODEL_NAME, device= "cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded.")

# --- 3. Load, Chunk, and Prepare Document ---
print(f"Loading and splitting document: {MARKDOWN_FILE}...")
loader = UnstructuredMarkdownLoader(MARKDOWN_FILE)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Load and split the document into chunks
docs = loader.load_and_split(text_splitter=text_splitter)
print(f"Document split into {len(docs)} chunks.")

# Prepare data for Chroma
# We need a list of texts, a list of metadatas, and a list of unique IDs
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]
ids = [str(uuid.uuid4()) for _ in texts] # Generate unique IDs for each chunk

# --- 4. Generate Embeddings ---
print("Generating embeddings for all chunks...")
start_time = time.time()
embeddings = model.encode(
    texts,
    normalize_embeddings=True,  # Normalize for BGE, crucial for cosine similarity
    show_progress_bar=True
)
end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

# --- 5. Initialize ChromaDB and Store Data ---
print(f"Initializing ChromaDB at: {CHROMA_PATH}")
# Create a persistent client. Data will be saved to disk
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Get or create the collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Adding {len(texts)} chunks to the '{COLLECTION_NAME}' collection...")
# Add the data to Chroma in a batch
# Note: ChromaDB takes 'documents', not 'texts'
collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
    ids=ids
)

print("Data insertion complete.")

# --- 6. Test Query (Optional) ---
# This part will still run to verify the insertion
print("\n--- Verification Search ---")
query_text = "What is a vector database?"
print(f"Query: '{query_text}'")

# Embed the query
# **Must** use the same model and normalization
query_vector = model.encode(
    query_text,
    normalize_embeddings=True
).tolist()  # Convert to list for Chroma

# Perform the search
# query_embeddings expects a list of embeddings
search_results = collection.query(
    query_embeddings=[query_vector],
    n_results=2  # Number of results to return
)

# Print results
print("Search Results:")
if search_results['documents']:
    for i, (doc, dist) in enumerate(zip(search_results['documents'][0], search_results['distances'][0])):
        print(f"\nResult {i+1}:")
        print(f"  Distance: {dist:.4f}")
        print(f"  Text: {doc[:150]}...")
else:
    print("No results found for verification query.")

print(f"\nDone processing for collection: {COLLECTION_NAME}.")