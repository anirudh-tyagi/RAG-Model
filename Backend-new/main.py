from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 
from pathlib import Path
import speech_recognition as sr
import io
from pydub import AudioSegment
import subprocess
import sys
from pydantic import BaseModel  
from contextlib import asynccontextmanager
import re  # <-- ADDED THIS IMPORT

# --- NEW: Import RAG components ---
from rag_components import load_models, get_rag_chain_for_collection

# --- NEW: Lifespan event handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Application startup...")
    load_models()  # Load the LLM and Embedding models
    yield
    # This code runs on shutdown (if needed)
    print("Application shutdown...")

# --- MODIFIED: Initialize FastAPI with the lifespan event ---
app = FastAPI(lifespan=lifespan)

# --- NEW: Pydantic model for the chat request ---
class ChatRequest(BaseModel):
    message: str
    collection_name: str

# -----------------------------------------------------------
# CRITICAL: Configure the path to your Frontend directory.
# -----------------------------------------------------------
ABSOLUTE_FRONTEND_PATH = Path(__file__).parent.parent / "Frontend-new"
PDF_FOLDER = Path(__file__).parent / "pdf"
PDF_FOLDER.mkdir(exist_ok=True) 

try:
    app.mount("/static", StaticFiles(directory=ABSOLUTE_FRONTEND_PATH), name="static")
    print(f"INFO: Successfully mounted Frontend to the /static URL path.")
except RuntimeError as e:
    print(f"FATAL ERROR: StaticFiles could not find the directory at: {ABSOLUTE_FRONTEND_PATH}")
    raise e 

# -----------------------------------------------------------
# NEW: Filename Sanitizer Function
# -----------------------------------------------------------
def sanitize_name(name: str) -> str:
    """Cleans a string to be a valid ChromaDB collection name."""
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove any character that is not a letter, number, underscore, hyphen, or period
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    
    # Ensure it's at least 3 chars long
    if len(name) < 3:
        name = f"doc_{name}"
        
    # Ensure it doesn't start or end with a non-alphanumeric char
    # (ChromaDB requires start/end with [a-zA-Z0-9])
    if not name[0].isalnum():
        name = f"c_{name}"
    if not name[-1].isalnum():
        name = f"{name}_c"

    # Ensure it's not too long (Chroma's actual limit is 63)
    return name[:63]

# -----------------------------------------------------------
# PDF Processing Pipeline (MODIFIED)
# -----------------------------------------------------------
def run_processing_pipeline(pdf_path: Path):
    """
    Runs the full PDF processing pipeline (Base, Image-Testo, Emmbed)
    in the background using subprocess.
    """
    try:
        # --- MODIFIED: Use the sanitizer ---
        file_stem = sanitize_name(pdf_path.stem)
        
        # Define the paths for the intermediate and final files
        # Base.py creates an output dir named after the *original* stem
        original_stem = pdf_path.stem
        output_dir = Path(__file__).parent / original_stem
        base_md_file = output_dir / f"{original_stem}.md"
        described_md_file = output_dir / f"{original_stem}_with_descriptions.md"
        
        # Get the path to the current Python executable
        python_executable = sys.executable
        
        print(f"\n--- [PIPELINE START] Processing: {pdf_path.name} (Collection: {file_stem}) ---")

        # --- 1. Run Base.py ---
        # Note: Base.py still uses the *original* PDF path
        print(f"[TASK 1/3] Running Base.py (PDF to Markdown)...")
        cmd_base = [python_executable, "Base.py", str(pdf_path)]
        subprocess.run(cmd_base, check=True, capture_output=True, text=True)
        print(f"[TASK 1/3] COMPLETE. Created: {base_md_file}")

        # --- 2. Run Image-Testo.py ---
        print(f"[TASK 2/3] Running Image-Testo.py (Describing Images)...")
        cmd_image = [
            python_executable, 
            "Image-Testo.py", 
            str(base_md_file),     # input_file
            str(output_dir),       # image_directory
            str(described_md_file) # output_file
        ]
        subprocess.run(cmd_image, check=True, capture_output=True, text=True)
        print(f"[TASK 2/3] COMPLETE. Created: {described_md_file}")

        # --- 3. Run Emmbed.py ---
        # Emmbed.py uses the *sanitized* file_stem as the collection name
        print(f"[TASK 3/3] Running Emmbed.py (Generating Embeddings)...")
        cmd_embed = [
            python_executable,
            "Emmbed.py",
            str(described_md_file), # markdown_file
            file_stem               # collection_name
        ]
        subprocess.run(cmd_embed, check=True, capture_output=True, text=True)
        print(f"[TASK 3/3] COMPLETE. Embedded to collection: '{file_stem}'")
        
        print(f"--- [PIPELINE SUCCESS] Finished processing: {pdf_path.name} ---")

    except subprocess.CalledProcessError as e:
        # Log errors if any script fails
        print(f"!!!!!! [PIPELINE FAILED] for {pdf_path.name} !!!!!!")
        print(f"COMMAND: {' '.join(e.cmd)}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except Exception as e:
        print(f"!!!!!! [PIPELINE FAILED] with unexpected error for {pdf_path.name}: {e} !!!!!!")


# -----------------------------------------------------------
# Audio Transcription Utility (Unchanged)
# -----------------------------------------------------------
def transcribe_and_translate_audio(audio_content: bytes) -> dict:
    r = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_content))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav", parameters=["-ac", "1", "-ar", "22050"])
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            audio = r.record(source)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file format: {e}")
    
    try:
        text_transcribed = r.recognize_google(audio, language="en-US")
        return {"text_english": text_transcribed}
    except sr.UnknownValueError:
        return {"text_english": "Could not understand audio."}
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Speech API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

# -----------------------------------------------------------
# --- Frontend Serving Endpoints (Unchanged) ---
# -----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_upload_page():
    upload_file_path = ABSOLUTE_FRONTEND_PATH / "upload.html"
    if not upload_file_path.exists():
        return HTMLResponse(status_code=404, content="<h1>404 Not Found</h1><p>upload.html not found.</p>")
    with open(upload_file_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_page():
    chat_file_path = ABSOLUTE_FRONTEND_PATH / "index.html"
    if not chat_file_path.exists():
        return HTMLResponse(status_code=404, content="<h1>404 Not Found</h1><p>index.html not found.</p>")
    with open(chat_file_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

# -----------------------------------------------------------
# --- API Endpoints ---
# -----------------------------------------------------------

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Receives a PDF, stores it, and triggers the background processing.
    MODIFIED: Returns the SANITIZED collection_name to the frontend.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        file_path = PDF_FOLDER / file.filename
        
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  
                buffer.write(content)

        # --- MODIFIED: Get the SANITIZED collection name ---
        collection_name = sanitize_name(Path(file.filename).stem)

        # Start background task to process this file
        background_tasks.add_task(run_processing_pipeline, file_path)
        
        return {
            "filename": file.filename, 
            "message": "File upload successful. Processing started in background.", 
            "path": str(file_path),
            "collection_name": collection_name  # <-- This is now sanitized
        }
        
    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")


@app.post("/transcribe-audio/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Receives an audio file, converts it, and transcribes it (English-only)."""
    try:
        audio_content = await audio_file.read()
        result = transcribe_and_translate_audio(audio_content)
        return result
    except HTTPException as h:
        raise h
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process audio: {e}")

# --- NEW: Chat Endpoint for RAG ---
@app.post("/chat/")
async def handle_chat_message(request: ChatRequest):
    """
    Receives a message and a collection_name,
    gets the RAG chain for that collection,
    and returns the model's answer.
    """
    print(f"Received chat request for collection: {request.collection_name}")
    try:
        # 1. Get the pre-loaded RAG chain for the specific collection
        rag_chain = get_rag_chain_for_collection(request.collection_name)
        
        if rag_chain is None:
            return {"answer": "Sorry, I'm still processing that document or I can't find it. Please wait a moment and try again."}

        # 2. Invoke the chain with the user's message
        answer = rag_chain.invoke(request.message)
        
        # --- Print the answer to the terminal for debugging ---
        print(f"--- RAG Answer: {answer} ---")
        
        # 3. Return the answer
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {e}")