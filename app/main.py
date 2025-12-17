from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# MODULE IMPORTS
from app.core.corrector import AdvancedCorrector
from app.utils.parsers import parse_txt, parse_pdf, parse_docx

# APPLICATION SETUP
app = FastAPI(title="Indonesian Text Correction API")

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SINGLETON MODEL INITIALIZATION
print("Initializing Global Logic...")
global_corrector = AdvancedCorrector()
print("Logic Initialized Successfully.")

# DATA MODELS
class TextRequest(BaseModel):
    text: str

# ROUTES
@app.get("/")
def health_check():
    return {"status": "active", "message": "Service is running."}

@app.post("/correct-raw")
async def correct_raw_text(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    corrected_text = global_corrector.process(request.text)
    logs = global_corrector.changes_log
    
    return {
        "original": request.text,
        "corrected": corrected_text,
        "logs": logs
    }

@app.post("/correct-file")
async def correct_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()
    raw_text = ""

    try:
        if filename.endswith(".txt"):
            raw_text = parse_txt(content)
        elif filename.endswith(".pdf"):
            raw_text = parse_pdf(content)
        elif filename.endswith(".docx"):
            raw_text = parse_docx(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use .txt, .pdf, or .docx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing error: {str(e)}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="File is empty or could not be read.")

    corrected_text = global_corrector.process(raw_text)
    logs = global_corrector.changes_log

    return {
        "filename": file.filename,
        "original_preview": raw_text[:500],
        "corrected": corrected_text,
        "logs": logs
    }