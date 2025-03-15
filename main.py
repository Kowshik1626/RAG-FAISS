from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

# Import routes
from routes.document import router as document_router
from routes.question import router as question_router

# Create necessary directories
Path("uploads").mkdir(parents=True, exist_ok=True)
Path("static").mkdir(parents=True, exist_ok=True)
Path("static/css").mkdir(parents=True, exist_ok=True)
Path("static/js").mkdir(parents=True, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="Document RAG API",
    description="API for answering questions about documents using Retrieval Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(document_router, tags=["Documents"])
app.include_router(question_router, tags=["Questions"])

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Return the HTML interface."""
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/status")
async def get_status():
    """Get the status of the API."""
    from services.storage_service import get_active_documents_count
    return {
        "status": "running", 
        "active_documents": get_active_documents_count()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)