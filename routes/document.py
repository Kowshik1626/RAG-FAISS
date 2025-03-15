from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os

from models.response_models import DocumentUploadResponse
from services.document_service import initialize_document
from services.storage_service import save_file, delete_file, remove_rag_instance
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/documents")

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):
    """Upload a file and initialize the RAG system."""
    # Generate a unique ID for this file
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    logger.info(f"Received file: {file.filename} (ID: {file_id}) with extension {file_ext}")
    
    try:
        # Save the uploaded file
        file_path = save_file(file, file_id, file_ext)
        
        # Initialize RAG system
        rag_instance, num_chunks, text_sample = initialize_document(file_path, api_key)
        
        logger.info(f"RAG system initialized with {num_chunks} chunks")
        
        return DocumentUploadResponse(
            file_id=file_id,
            filename=file.filename,
            num_chunks=num_chunks,
            message="File processed successfully",
            debug_info=True,
            text_sample=text_sample
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Clean up on error
        delete_file(file_id)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.delete("/{file_id}")
async def delete_document(file_id: str):
    """Delete a file and its RAG instance."""
    logger.info(f"Delete request received for file {file_id}")
    
    if not remove_rag_instance(file_id):
        logger.error(f"File ID {file_id} not found for deletion")
        raise HTTPException(status_code=404, detail="File not found")
    
    delete_file(file_id)
    
    return {"message": "File and RAG instance deleted successfully"}