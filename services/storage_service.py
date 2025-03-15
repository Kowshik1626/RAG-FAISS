import os
import shutil
from fastapi import UploadFile
from typing import Dict, Any, Optional
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# In-memory storage for RAG instances
_rag_instances = {}
_file_id_to_path = {}

def save_file(file: UploadFile, file_id: str, file_ext: str) -> str:
    """Save an uploaded file to disk and return the file path."""
    file_path = f"uploads/{file_id}{file_ext}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"File saved to {file_path}")
    _file_id_to_path[file_id] = file_path
    
    return file_path

def delete_file(file_id: str) -> bool:
    """Delete a file from disk."""
    # Find file path (account for different extensions)
    file_path = _file_id_to_path.get(file_id)
    
    if not file_path and file_id:
        # Try to find the file if not in memory
        for ext in [".pdf", ".txt", ".docx", ""]:
            temp_path = f"uploads/{file_id}{ext}"
            if os.path.exists(temp_path):
                file_path = temp_path
                break
    
    # Remove file if it exists
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted file at {file_path}")
        
        # Remove from path mapping
        if file_id in _file_id_to_path:
            del _file_id_to_path[file_id]
            
        return True
    
    return False

def add_rag_instance(file_path: str, rag_instance: Any) -> str:
    """Add a RAG instance to memory and return the file ID."""
    # Extract file_id from path
    file_id = os.path.basename(file_path).split('.')[0]
    _rag_instances[file_id] = rag_instance
    return file_id

def get_rag_instance(file_id: str) -> Optional[Any]:
    """Get a RAG instance by file ID."""
    return _rag_instances.get(file_id)

def remove_rag_instance(file_id: str) -> bool:
    """Remove a RAG instance from memory."""
    if file_id in _rag_instances:
        del _rag_instances[file_id]
        return True
    return False

def get_active_documents_count() -> int:
    """Get the number of active documents."""
    return len(_rag_instances)