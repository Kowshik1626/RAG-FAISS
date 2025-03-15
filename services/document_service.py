from services.rag_service import create_rag_instance
from services.extraction_service import extract_text_from_pdf, chunk_text
from services.storage_service import add_rag_instance
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def initialize_document(file_path: str, api_key: str):
    """Initialize the RAG system with a document file."""
    # Extract text
    text = extract_text_from_pdf(file_path)
    logger.info(f"Extracted text length: {len(text)}")
    
    if len(text.strip()) < 50:
        logger.warning(f"Extracted text seems too short: '{text[:50]}'")
    
    # Chunk the text
    chunks = chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Create RAG instance
    rag_instance = create_rag_instance(api_key, chunks)
    
    # Get sample text for debugging
    text_sample = chunks[0][:200] + "..." if chunks else "No text extracted"
    
    # Store RAG instance in memory
    add_rag_instance(file_path, rag_instance)
    
    return rag_instance, len(chunks), text_sample