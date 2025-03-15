from fastapi import APIRouter, HTTPException
from models.request_models import QuestionRequest
from models.response_models import AnswerResponse
from services.rag_service import process_question
from services.storage_service import get_rag_instance
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/questions")

@router.post("/{file_id}", response_model=AnswerResponse)
async def ask_question(
    file_id: str,
    request: QuestionRequest
):
    """Ask a question about the uploaded file."""
    logger.info(f"Question received for file {file_id}: '{request.question}'")
    
    rag_instance = get_rag_instance(file_id)
    if not rag_instance:
        logger.error(f"File ID {file_id} not found")
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    
    try:
        answer, chunks = process_question(rag_instance, request.question)
        logger.info(f"Generated answer of length {len(answer)}")
        return AnswerResponse(answer=answer, chunks=chunks)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")