from pydantic import BaseModel, Field
from typing import List, Optional

class AnswerResponse(BaseModel):
    """Response model for a question about a document."""
    answer: str = Field(..., description="The generated answer to the question")
    chunks: List[str] = Field(..., description="The source chunks used to generate the answer")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    file_id: str = Field(..., description="The unique ID of the uploaded document")
    filename: str = Field(..., description="The original filename of the document")
    num_chunks: int = Field(..., description="The number of chunks the document was split into")
    message: str = Field("File processed successfully", description="Status message")
    debug_info: bool = Field(False, description="Whether debug information is available")
    text_sample: Optional[str] = Field(None, description="Sample text from the document")