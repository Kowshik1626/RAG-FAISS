from pydantic import BaseModel, Field
from typing import Optional

class QuestionRequest(BaseModel):
    """Request model for asking a question about a document."""
    question: str = Field(..., description="The question to ask about the document")
    
    model_config = {
    "json_schema_extra": {
        "example": {
            "question": "What is the main topic of this document?"
        }
    }
}