import re
from typing import List
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    try:
        logger.info(f"Attempting to extract text from PDF: {file_path}")
        import pdfplumber
        text = ""
        
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"PDF opened successfully with {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                logger.info(f"Extracted {len(page_text)} characters from page {i+1}")
                text += page_text + "\n\n"
                
        if not text.strip():
            logger.warning("PDF text extraction returned empty text, trying alternate method")
            # If pdfplumber fails, try a different method or library here
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        # Fall back to treating it as a text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                logger.info(f"Fallback text extraction successful, got {len(text)} characters")
            return text
        except Exception as inner_e:
            logger.error(f"Error in fallback text extraction: {str(inner_e)}")
            # For debugging purposes, try to read file as binary
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read(100)  # Read first 100 bytes to check file type
                    logger.info(f"File starts with bytes: {file_data[:20]}")
                return f"Could not extract text. File appears to be binary or corrupted."
            except Exception as bin_error:
                logger.error(f"Cannot read file at all: {str(bin_error)}")
                return f"Error extracting text: {str(e)} -> {str(inner_e)}"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap."""
    logger.info(f"Chunking text of length {len(text)}")
    
    # First, clean the text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size, save current chunk and start new one
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            
            # For overlap, take the last few sentences based on overlap size
            words = current_chunk.split()
            overlap_words = words[-min(overlap, len(words)):]
            current_chunk = ' '.join(overlap_words) + ' ' + sentence
        else:
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Ensure chunks are not empty and not too short
    chunks = [chunk for chunk in chunks if len(chunk) > 50]  # Filter out very short chunks
    
    logger.info(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Log first few chunks for debugging
        logger.info(f"Chunk {i}: {chunk[:100]}... ({len(chunk)} chars)")
    
    return chunks