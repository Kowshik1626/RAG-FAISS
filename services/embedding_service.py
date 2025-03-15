import google.generativeai as genai
import numpy as np
from typing import List, Optional, Tuple
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_embeddings(chunks: List[str], embedding_model: str, api_key: str) -> Tuple[List, Optional[object]]:
    """Create embeddings and FAISS index for the chunks."""
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    try:
        # Try to import FAISS
        import faiss
        logger.info("Successfully imported FAISS")
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=chunk,
                    task_type="retrieval_query"
                )
                embeddings.append(embedding["embedding"])
                logger.info(f"Created embedding for chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error creating embedding for chunk {i+1}: {str(e)}")
                # Add a placeholder (zeros) embedding
                if embeddings:
                    # Use same dimension as previous embeddings
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    # If this is the first embedding, use a default dimension
                    embeddings.append([0.0] * 768)  # Common embedding dimension
        
        if embeddings:
            # Convert to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            
            logger.info(f"Created FAISS index with dimension {dimension}")
            return embeddings, index
        else:
            logger.error("No valid embeddings were created")
            return [], None
            
    except ImportError:
        logger.warning("FAISS is not installed, falling back to in-memory embeddings")
        # Store embeddings without FAISS
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=chunk,
                    task_type="retrieval_query"
                )
                embeddings.append(embedding["embedding"])
                logger.info(f"Created embedding for chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error creating embedding for chunk {i+1}: {str(e)}")
                # Add None for failed embeddings
                embeddings.append(None)
        
        return embeddings, None

def create_query_embedding(query: str, embedding_model: str) -> List[float]:
    """Create an embedding for a query."""
    try:
        embedding = genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return embedding["embedding"]
    except Exception as e:
        logger.error(f"Error creating query embedding: {str(e)}")
        return None

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a*b for a, b in zip(v1, v2))
    magnitude1 = sum(a*a for a in v1) ** 0.5
    magnitude2 = sum(b*b for b in v2) ** 0.5
    
    if magnitude1 * magnitude2 == 0:
        return 0
        
    return dot_product / (magnitude1 * magnitude2)