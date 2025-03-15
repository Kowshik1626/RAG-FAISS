import google.generativeai as genai
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from services.embedding_service import create_embeddings, create_query_embedding, cosine_similarity
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class FaissRAG:
    def __init__(self, api_key: str, chunks: List[str] = None):
        self.chunks = chunks or []
        self.embeddings = None
        self.index = None
        self.api_key = api_key
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Set up the models
        self.generation_model = "gemini-2.0-flash-001"
        self.embedding_model = "models/documentembedding-gemini-001" 
        
        # Create embeddings if chunks are provided
        if chunks:
            self.embeddings, self.index = create_embeddings(chunks, self.embedding_model, api_key)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the most relevant chunks for a given query."""
        try:
            logger.info(f"Retrieving chunks for query: '{query}'")
            
            # Get query embedding
            query_vector = create_query_embedding(query, self.embedding_model)
            if not query_vector:
                raise ValueError("Failed to create query embedding")
            
            if self.index is not None:
                # Use FAISS for retrieval
                import faiss
                query_np = np.array([query_vector]).astype('float32')
                distances, indices = self.index.search(query_np, top_k)
                
                # Get the chunks corresponding to the indices
                relevant_chunks = [self.chunks[i] for i in indices[0]]
                logger.info(f"Retrieved {len(relevant_chunks)} chunks using FAISS")
            else:
                # Fallback to manual similarity calculation
                similarities = []
                for i, embedding in enumerate(self.embeddings):
                    if embedding is not None:
                        similarity = cosine_similarity(query_vector, embedding)
                        similarities.append((i, similarity))
                
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-k chunks
                top_indices = [idx for idx, _ in similarities[:top_k]]
                relevant_chunks = [self.chunks[i] for i in top_indices]
                logger.info(f"Retrieved {len(relevant_chunks)} chunks using cosine similarity")
            
            return relevant_chunks
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            # Fallback to the first chunks
            return self.chunks[:min(top_k, len(self.chunks))]
            
    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        """Generate an answer based on the query and relevant chunks."""
        # Combine the chunks into a single context
        context = "\n\n".join(relevant_chunks)
        logger.info(f"Combined context length: {len(context)} characters")
        
        # Create prompt with the context
        prompt = f"""
        You are a helpful AI assistant that specializes in providing clear, well-structured answers about documents.

        TASK:
        Analyze the provided CONTEXT and answer the QUESTION in a structured, organized format.

        GUIDELINES:
        - Format your response with clear sections and headings where appropriate
        - Use bullet points or numbered lists for multiple points
        - Bold important information or key concepts
        - Provide concise, direct answers
        - If the answer is not in the provided context, respond with "I don't have enough information to answer that question."
        - When quoting from the document, use proper citation formatting

        CONTEXT:
        {context}

        QUESTION:
        {query}

        Please provide a comprehensive, well-structured response:
        """
        
        # Generate response
        logger.info(f"Generating response using {self.generation_model}")
        model = genai.GenerativeModel(model_name=self.generation_model)
        
        try:
            response = model.generate_content(prompt)
            logger.info("Successfully generated response")
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating answer: {str(e)}"

def create_rag_instance(api_key: str, chunks: List[str] = None) -> FaissRAG:
    """Create a new RAG instance."""
    return FaissRAG(api_key, chunks)

def process_question(rag_instance: FaissRAG, question: str) -> Tuple[str, List[str]]:
    """Process a question using the RAG instance."""
    # Retrieve relevant chunks
    relevant_chunks = rag_instance.retrieve_relevant_chunks(question)
    
    # Generate answer
    answer = rag_instance.generate_answer(question, relevant_chunks)
    
    return answer, relevant_chunks