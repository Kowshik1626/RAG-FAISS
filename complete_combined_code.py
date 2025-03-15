from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
import os
import google.generativeai as genai
import uuid
from pydantic import BaseModel
import shutil
from pathlib import Path
import re
import io
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
Path("uploads").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Document RAG API", description="API for answering questions about documents using RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models for API requests and responses
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    chunks: List[str]

# In-memory storage for RAG instances
rag_instances = {}

def extract_text_from_pdf(file_path):
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

def chunk_text(text, chunk_size=500, overlap=100):
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

class FaissRAG:
    def __init__(self, api_key: str):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.api_key = api_key
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Set up the models
        self.generation_model = "gemini-2.0-flash-001"
        self.embedding_model = "models/documentembedding-gemini-001" 
        
    def initialize_with_file(self, file_path: str) -> None:
        """Initialize the RAG system with a document file."""
        # Extract text
        text = extract_text_from_pdf(file_path)
        logger.info(f"Extracted text length: {len(text)}")
        
        if len(text.strip()) < 100:
            logger.warning(f"Extracted text seems too short: '{text[:100]}'")
        
        # Chunk the text
        self.chunks = chunk_text(text)
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Create FAISS index with embeddings
        self._create_embeddings()
        
        return len(self.chunks)
    
    def _create_embeddings(self):
        """Create embeddings and build FAISS index."""
        try:
            import faiss
            logger.info("Successfully imported FAISS")
            
            logger.info(f"Creating embeddings for {len(self.chunks)} chunks")
            embeddings = []
            
            for i, chunk in enumerate(self.chunks):
                try:
                    embedding = genai.embed_content(
                        model=self.embedding_model,
                        content=chunk,
                        task_type="retrieval_query"
                    )
                    embeddings.append(embedding["embedding"])
                    logger.info(f"Created embedding for chunk {i+1}/{len(self.chunks)}")
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
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings_np)
                self.embeddings = embeddings_np
                
                logger.info(f"Created FAISS index with dimension {dimension}")
            else:
                logger.error("No valid embeddings were created")
        except ImportError:
            logger.error("FAISS is not installed, falling back to in-memory embeddings")
            # Store embeddings without FAISS
            self.embeddings = []
            
            for i, chunk in enumerate(self.chunks):
                try:
                    embedding = genai.embed_content(
                        model=self.embedding_model,
                        content=chunk,
                        task_type="retrieval_query"
                    )
                    self.embeddings.append(embedding["embedding"])
                    logger.info(f"Created embedding for chunk {i+1}/{len(self.chunks)}")
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk {i+1}: {str(e)}")
                    # Add None for failed embeddings
                    self.embeddings.append(None)
        except Exception as e:
            logger.error(f"Error in _create_embeddings: {str(e)}")
    
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a*b for a, b in zip(v1, v2))
        magnitude1 = sum(a*a for a in v1) ** 0.5
        magnitude2 = sum(b*b for b in v2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the most relevant chunks for a given query."""
        try:
            logger.info(f"Retrieving chunks for query: '{query}'")
            
            # Get query embedding
            query_embedding = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_vector = query_embedding["embedding"]
            
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
                        similarity = self._cosine_similarity(query_vector, embedding)
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
    
    def process_question(self, question: str) -> tuple:
        """Process a question and return an answer."""
        # Retrieve relevant chunks
        logger.info(f"Processing question: '{question}'")
        relevant_chunks = self.retrieve_relevant_chunks(question)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        return answer, relevant_chunks

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Return the HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document RAG System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            button {
                padding: 10px 15px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            input, textarea {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }
            .hidden {
                display: none;
            }
            .chunks {
                margin-top: 20px;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            .chunk {
                margin-bottom: 15px;
                padding: 10px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0,0,0,.3);
                border-radius: 50%;
                border-top-color: #3498db;
                animation: spin 1s ease-in-out infinite;
                margin-left: 10px;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .toggle-btn {
                margin-top: 15px;
                background-color: #95a5a6;
            }
            .toggle-btn:hover {
                background-color: #7f8c8d;
            }
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .answer-content {
                background-color: #e8f4fc;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                border-left: 4px solid #3498db;
            }
            .info-box {
                padding: 10px 15px;
                background-color: #d5f5e3;
                border-left: 4px solid #2ecc71;
                margin: 10px 0;
                border-radius: 5px;
            }
            .debug-info {
                background-color: #fadbd8;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <h1>Document RAG System with FAISS</h1>
        <div class="info-box">
            Upload a document (PDF or text) and ask questions about it. This system uses FAISS for vector storage and Google's Gemini API for retrieving information and generating answers.
        </div>
        
        <div class="section" id="uploadSection">
            <h2>Step 1: Upload Document</h2>
            <div>
                <label for="apiKey">Google API Key:</label>
                <input type="password" id="apiKey" placeholder="Enter your Google Gemini API Key">
            </div>
            <div>
                <label for="docFile">Document File (PDF or Text):</label>
                <input type="file" id="docFile">
            </div>
            <div>
                <button id="uploadButton">Upload and Process</button>
                <span id="uploadStatus"></span>
            </div>
            <div id="debugInfo" class="debug-info hidden">
                <h4>Document Processing Info:</h4>
                <div id="debugContent"></div>
            </div>
        </div>
        
        <div class="section hidden" id="questionSection">
            <h2>Step 2: Ask Questions</h2>
            <div>
                <label for="question">Your Question:</label>
                <textarea id="question" rows="3" placeholder="Enter your question about the document"></textarea>
            </div>
            <div>
                <button id="askButton">Ask</button>
                <span id="questionStatus"></span>
            </div>
            
            <div id="answerContainer" class="hidden">
                <h3>Answer:</h3>
                <div id="answer" class="answer-content"></div>
                
                <div id="chunksToggle">
                    <button id="showChunksButton" class="toggle-btn">Show Source Chunks</button>
                </div>
                <div id="chunksContainer" class="chunks hidden"></div>
            </div>
        </div>
        
        <script>
            let fileId = null;
            
            document.getElementById('uploadButton').addEventListener('click', async () => {
                const apiKey = document.getElementById('apiKey').value;
                const fileInput = document.getElementById('docFile');
                const statusEl = document.getElementById('uploadStatus');
                const debugEl = document.getElementById('debugInfo');
                const debugContentEl = document.getElementById('debugContent');
                
                if (!apiKey) {
                    statusEl.textContent = 'Please enter your Google API Key';
                    return;
                }
                
                if (!fileInput.files.length) {
                    statusEl.textContent = 'Please select a file';
                    return;
                }
                
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);
                formData.append('api_key', apiKey);
                
                statusEl.innerHTML = '<div class="loading"></div> Processing...';
                debugEl.classList.add('hidden');
                
                try {
                    const response = await fetch('/upload-file', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        fileId = data.file_id;
                        statusEl.textContent = `Success! Processed ${data.num_chunks} chunks`;
                        document.getElementById('questionSection').classList.remove('hidden');
                        
                        // Show debug info
                        if (data.debug_info) {
                            debugContentEl.innerHTML = `
                                <p><strong>File:</strong> ${data.filename}</p>
                                <p><strong>Chunks:</strong> ${data.num_chunks}</p>
                                <p><strong>Text Sample:</strong> ${data.text_sample || 'N/A'}</p>
                                <p><strong>Vector DB:</strong> FAISS index created</p>
                            `;
                            debugEl.classList.remove('hidden');
                        }
                    } else {
                        statusEl.textContent = data.detail || 'Error processing file';
                    }
                } catch (error) {
                    statusEl.textContent = 'Error: ' + error.message;
                }
            });
            
            document.getElementById('askButton').addEventListener('click', async () => {
                if (!fileId) {
                    alert('Please upload a file first');
                    return;
                }
                
                const question = document.getElementById('question').value;
                const statusEl = document.getElementById('questionStatus');
                
                if (!question) {
                    statusEl.textContent = 'Please enter a question';
                    return;
                }
                
                statusEl.innerHTML = '<div class="loading"></div> Generating answer...';
                document.getElementById('answerContainer').classList.add('hidden');
                
                try {
                    const response = await fetch(`/ask-question/${fileId}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        statusEl.textContent = '';
                        document.getElementById('answer').textContent = data.answer;
                        
                        // Display chunks
                        const chunksContainer = document.getElementById('chunksContainer');
                        chunksContainer.innerHTML = '';
                        
                        data.chunks.forEach((chunk, i) => {
                            const chunkEl = document.createElement('div');
                            chunkEl.className = 'chunk';
                            chunkEl.innerHTML = `<strong>Chunk ${i+1}</strong><pre>${chunk}</pre>`;
                            chunksContainer.appendChild(chunkEl);
                        });
                        
                        document.getElementById('answerContainer').classList.remove('hidden');
                    } else {
                        statusEl.textContent = data.detail || 'Error generating answer';
                    }
                } catch (error) {
                    statusEl.textContent = 'Error: ' + error.message;
                }
            });
            
            // Toggle chunks visibility
            document.getElementById('showChunksButton').addEventListener('click', () => {
                const chunksContainer = document.getElementById('chunksContainer');
                const button = document.getElementById('showChunksButton');
                
                if (chunksContainer.classList.contains('hidden')) {
                    chunksContainer.classList.remove('hidden');
                    button.textContent = 'Hide Source Chunks';
                } else {
                    chunksContainer.classList.add('hidden');
                    button.textContent = 'Show Source Chunks';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):
    """Upload a file and initialize the RAG system."""
    # Generate a unique ID for this file
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = f"uploads/{file_id}{file_ext}"
    
    logger.info(f"Received file: {file.filename} (ID: {file_id}) with extension {file_ext}")
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {file_path}")
        
        # Initialize RAG system
        rag = FaissRAG(api_key=api_key)
        num_chunks = rag.initialize_with_file(file_path)
        
        # Get sample text for debugging
        text_sample = rag.chunks[0][:200] + "..." if rag.chunks else "No text extracted"
        
        # Store RAG instance in memory
        rag_instances[file_id] = rag
        
        logger.info(f"RAG system initialized with {num_chunks} chunks")
        
        return JSONResponse(
            content={
                "file_id": file_id,
                "filename": file.filename,
                "num_chunks": num_chunks,
                "message": "File processed successfully",
                "debug_info": True,
                "text_sample": text_sample
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/ask-question/{file_id}", response_model=AnswerResponse)
async def ask_question(
    file_id: str,
    request: QuestionRequest
):
    """Ask a question about the uploaded file."""
    logger.info(f"Question received for file {file_id}: '{request.question}'")
    
    if file_id not in rag_instances:
        logger.error(f"File ID {file_id} not found")
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    
    rag = rag_instances[file_id]
    
    try:
        answer, chunks = rag.process_question(request.question)
        logger.info(f"Generated answer of length {len(answer)}")
        return AnswerResponse(answer=answer, chunks=chunks)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.delete("/delete/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and its RAG instance."""
    logger.info(f"Delete request received for file {file_id}")
    
    if file_id not in rag_instances:
        logger.error(f"File ID {file_id} not found for deletion")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Find file path (account for different file extensions)
    file_path = None
    for ext in [".pdf", ".txt", ".docx", ""]:
        temp_path = f"uploads/{file_id}{ext}"
        if os.path.exists(temp_path):
            file_path = temp_path
            break
    
    # Remove from memory
    if file_id in rag_instances:
        del rag_instances[file_id]
    
    # Remove file if it exists
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted file at {file_path}")
    
    return {"message": "File and RAG instance deleted successfully"}

@app.get("/status")
async def get_status():
    """Get the status of the API."""
    return {
        "status": "running", 
        "active_documents": len(rag_instances)
    }

if __name__ == "__main__":
    uvicorn.run("faiss_rag:app", host="127.0.0.1", port=8000, reload=True)