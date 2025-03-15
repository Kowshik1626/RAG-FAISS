# RAG-Based Document Question Answering System

## Overview

This project implements a Retrieval Augmented Generation (RAG) system that allows users to upload documents, ask questions about their content, and receive AI-generated answers based on the document's information. The system uses cutting-edge AI technology from Google's Gemini models combined with efficient vector search techniques.

## Technology Stack

-   **Backend Framework**: FastAPI
-   **Language Models**:
    -   LLM: `gemini-2.0-flash-001` (for answer generation)
    -   Embedding Model: `models/documentembedding-gemini-001` (for text embedding)
-   **Vector Database**: FAISS (Facebook AI Similarity Search)
-   **Document Processing**: pdfplumber
-   **Frontend**: HTML/CSS/JavaScript

## Project Structure

```
Project/
│
├── readme.md                # To find the brief about project outline 
│
├── main.py                  # Application entry point and server configuration
│
├── models/                  # Data models directory
│   ├── __init__.py          # Makes the directory a package
│   ├── request_models.py    # Request data models (e.g., QuestionRequest)
│   └── response_models.py   # Response data models (e.g., AnswerResponse)
│
├── routes/                  # API endpoints directory
│   ├── __init__.py          # Makes the directory a package
│   ├── document.py          # Document upload and management routes
│   └── question.py          # Question answering routes
│
├── services/                # Business logic directory
│   ├── __init__.py          # Makes the directory a package
│   ├── document_service.py  # Document processing service
│   ├── embedding_service.py # Vector embedding service with Gemini
│   ├── extraction_service.py # Text extraction from documents
│   ├── rag_service.py       # RAG orchestration service
│   └── storage_service.py   # File storage service
│
├── static/                  # Frontend assets directory
│   ├── css/                 # CSS stylesheets
│   │   └── style.css        # Main stylesheet
│   ├── js/                  # JavaScript files
│   │   └── main.js          # Main frontend logic
│   └── index.html           # Main HTML interface
│
├── utils/                   # Utility functions directory
│   ├── __init__.py          # Makes the directory a package
│   └── logger.py            # Logging configuration
│
├── uploads/                 # Directory for uploaded documents
│   └── .gitkeep             # Keeps the directory in git
│
├── input_pdfs/              # Sample documents for testing
│   └── sample_document.pdf  # Example PDF document
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
├── complete_combined_code   # Here I've combined the all code chunks 
│
├── templates/               # HTML templates (if using Jinja)
│   └── index.html           # Main template
│
└── _pycache_/               # Python cache directory (auto-generated)

```

## How It Works

### 1. Document Processing Pipeline

1.  **Upload**: User uploads a document (PDF/text)
2.  **Text Extraction**: System extracts text using pdfplumber
3.  **Chunking**: Text is divided into overlapping chunks (~500 tokens each)
4.  **Embedding**: Each chunk is converted to a vector using Gemini's document embedding model
5.  **Indexing**: Vectors are stored in a FAISS index for efficient similarity search

### 2. Question Answering Process

1.  **Query Embedding**: User's question is converted to a vector using the same embedding model
2.  **Similarity Search**: System finds the top 5 most relevant chunks by calculating vector similarity
3.  **Context Assembly**: The 5 retrieved chunks are combined to form the context
4.  **Answer Generation**: Gemini LLM generates a comprehensive, structured answer based on:
    -   The user's question
    -   The retrieved context
    -   Formatting guidelines for clear presentation

### 3. Vector Similarity Matching

The core of the RAG system lies in matching the question to the most relevant document chunks:

-   **Document Chunks**: Each chunk is represented as a high-dimensional vector (embedding)
-   **Question Vector**: The question is also converted to a vector in the same space
-   **Similarity Metric**: Cosine similarity or L2 distance measures how closely the vectors align
-   **Top-K Retrieval**: The 5 chunks with the highest similarity scores are selected
-   **Multiple Perspectives**: Using 5 chunks instead of fewer provides broader context and reduces the chance of missing relevant information

## Key Features

-   **Professional Structure**: Clean separation of concerns with modular design
-   **Scalable Architecture**: Can handle documents of various sizes and formats
-   **Robust Error Handling**: Graceful fallbacks at each processing stage
-   **Enhanced Context Retrieval**: Uses 5 similar chunks for more comprehensive answers
-   **Structured Responses**: Generates well-formatted, organized answers with headings, lists, and highlighted key points

## Getting Started

1.  **Install Dependencies**:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
2.  **Install FAISS**:
    
    ```bash
    # CPU-only version
    pip install faiss-cpu
    # OR GPU version (requires CUDA)
    pip install faiss-gpu
    
    ```
    
3.  **Set Up Gemini API Key**:
    
    -   Obtain an API key from [Google AI Studio](https://ai.google.dev/)
4.  **Run the Application**:
    
    ```bash
    python main.py
    
    ```
    
5.  **Access the Interface**:
    
    -   Open a browser and navigate to `http://127.0.0.1:8000`

## Usage

1.  Enter your Gemini API key
2.  Upload a document
3.  Ask questions about the document
4.  Receive structured, comprehensive answers
5.  View the source chunks used for each answer

## Advantages of the 5-Chunk Approach

Using 5 similar chunks instead of 3 provides several benefits:

1.  **More Comprehensive Context**: Captures a wider range of relevant information
2.  **Reduced Information Gaps**: Less likely to miss important details that might be split across chunks
3.  **Better Handling of Complex Questions**: Questions that require information from multiple parts of the document are answered more accurately
4.  **Improved Cross-Reference Capability**: The model can synthesize information from different sections
5.  **Higher Confidence Answers**: More context leads to more authoritative and complete responses

This multi-chunk approach allows the system to maintain high precision while significantly improving recall and answer completeness.