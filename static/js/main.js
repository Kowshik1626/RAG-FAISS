// DOM Elements
const apiKeyInput = document.getElementById('apiKey');
const fileInput = document.getElementById('docFile');
const uploadButton = document.getElementById('uploadButton');
const uploadStatus = document.getElementById('uploadStatus');
const debugInfo = document.getElementById('debugInfo');
const debugContent = document.getElementById('debugContent');
const questionSection = document.getElementById('questionSection');
const questionInput = document.getElementById('question');
const askButton = document.getElementById('askButton');
const questionStatus = document.getElementById('questionStatus');
const answerContainer = document.getElementById('answerContainer');
const answerElement = document.getElementById('answer');
const showChunksButton = document.getElementById('showChunksButton');
const chunksContainer = document.getElementById('chunksContainer');

// Global state
let fileId = null;

// Event Listeners
uploadButton.addEventListener('click', handleFileUpload);
askButton.addEventListener('click', handleAskQuestion);
showChunksButton.addEventListener('click', toggleChunks);

// File Upload Handler
async function handleFileUpload() {
    if (!validateUploadForm()) return;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('api_key', apiKeyInput.value);
    
    showLoading(uploadStatus);
    hideElement(debugInfo);
    
    try {
        const response = await fetch('/documents/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            handleUploadSuccess(data);
        } else {
            handleError(uploadStatus, data.detail || 'Error processing file');
        }
    } catch (error) {
        handleError(uploadStatus, 'Error: ' + error.message);
    }
}

// Question Handler
async function handleAskQuestion() {
    if (!validateQuestionForm()) return;
    
    showLoading(questionStatus);
    hideElement(answerContainer);
    
    try {
        const response = await fetch(`/questions/${fileId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: questionInput.value })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            handleQuestionSuccess(data);
        } else {
            handleError(questionStatus, data.detail || 'Error generating answer');
        }
    } catch (error) {
        handleError(questionStatus, 'Error: ' + error.message);
    }
}

// Form Validation
function validateUploadForm() {
    if (!apiKeyInput.value) {
        handleError(uploadStatus, 'Please enter your Google API Key');
        return false;
    }
    
    if (!fileInput.files.length) {
        handleError(uploadStatus, 'Please select a file');
        return false;
    }
    
    return true;
}

function validateQuestionForm() {
    if (!fileId) {
        alert('Please upload a file first');
        return false;
    }
    
    if (!questionInput.value) {
        handleError(questionStatus, 'Please enter a question');
        return false;
    }
    
    return true;
}

// Success Handlers
function handleUploadSuccess(data) {
    fileId = data.file_id;
    uploadStatus.textContent = `Success! Processed ${data.num_chunks} chunks`;
    showElement(questionSection);
    
    // Show debug info if available
    if (data.debug_info) {
        debugContent.innerHTML = `
            <p><strong>File:</strong> ${data.filename}</p>
            <p><strong>Chunks:</strong> ${data.num_chunks}</p>
            <p><strong>Text Sample:</strong> ${data.text_sample || 'N/A'}</p>
            <p><strong>Vector DB:</strong> FAISS index created</p>
        `;
        showElement(debugInfo);
    }
}

function handleQuestionSuccess(data) {
    clearElement(questionStatus);
    answerElement.textContent = data.answer;
    
    // Display chunks
    displayChunks(data.chunks);
    
    showElement(answerContainer);
}

// Chunk Display
function displayChunks(chunks) {
    clearElement(chunksContainer);
    
    chunks.forEach((chunk, i) => {
        const chunkEl = document.createElement('div');
        chunkEl.className = 'chunk';
        chunkEl.innerHTML = `<strong>Chunk ${i+1}</strong><pre>${chunk}</pre>`;
        chunksContainer.appendChild(chunkEl);
    });
}

// Toggle chunks visibility
function toggleChunks() {
    if (chunksContainer.classList.contains('hidden')) {
        showElement(chunksContainer);
        showChunksButton.textContent = 'Hide Source Chunks';
    } else {
        hideElement(chunksContainer);
        showChunksButton.textContent = 'Show Source Chunks';
    }
}

// UI Helpers
function showLoading(element) {
    element.innerHTML = '<div class="loading"></div> Processing...';
}

function handleError(element, message) {
    element.textContent = message;
}

function clearElement(element) {
    element.textContent = '';
}

function hideElement(element) {
    element.classList.add('hidden');
}

function showElement(element) {
    element.classList.remove('hidden');
}