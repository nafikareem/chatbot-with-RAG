# Knowledge-Based Chatbot with RAG

## Overview
This project is a **Knowledge-Based Chatbot** built using Retrieval-Augmented Generation (RAG) techniques. It allows users to upload PDF documents, input web URLs, or provide raw text as knowledge sources, and then ask questions based on the provided content. The chatbot leverages the `google/flan-t5-base` model for text generation and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for embedding documents into a vector store. The user interface is developed using Streamlit, with a streaming output effect to enhance user experience.

### RAG Pipeline

![alt text](img/Pipeline.drawio.png)


### Features
- **Multi-Source Input**: Supports PDF files, web URLs, and raw text as knowledge sources.
- **RAG Pipeline**: Combines retrieval (using FAISS vector store) and generation (using `flan-t5-base`) for answering questions.
- **Reranking**: Implements a simple cosine similarity-based reranking to improve document relevance.
- **Streaming Output**: Answers are displayed with a typing effect for better user engagement.
- **Chat History**: Keeps track of questions and answers during the session.
- **Streamlit UI**: Intuitive interface for document processing and question answering.

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- NVIDIA GPU (optional but recommended for faster inference with `flan-t5-base`)

### Setup
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd project_chatbot
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have the required packages listed in `requirements.txt`:
   ```
   langchain-community
   transformers==4.52.3
   sentence-transformers==2.7.0
   faiss-cpu==1.11.0
   torch==2.7.0+cu128
   streamlit
   ```
   
   Install them using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application
1. **Start the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   This will launch the application in your default web browser at `http://localhost:8501`.

2. **Process Documents**:
   - **Upload PDF**: Use the file uploader in the sidebar to upload a PDF document.
   - **Enter URL**: Input a valid URL (e.g., `https://example.com`) to scrape web content.
   - **Enter Raw Text**: Type or paste text directly into the text area.
   - Click **"Process Documents"** to load and process the documents into the vector store.

3. **Ask Questions**:
   - Type your question in the "Ask a Question" input field (e.g., "What is this about?").
   - Click **"Get Answer"** to receive a response.
   - The answer will appear with a streaming effect, simulating a typing experience.

4. **View Chat History**:
   - Previous questions and answers are displayed below the chat interface for reference.

### Example
- **Input Document**: A PDF containing "This document explains the benefits of RAG in chatbot systems."
- **Question**: "What is the document about?"
- **Output**: "The document explains the benefits of RAG in chatbot systems." (appears with streaming effect)

## Project Structure
```
project_chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # List of dependencies
├── data/                # Directory for sample documents
│   └── sample.txt       # Sample text file
├── src/                 # Source code for the RAG pipeline
│   ├── document_loader.py  # Handles loading of PDF, URL, and text
│   ├── embedding.py        # Manages document embedding and reranking
│   └── rag_pipeline.py     # Core RAG pipeline for retrieval and generation
├── tests/               # Test scripts
│   ├── test_model.py
│   └── test_pipeline.py
└── README.md            # Project documentation
```

## Dependencies
- `langchain-community`: For document loading, embedding, and RAG pipeline.
- `transformers`: For the `google/flan-t5-base` model.
- `sentence-transformers`: For embedding documents with `paraphrase-multilingual-MiniLM-L12-v2`.
- `faiss-cpu`: For vector store implementation.
- `torch`: For GPU/CPU inference.
- `streamlit`: For the web interface.

## Limitations
- **Reranking**: Currently uses a simple cosine similarity-based reranking due to dependency conflicts with `ragatouille` (ColBERT). Advanced reranking can be implemented once conflicts are resolved.
- **VRAM**: The `flan-t5-base` model requires ~2-3GB VRAM. If running on a GPU with limited memory (e.g., GTX 1650Ti with 4GB VRAM), use `device_map="auto"` in `rag_pipeline.py`.
- **Streaming Effect**: Simulated using `time.sleep`, which may vary in speed depending on server/client performance.

## Deployment on Hugging Face Spaces
1. Create a new Space on Hugging Face and select the Streamlit template.
2. Upload the project files (`app.py`, `src/`, `data/`, `requirements.txt`).
3. Ensure `requirements.txt` is up-to-date.
4. Activate the Space and access the provided URL.

## Future Improvements
- Implement advanced reranking with ColBERT once dependency issues are resolved.
- Add support for more file formats (e.g., DOCX, CSV).
- Enhance streaming with true token-by-token generation (requires model-level streaming support).
- Add evaluation metrics for answer quality (e.g., BLEU, ROUGE scores).
- Support for multilingual queries (e.g., Bahasa Indonesia) with prompt engineering.

## Contributing
Feel free to fork the repository, make improvements, and submit pull requests. Issues and feature requests can be submitted via GitHub.


## Acknowledgments
- Built with the help of LangChain, Hugging Face Transformers, and Streamlit communities.