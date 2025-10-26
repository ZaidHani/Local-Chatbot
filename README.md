# Local Chatbot

A local chatbot application built with Streamlit, LangChain, and Ollama, featuring document retrieval and embedding capabilities.

## Overview

This project implements a local chatbot that can answer questions based on provided documents. It uses document embeddings for efficient retrieval and local language models through Ollama for generating responses.

## Features

- Document processing and embedding generation
- Local language model integration through Ollama
- Vector-based document retrieval using FAISS
- Optional Groq integration for cloud-based language model access
- Streamlit-based user interface
- PDF document support

## Project Structure

```
Local Chatbot/
├── data/
│   └── embeddings/
│       └── index.faiss
├── src/
│   ├── app.py         # Main Streamlit application
│   ├── chat.py        # Chat models and functionality
│   └── vectorizor.py  # Document processing and embedding
└── requirements.txt
```

## Prerequisites

- Python 3.x
- Ollama installed and running locally
- (Optional) Groq API key for cloud LLM access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ZaidHani/Local-Chatbot.git
cd Local-Chatbot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
OLLAMA_BASE_URL=http://localhost:11434  # Or your Ollama server URL
GROQ_API_KEY=your_groq_api_key         # Optional, for Groq integration
```

## Usage

1. Process documents and generate embeddings:
```bash
python src/vectorizor.py
```

2. Run the Streamlit application:
```bash
streamlit run src/app.py
```

3. Access the chatbot interface in your web browser at `http://localhost:8501`

## Components

### Vectorizor (vectorizor.py)
- Handles document loading from PDF files
- Splits documents into manageable chunks
- Generates embeddings using Ollama
- Creates and manages the FAISS vector store

### Chat Models (chat.py)
- Implements chat functionality using local Ollama models
- Provides optional integration with Groq cloud models
- Includes performance timing decorator for monitoring

### Main Application (app.py)
- Implements the Streamlit user interface
- Manages the chat interaction flow
- Integrates retrieval and language model components

## Configuration

The application can be configured through environment variables and code settings:

- Document processing:
  - Chunk size: 500 characters
  - Chunk overlap: 200 characters
- Vector store: FAISS index stored in `data/embeddings/`
- Default model: gemma3:270m (can be modified in `chat.py`)

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Insert your chosen license here]