# Multi-Model RAG System

A sophisticated multi-modal Retrieval-Augmented Generation (RAG) system that retrieves and integrates information from both text and visual sources, providing accurate, context-aware answers along with relevant reference images from the knowledge base.

##  Features

### Comprehensive Document Processing
- **Multi-format Support**: Processes all kinds of documents including PDFs, Word documents, text files, and more
- **Image Analysis**: Automatically detects and reads images within documents
- **Table Extraction**: Identifies and extracts tabular data from documents
- **Smart Summarization**: Generates intelligent summaries of images and tables for enhanced context storage
- **Visual Context Integration**: Combines textual and visual information for comprehensive understanding

### Advanced RAG Capabilities
- **Contextual Retrieval**: Retrieves relevant information from both text and visual sources
- **Multi-modal Responses**: Provides answers that incorporate both textual information and relevant images
- **Accurate Context Matching**: Ensures responses are contextually appropriate and well-referenced
- **Knowledge Base Integration**: Maintains a structured knowledge base with text-image relationships

### User-Friendly Interface
- **Interactive Web UI**: Clean and intuitive Streamlit-based interface
- **Real-time Processing**: Fast document upload and query processing
- **Visual Feedback**: Shows relevant images alongside textual responses
- **Easy Document Management**: Simple upload and organization system

## 🚀 Installation & Setup


### Step 1: Install Dependencies
First, install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Download Project Files
Make sure you have downloaded all the necessary files from the repository:
- `FastAPI_Service.py`
- `MultiModelRAG_Service.py`
- `StreamLit_UI.py`
- `requirements.txt`
- `README.md`

### Step 3: Start the FastAPI Service
Launch the backend API service:

```bash
python FastAPI_Service.py
```
### Step 4: Launch the Streamlit Interface
In a new terminal window, start the Streamlit UI:

```bash
streamlit run StreamLit_UI.py
```
The web interface will be available at `http://localhost:8501`

## 🎥 Demo Video

A  demo video (`MultiModel_RAG_DEMO.mp4`) is included.


### 🔧 Configuration

#### API Keys Setup
This system uses **Groq API** and **Cohere API** for enhanced processing capabilities.

**Important**: If you encounter API rate limits, please add new API keys (both services offer free tiers):

1. **Groq API**: 
   - Visit [Groq Console](https://console.groq.com/)
   - Create a free account
   - Generate API key
   - Update in the configuration file

2. **Cohere API**:
   - Visit [Cohere Dashboard](https://dashboard.cohere.ai/)
   - Sign up for free account
   - Get API key
   - Update in the configuration file



