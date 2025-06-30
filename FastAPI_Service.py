from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uvicorn
import shutil
from typing import Optional, List
import logging

from MultiModelRAG_Service import MultimodalRAGProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Modal RAG API",
    description="API for processing multiple document types and answering questions",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = None

class QueryRequest(BaseModel):
    question: str
    return_sources: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[dict] = None
    images_count: int = 0

class ProcessResponse(BaseModel):
    message: str
    texts: int
    tables: int
    images: int
    filename: str

class HealthResponse(BaseModel):
    status: str
    message: str

def initialize_processor():
    global processor
    try:
        # Get API keys from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not groq_api_key or not cohere_api_key:
            logger.warning("API keys not found in environment variables")

        processor = MultimodalRAGProcessor(
            groq_api_key=groq_api_key,
            cohere_api_key=cohere_api_key
        )
        logger.info("RAG processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize the processor on startup."""
    initialize_processor()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Enhanced Multi-Modal RAG API is running"
    )


@app.post("/upload", response_model=ProcessResponse)
async def upload_document(file: UploadFile = File(...)):

    global processor

    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")

    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = os.path.splitext(filename)[1].lower()
    supported_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.csv'}

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}"
        )

    uploads_dir = "INPUT_DATA"
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing uploaded file: {filename}")

        processor.clear_vectorstore()

        result = processor.process_document(file_path)

        logger.info(f"Successfully processed {filename}")

        return ProcessResponse(
            message=f"Successfully processed {filename}",
            texts=result["texts"],
            tables=result["tables"],
            images=result["images"],
            filename=filename
        )

    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

        logger.error(f"Error processing {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):

    global processor

    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")

    try:
        logger.info(f"Processing query: {request.question}")

        response = processor.query(request.question, return_sources=request.return_sources)

        if request.return_sources:
            images_count = len(response.get('context', {}).get('images', []))
            return QueryResponse(
                answer=response['response'],
                sources=response.get('context'),
                images_count=images_count
            )
        else:
            return QueryResponse(
                answer=response,
                images_count=0
            )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    return {
        "supported_formats": [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".csv"],
        "descriptions": {
            ".pdf": "Portable Document Format",
            ".docx": "Microsoft Word Document",
            ".pptx": "Microsoft PowerPoint Presentation",
            ".xlsx": "Microsoft Excel Spreadsheet",
            ".txt": "Plain Text File",
            ".csv": "Comma-Separated Values"
        }
    }

@app.delete("/clear")
async def clear_vectorstore():
    global processor

    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")

    try:
        processor.clear_vectorstore()
        logger.info("Vectorstore cleared successfully")
        return {"message": "Vectorstore cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing vectorstore: {str(e)}")


if __name__ == "__main__":


    ## If the given API key show Limit error please login to groq and get new API key.

    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "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"
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "xgYlYbeVQZTTn4DGjCWNOQKFH15NL4TrVh1HNoT0"

    uvicorn.run(app, host="0.0.0.0", port=5000)