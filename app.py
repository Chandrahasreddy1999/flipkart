import os
import warnings
import sys
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, generate_latest

from RAG_chatbot.data_ingestion import DataIngestor
from RAG_chatbot.ragchain import RAGChainBuilder
from RAG_chatbot.config import Config
from utils.custom_exception import CustomException
from utils.logger import logger

# Initialization
warnings.filterwarnings("ignore")
load_dotenv()

# Global variable to store the chain
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle application startup and shutdown.
    Loads the vector store and builds the RAG chain on startup.
    """
    global rag_chain
    try:
        logger.info("Starting FastAPI application initialization...")

        # Load vector store (either existing or new)
        # Note: If DataIngestor().ingest is blocking/heavy, consider running it in a thread pool
        # if it takes too long, but for startup it's usually acceptable.
        vector_store = DataIngestor().ingest(load_existing=True)
        logger.info("Vector store loaded successfully.")

        # Build the RAG chain
        rag_chain = RAGChainBuilder(vector_store).build_chain()
        logger.info("RAG chain successfully built.")
        
        yield
        
        # Cleanup code (if any) goes here
        logger.info("Shutting down application...")
        
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        raise CustomException(e, sys)

app = FastAPI(title="Flipkart RAG Chatbot API", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Prometheus metric
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the chatbot UI and sets a session cookie if missing."""
    REQUEST_COUNT.inc()
    logger.info("Root endpoint '/' accessed.")
    
    response = templates.TemplateResponse("index.html", {"request": request})
    
    # Check if session_id cookie exists, if not, create one
    if not request.cookies.get("session_id"):
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
        logger.info(f"New session started with ID: {session_id}")
    
    return response


#  Chat endpoint
@app.post("/get", response_class=PlainTextResponse)
async def get_response(request: Request, msg: str = Form(...)):
    """
    Handles chat requests from frontend and returns model response.
    Uses async execution to prevent blocking the event loop.
    """
    REQUEST_COUNT.inc()
    
    # Retrieve session_id from cookie, fallback to a default if missing (e.g. direct API call)
    session_id = request.cookies.get("session_id", "default-session")
    logger.info(f"Received user message: {msg} | Session ID: {session_id}")

    try:
        if rag_chain is None:
            raise Exception("RAG Chain not initialized")

        # Use ainvoke for asynchronous execution
        result = await rag_chain.ainvoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )
        
        response_text = result["answer"]

        logger.info(f"Model response generated successfully.")
        return response_text

    except Exception as e:
        logger.error(f"Error in /get endpoint: {e}")
        raise CustomException(e, sys)


# Prometheus metrics
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Exposes Prometheus metrics."""
    REQUEST_COUNT.inc()
    logger.info("Metrics endpoint '/metrics' accessed.")
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")


#  Global Exception Handling
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    logger.error(f"CustomException: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )


#  Run Locally
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on port 5000...")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
