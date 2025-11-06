import os
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, generate_latest

from RAG_chatbot.data_ingestion import DataIngestor
from RAG_chatbot.ragchain import RAGChainBuilder
from RAG_chatbot.config import Config
from utils.custom_exception import CustomException


from utils.logger import logger
import sys


# Initialization
warnings.filterwarnings("ignore")
load_dotenv()

app = FastAPI(title="Flipkart RAG Chatbot API")
templates = Jinja2Templates(directory="templates")

# Prometheus metric
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")

# Application Startup
try:
    logger.info("Starting FastAPI application initialization...")

    # Load vector store (either existing or new)
    vector_store = DataIngestor().ingest(load_existing=True)
    logger.info("Vector store loaded successfully.")

    # Build the RAG chain
    rag_chain = RAGChainBuilder(vector_store).build_chain()
    logger.info("RAG chain successfully built.")

except Exception as e:
    logger.critical(f"Application failed to start: {e}")
    raise CustomException(e, sys)


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the chatbot UI."""
    REQUEST_COUNT.inc()
    logger.info("Root endpoint '/' accessed.")
    return templates.TemplateResponse("index.html", {"request": request})


#  Chat endpoint
@app.post("/get", response_class=PlainTextResponse)
async def get_response(msg: str = Form(...)):
    """
    Handles chat requests from frontend and returns model response.
    """
    REQUEST_COUNT.inc()
    logger.info(f"Received user message: {msg}")

    try:
        response = rag_chain.invoke(
            {"input": msg},
            config={"configurable": {"session_id": "user-session"}}
        )["answer"]

        logger.info(f"Model response generated successfully.")
        return response

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
