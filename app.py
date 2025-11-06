from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, generate_latest
from RAG_chatbot.data_ingestion import DataIngestor
from RAG_chatbot.ragchain import RAGChainBuilder
from RAG_chatbot import config
from dotenv import load_dotenv
import os
import warnings

# Load .env first
load_dotenv()


# Prometheus metric
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")

# Initialize FastAPI app
app = FastAPI(title="Flipkart RAG Chatbot API")


# Template setup
templates = Jinja2Templates(directory="templates")

# Load RAG components
vector_store = DataIngestor().ingest(load_existing=True)
rag_chain = RAGChainBuilder(vector_store).build_chain()

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    REQUEST_COUNT.inc()
    return templates.TemplateResponse("index.html", {"request": request})

# Chat endpoint
@app.post("/get", response_class=PlainTextResponse)
async def get_response(msg: str = Form(...)):
    REQUEST_COUNT.inc()
    response = rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": "user-session"}}
    )["answer"]
    return response

# Prometheus metricss
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
