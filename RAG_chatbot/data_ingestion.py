import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_astradb")
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_chatbot.data_converter import DataConverter
from RAG_chatbot.config import Config

class DataIngestor:
    def __init__(self):
        self.embedding=HuggingFaceEmbeddings(model=Config.embeddings)
        self.vstore=AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="Flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self,load_existing=True):
        if load_existing==True:
            return self.vstore
        docs = DataConverter("D:/Projects/LLM_GenAI/Fastapi/data/flipkart_product_review.csv").convert()

        self.vstore.add_documents(docs)
        return self.vstore

