import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_astradb")

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_chatbot.data_converter import DataConverter
from RAG_chatbot.config import Config
from utils.custom_exception import CustomException


from utils.logger import logger
import sys


class DataIngestor:
    def __init__(self):
        """
        Initializes embeddings and AstraDB vector store connection.
        """
        logger.info("Initializing DataIngestor...")  
        try:
            self.embedding = HuggingFaceEmbeddings(model=Config.embeddings)
            logger.info(f"Using embedding model: {Config.embeddings}")

            self.vstore = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name="Flipkart_database",
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
                token=Config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=Config.ASTRA_DB_KEYSPACE
            )
            logger.info("AstraDB Vector Store initialized successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def ingest(self, load_existing: bool = False):
        """
        Loads documents into AstraDB, or returns the existing store.
        """
        logger.info("Starting data ingestion process")  
        try:
            if load_existing:
                logger.info("Returning existing AstraDB vector store instance.")
                return self.vstore

            data_path = Config.DATA_PATH
            logger.info(f"Loading data from: {data_path}")

            docs = DataConverter(data_path).convert()
            logger.info(f"Documents loaded: {len(docs)}")

            self.vstore.add_documents(docs)
            logger.info("Documents successfully added to AstraDB Vector Store.")

            return self.vstore

        except FileNotFoundError:
            logger.error(f"Data file not found at path: {data_path}")
            raise CustomException(f"Data file not found at path: {data_path}", sys)

        except Exception as e:
            raise CustomException(e, sys)



