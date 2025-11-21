import os
import sys
from dotenv import load_dotenv

from utils.custom_exception import CustomException

from utils.logger import logger



# Load environment variables

load_dotenv()


class Config:
  
    try:
        logger.info("Loading environment variables from .env file...")  

        ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_ID")
        ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
        LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
        HF_TOKEN = os.getenv("HF_TOKEN")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

        embeddings = "sentence-transformers/all-MiniLM-L6-v2"
        RAGMODEL = "qwen/qwen2.5-coder-32b-instruct"
        DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "flipkart_product_review.csv")

        logger.info("Environment variables loaded successfully.") 

    except Exception as e:
        raise CustomException(e, sys)
