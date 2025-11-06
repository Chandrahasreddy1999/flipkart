import pandas as pd
from langchain.schema import Document

from utils.custom_exception import CustomException

from utils.logger import logger
import sys


class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.info(f"Initialized DataConverter with file: {self.file_path}") 

    def convert(self):
        """
        Reads a CSV file and converts each row into a LangChain Document object.
        """
        logger.info("Starting data conversion process...") 

        try:
            df = pd.read_csv(self.file_path)[["product_title", "review"]]
            logger.info(f"Successfully read CSV file: {self.file_path} with {len(df)} rows.")

            docs = [
                Document(
                    page_content=row["review"],
                    metadata={"product_name": row["product_title"]}
                )
                for _, row in df.iterrows()
            ]

            logger.info(f"✅ Conversion complete. Total documents created: {len(docs)}")
            return docs

        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise CustomException(f"File not found: {self.file_path}", sys)

        except KeyError as e:
            logger.error(f"Missing expected columns in CSV: {e}")
            raise CustomException(f"Missing expected columns in CSV: {e}", sys)

        except Exception as e:
            raise CustomException(e, sys)
        

