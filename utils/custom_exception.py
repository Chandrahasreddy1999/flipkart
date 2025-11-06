import sys
from utils.logger import logger

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information from the traceback.
    """
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    # Build a formatted error message
    error_message = (
        f"Error occurred in Python script [{file_name}] "
        f"at line [{line_number}] | Error message: {str(error)}"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception that provides detailed traceback info
    and logs automatically.
    """

    def __init__(self, error_message, error_detail: sys):
        # Generate the detailed message
        detailed_message = error_message_detail(error_message, error_detail)

        # Initialize base Exception class
        super().__init__(detailed_message)

        # Store for internal access
        self.error_message = detailed_message

        # ✅ Log automatically using central logger
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
