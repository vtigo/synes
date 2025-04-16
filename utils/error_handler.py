"""
Error handler for managing error cases.
"""
import traceback
from typing import Optional


class ErrorHandler:
    """
    Handles errors that occur during processing.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        pass
    
    def handle_error(self, error: Exception, error_code: Optional[str] = None) -> None:
        """
        Handle an error.
        
        Args:
            error: Exception that occurred.
            error_code: Optional error code to use.
        """
        print(f"Error: {str(error)}")
        print(traceback.format_exc())
        
        if error_code is not None:
            from config import ERROR_CODES
            print(f"Error code: {error_code} - {ERROR_CODES.get(error_code, 'Unknown error')}")