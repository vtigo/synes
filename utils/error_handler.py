"""
Error handler for managing error cases.
"""
import traceback
from typing import Optional


class AudioProcessingError(Exception):
    """
    Custom exception for audio processing errors.
    """
    def __init__(self, message: str, error_code: str):
        """
        Initialize the audio processing error.
        
        Args:
            message: Error message.
            error_code: Error code (e.g., E001, E002).
        """
        self.message = message
        self.error_code = error_code
        super().__init__(f"{error_code}: {message}")


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
        if isinstance(error, AudioProcessingError):
            error_code = error.error_code
            
        print(f"Error: {str(error)}")
        print(traceback.format_exc())
        
        if error_code is not None:
            from config import ERROR_CODES
            print(f"Error code: {error_code} - {ERROR_CODES.get(error_code, 'Unknown error')}")