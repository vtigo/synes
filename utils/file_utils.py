"""
File utilities for handling file operations.
"""
import os
from typing import Optional


def validate_file(file_path: str, file_type: Optional[str] = None) -> bool:
    """
    Validate that a file exists and has the correct extension.
    
    Args:
        file_path: Path to the file to validate.
        file_type: Optional file extension to validate.
        
    Returns:
        Boolean indicating whether the file is valid.
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        return False
    
    # Check file type if specified
    if file_type is not None:
        _, ext = os.path.splitext(file_path)
        if ext.lower() != f".{file_type.lower()}":
            return False
    
    return True