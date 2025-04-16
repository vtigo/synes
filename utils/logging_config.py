"""
Logging configuration for the Music-to-Visual Emotion Interpreter System.
"""
import logging
import os
from datetime import datetime


class LoggingConfig:
    """
    Configuration for logging system.
    """
    
    def __init__(self, log_level=logging.INFO, log_file=None):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level (default: INFO).
            log_file: Path to log file (default: logs/YYYY-MM-DD-HH-MM-SS.log).
        """
        self.log_level = log_level
        
        # Create logs directory if it doesn't exist
        if log_file is None:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file = os.path.join("logs", f"{timestamp}.log")
            
        self.log_file = log_file
        
        # Configure logging
        self._configure_logging()
        
    def _configure_logging(self):
        """Configure logging with file and console handlers."""
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logging.info(f"Logging configured. Log file: {self.log_file}")
    
    @staticmethod
    def get_logger(name=None):
        """
        Get a named logger.
        
        Args:
            name: Name for the logger (default: None - root logger).
            
        Returns:
            Logger instance.
        """
        return logging.getLogger(name)


# Initialize default logger
def initialize_logging(log_level=logging.INFO, log_file=None):
    """
    Initialize logging configuration.
    
    Args:
        log_level: Logging level (default: INFO).
        log_file: Path to log file (default: logs/YYYY-MM-DD-HH-MM-SS.log).
        
    Returns:
        LoggingConfig instance.
    """
    return LoggingConfig(log_level=log_level, log_file=log_file)