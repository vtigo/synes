"""
Input processor for handling MP3 file reading and validation.
"""
import os
import logging
import librosa
import soundfile as sf
from typing import Dict, Any, Tuple, Optional

from utils.error_handler import AudioProcessingError


class InputProcessor:
    """
    Handles loading, validation, and preprocessing of MP3 files.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the input processor.
        
        Args:
            sample_rate: The sample rate to use for audio processing (default: 22050 Hz).
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def validate_file_exists(self, file_path: str) -> bool:
        """
        Validate that a file exists.
        
        Args:
            file_path: Path to the file to validate.
            
        Returns:
            Boolean indicating if the file exists.
            
        Raises:
            AudioProcessingError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise AudioProcessingError(
                f"File does not exist: {file_path}",
                "E001"
            )
        
        if not os.path.isfile(file_path):
            raise AudioProcessingError(
                f"Path is not a file: {file_path}",
                "E001"
            )
        
        return True
    
    def validate_file_extension(self, file_path: str, expected_ext: str = '.mp3') -> bool:
        """
        Validate that a file has the expected extension.
        
        Args:
            file_path: Path to the file to validate.
            expected_ext: Expected file extension (default: '.mp3').
            
        Returns:
            Boolean indicating if the file has the expected extension.
            
        Raises:
            AudioProcessingError: If the file does not have the expected extension.
        """
        _, ext = os.path.splitext(file_path)
        if ext.lower() != expected_ext.lower():
            raise AudioProcessingError(
                f"File has incorrect extension: {file_path} (expected: {expected_ext}, got: {ext})",
                "E001"
            )
        
        return True
    
    def check_mp3_integrity(self, file_path: str) -> bool:
        """
        Check the integrity of an MP3 file.
        
        Args:
            file_path: Path to the MP3 file to check.
            
        Returns:
            Boolean indicating if the file has integrity.
            
        Raises:
            AudioProcessingError: If the file lacks integrity or is corrupted.
        """
        try:
            # Try to load a small portion of the file to check integrity
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=0.1)
            
            # Check if we got any audio data
            if len(y) == 0:
                raise AudioProcessingError(
                    f"File contains no audio data: {file_path}", 
                    "E001"
                )
            
            return True
        except Exception as e:
            # Catch any librosa exceptions
            raise AudioProcessingError(
                f"Failed to load MP3 file (possibly corrupted): {str(e)}", 
                "E001"
            ) from e
    
    def check_audio_quality(self, audio_data: Dict[str, Any], min_sample_rate: int = 16000) -> bool:
        """
        Check the audio quality meets minimum requirements.
        
        Args:
            audio_data: Dictionary containing audio data.
            min_sample_rate: Minimum required sample rate (default: 16000 Hz).
            
        Returns:
            Boolean indicating if the audio quality is acceptable.
            
        Raises:
            AudioProcessingError: If the audio quality is below requirements.
        """
        sr = audio_data.get("sample_rate", 0)
        if sr < min_sample_rate:
            raise AudioProcessingError(
                f"Audio sample rate too low: {sr} Hz (minimum: {min_sample_rate} Hz)",
                "E001"
            )
        
        return True
    
    def validate_mp3_file(self, file_path: str) -> bool:
        """
        Perform comprehensive validation on an MP3 file.
        
        Args:
            file_path: Path to the MP3 file to validate.
            
        Returns:
            Boolean indicating if the file is valid.
            
        Raises:
            AudioProcessingError: If the file is invalid for any reason.
        """
        self.logger.info(f"Validating MP3 file: {file_path}")
        
        # Check file exists
        self.validate_file_exists(file_path)
        
        # Check file extension
        self.validate_file_extension(file_path, '.mp3')
        
        # Check file integrity
        self.check_mp3_integrity(file_path)
        
        self.logger.info(f"MP3 file validation successful: {file_path}")
        return True
    
    def load_mp3_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load an MP3 file and prepare it for processing.
        
        Args:
            file_path: Path to the MP3 file to load.
            
        Returns:
            Dictionary containing audio data and metadata.
            
        Raises:
            AudioProcessingError: If there are issues loading the file.
        """
        self.logger.info(f"Loading MP3 file: {file_path}")
        
        # Validate the file first
        self.validate_mp3_file(file_path)
        
        try:
            # Load the full audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Create audio data dictionary
            audio_data = {
                "waveform": y,
                "sample_rate": sr,
                "duration": duration,
                "file_path": file_path,
                "n_samples": len(y)
            }
            
            # Check audio quality
            self.check_audio_quality(audio_data)
            
            self.logger.info(f"Successfully loaded MP3 file: {file_path} "
                            f"(duration: {duration:.2f}s, sample rate: {sr} Hz)")
            
            return audio_data
        except AudioProcessingError:
            # Re-raise AudioProcessingErrors
            raise
        except Exception as e:
            # Handle other exceptions
            error_msg = f"Failed to process MP3 file: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata from an MP3 file without loading the full audio.
        
        Args:
            file_path: Path to the MP3 file.
            
        Returns:
            Dictionary containing file metadata.
            
        Raises:
            AudioProcessingError: If there are issues reading the metadata.
        """
        self.logger.info(f"Getting metadata for MP3 file: {file_path}")
        
        # Validate the file exists and has correct extension
        self.validate_file_exists(file_path)
        self.validate_file_extension(file_path, '.mp3')
        
        try:
            # Load just a tiny bit of the file to get metadata
            y, sr = librosa.load(file_path, sr=None, duration=0.1)
            
            # Get basic file information
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # Get duration (more accurate way)
            try:
                duration = librosa.get_duration(filename=file_path)
            except Exception:
                # Fallback to estimate from waveform
                duration = librosa.get_duration(y=y, sr=sr)
            
            metadata = {
                "file_path": file_path,
                "file_name": file_name,
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "sample_rate": sr,
                "duration": duration,
                "channels": 1 if y.ndim == 1 else y.shape[1]
            }
            
            self.logger.info(f"Successfully extracted metadata from {file_path}")
            return metadata
        
        except Exception as e:
            error_msg = f"Failed to extract metadata from MP3 file: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E001") from e