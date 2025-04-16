"""
Audio processor for loading and preprocessing MP3 files.
"""
import librosa
import numpy as np
from typing import Dict, Any


class AudioProcessor:
    """
    Handles loading and preprocessing of audio files.
    """
    
    def __init__(self, sample_rate=22050):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: The sample rate to use for audio processing.
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Load an audio file and prepare it for processing.
        
        Args:
            file_path: Path to the audio file to load.
            
        Returns:
            Dictionary containing audio data and metadata.
        """
        # Load audio file
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Return audio data and metadata
        return {
            "waveform": y,
            "sample_rate": sr,
            "duration": librosa.get_duration(y=y, sr=sr),
            "file_path": file_path
        }