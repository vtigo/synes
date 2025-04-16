"""
Audio processor for loading and preprocessing MP3 files.
"""
import librosa
import numpy as np
import logging
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
        self.logger = logging.getLogger(__name__)
    
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
    
    def get_audio_stats(self, audio_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate basic audio statistics.
        
        Args:
            audio_data: Dictionary containing audio waveform and metadata.
            
        Returns:
            Dictionary containing basic audio statistics.
        """
        y = audio_data["waveform"]
        
        # Calculate basic statistics
        min_val = float(np.min(y))
        max_val = float(np.max(y))
        mean_val = float(np.mean(y))
        rms = float(np.sqrt(np.mean(y**2)))
        
        self.logger.info(f"Audio stats calculated - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, RMS: {rms:.4f}")
        
        return {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "rms": rms
        }