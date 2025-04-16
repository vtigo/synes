"""
Feature extractor for processing audio data and extracting relevant features.
"""
import logging
import librosa
import numpy as np
from typing import Dict, Any, List, Tuple

from utils.error_handler import AudioProcessingError


class FeatureExtractor:
    """
    Extracts audio features from waveform data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Optional configuration dictionary with processing parameters.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from audio data.
        
        Args:
            audio_data: Dictionary containing audio waveform and metadata.
            
        Returns:
            Dictionary containing extracted audio features.
            
        Raises:
            AudioProcessingError: If feature extraction fails.
        """
        try:
            # Extract waveform and sample rate from audio data
            y = audio_data["waveform"]
            sr = audio_data["sample_rate"]
            
            self.logger.info("Extracting audio features...")
            
            # Initialize features dictionary
            features = {}
            
            # Extract basic statistics
            features["basic_stats"] = self.extract_basic_stats(y, sr)
            
            # Extract tempo and beat information
            features["tempo"] = self.extract_tempo(y, sr)
            
            # Extract key and mode
            features["key"] = self.extract_key(y, sr)
            
            self.logger.info("Audio features extracted successfully")
            return features
            
        except Exception as e:
            error_msg = f"Feature extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_tempo(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract tempo and beat information from the audio.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing tempo and beat information.
            
        Raises:
            AudioProcessingError: If tempo extraction fails.
        """
        try:
            self.logger.info("Extracting tempo information...")
            
            # Compute onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Dynamic programming beat tracker
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Calculate beat intervals (time between beats)
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                avg_beat_interval = beat_intervals.mean()
                std_beat_interval = beat_intervals.std()
            else:
                avg_beat_interval = 0.0
                std_beat_interval = 0.0
            
            # Normalize values between 0 and 1 for consistent scaling
            # Most songs fall between 50-200 BPM, so we normalize within that range
            normalized_tempo = max(0.0, min(1.0, (tempo - 50) / 150)) if tempo > 0 else 0.0
            
            # Convert tempo to float to ensure it's not a numpy array
            tempo_float = float(tempo)
            
            # Create tempo features dictionary
            tempo_features = {
                "bpm": tempo_float,
                "normalized_tempo": float(normalized_tempo),
                "beat_count": len(beats),
                "beat_times": beat_times.tolist(),
                "avg_beat_interval": float(avg_beat_interval),
                "std_beat_interval": float(std_beat_interval),
                # Categorize tempo
                "tempo_category": self._categorize_tempo(tempo_float)
            }
            
            self.logger.info(f"Tempo extraction complete: {tempo_float:.2f} BPM")
            return tempo_features
            
        except Exception as e:
            error_msg = f"Tempo extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_key(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Determine the musical key and mode of the audio.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing key and mode information.
            
        Raises:
            AudioProcessingError: If key extraction fails.
        """
        try:
            self.logger.info("Extracting key and mode information...")
            
            # Compute the chromagram from the audio
            # A chromagram represents the intensity of the 12 different pitch classes
            # hop_length determines the frame rate of the chromagram
            hop_length = 512
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            
            # Compute key using the Krumhansl-Schmuckler key-finding algorithm
            # Sum over time to get the total intensity of each pitch class
            chroma_sum = np.sum(chroma, axis=1)
            
            # Key profiles for major and minor keys
            # These are correlation coefficients that indicate how well each pitch fits into a key
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Normalize profiles
            major_profile = major_profile / major_profile.sum()
            minor_profile = minor_profile / minor_profile.sum()
            
            # Normalize chroma
            chroma_normalized = chroma_sum / chroma_sum.sum() if chroma_sum.sum() > 0 else chroma_sum
            
            # Calculate correlation for all possible keys
            correlations = np.zeros(24)
            for i in range(12):
                # Major key correlations
                rotated_major = np.roll(major_profile, i)
                correlations[i] = np.corrcoef(chroma_normalized, rotated_major)[0, 1]
                
                # Minor key correlations
                rotated_minor = np.roll(minor_profile, i)
                correlations[i + 12] = np.corrcoef(chroma_normalized, rotated_minor)[0, 1]
            
            # Find the key with the highest correlation
            key_index = np.argmax(correlations)
            key = key_index % 12
            mode = "minor" if key_index >= 12 else "major"
            
            # Map key number to key name
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = key_names[key]
            
            # Calculate confidence as normalized correlation value
            confidence = float(max(0.0, correlations[key_index]))
            
            # Create key features dictionary
            key_features = {
                "key": key_name,
                "mode": mode,
                "confidence": confidence,
                "key_index": int(key),
                "key_mode_combined": f"{key_name} {mode}",
                "chroma_energy": chroma_normalized.tolist()
            }
            
            self.logger.info(f"Key extraction complete: {key_name} {mode}")
            return key_features
            
        except Exception as e:
            error_msg = f"Key extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_basic_stats(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate basic statistics from the audio.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing basic audio statistics.
            
        Raises:
            AudioProcessingError: If statistics calculation fails.
        """
        try:
            self.logger.info("Extracting basic audio statistics...")
            
            # Basic statistics
            duration = float(librosa.get_duration(y=y, sr=sr))
            mean_amplitude = float(np.mean(np.abs(y)))
            std_amplitude = float(np.std(y))
            max_amplitude = float(np.max(np.abs(y)))
            min_amplitude = float(np.min(np.abs(y)))
            
            # Root Mean Square (RMS) energy
            rms = float(np.sqrt(np.mean(y**2)))
            
            # Dynamic range
            dynamic_range = float(max_amplitude - min_amplitude)
            
            # Calculate silence ratio (portions where amplitude is very low)
            silence_threshold = 0.01  # Define threshold for silence
            silence_ratio = float(np.mean(np.abs(y) < silence_threshold))
            
            # Create basic stats dictionary
            basic_stats = {
                "duration": duration,
                "mean_amplitude": mean_amplitude,
                "std_amplitude": std_amplitude,
                "max_amplitude": max_amplitude,
                "min_amplitude": min_amplitude,
                "rms": rms,
                "dynamic_range": dynamic_range,
                "silence_ratio": silence_ratio
            }
            
            self.logger.info("Basic statistics extraction complete")
            return basic_stats
            
        except Exception as e:
            error_msg = f"Basic statistics calculation failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def _categorize_tempo(self, bpm: float) -> str:
        """
        Categorize tempo based on BPM value.
        
        Args:
            bpm: Tempo in beats per minute.
            
        Returns:
            String describing the tempo category.
        """
        if bpm < 60:
            return "very_slow"
        elif bpm < 90:
            return "slow"
        elif bpm < 120:
            return "moderate"
        elif bpm < 150:
            return "fast"
        else:
            return "very_fast"