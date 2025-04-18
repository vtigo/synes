"""
Feature extractor for processing audio data and extracting relevant features.
"""
import logging
import librosa
import numpy as np
from typing import Dict, Any, List, Tuple

from utils.error_handler import AudioProcessingError


"""
Feature extractor for processing audio data and extracting relevant features.
"""
import logging
import librosa
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

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
        
        # Default parameters if not provided in config
        self.n_mfcc = self.config.get("n_mfcc", 13)
        self.hop_length = self.config.get("hop_length", 512)
        self.n_fft = self.config.get("n_fft", 2048)
    
    def extract_features(self, audio_data: Dict[str, Any], include_temporal: bool = False) -> Dict[str, Any]:
        """
        Extract features from audio data.
        
        Args:
            audio_data: Dictionary containing audio waveform and metadata.
            include_temporal: Whether to include temporal features (default: False).
                Setting this to True will increase processing time but provide more detailed analysis.
                
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
            
            if include_temporal:
                # Generate comprehensive fingerprint with temporal features
                features = self.generate_audio_fingerprint(y, sr)
                self.logger.info("Comprehensive audio fingerprint extracted with temporal features")
            else:
                # Initialize features dictionary for basic extraction
                features = {}
                
                # Extract basic statistics
                features["basic_stats"] = self.extract_basic_stats(y, sr)
                
                # Extract tempo and beat information
                features["tempo"] = self.extract_tempo(y, sr)
                
                # Extract key and mode
                features["key"] = self.extract_key(y, sr)
                
                # Extract spectral features
                features["spectral"] = self.extract_spectral_features(y, sr)
                
                # Extract MFCCs for timbre analysis
                features["mfccs"] = self.extract_mfccs(y, sr)
                
                # Extract chroma features for harmonic content
                features["chroma"] = self.extract_chroma_features(y, sr)
                
                # Extract RMS energy for dynamics
                features["rms_energy"] = self.extract_rms_energy(y, sr)
                
                # Extract zero crossing rate for texture
                features["zero_crossing_rate"] = self.extract_zero_crossing_rate(y, sr)
                
                # Extract spectral contrast for instrumentation
                features["spectral_contrast"] = self.extract_spectral_contrast(y, sr)
                
                # Normalize all features to ensure consistency
                features = self.normalize_features(features)
                
                self.logger.info("Basic audio features extracted successfully")
            
            return features
                
        except Exception as e:
            error_msg = f"Feature extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e    
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral features from the audio.
        
        Spectral features describe the frequency content of the audio signal:
        - Spectral centroid: Represents the "center of mass" of the spectrum, 
          indicating the brightness of the sound
        - Spectral rolloff: Frequency below which a specific percentage of the 
          spectral energy is contained, indicating the skew of frequencies
        - Spectral flux: Measures how quickly the spectrum changes, indicating 
          the amount of variation in the timbre over time
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing spectral features.
            
        Raises:
            AudioProcessingError: If spectral feature extraction fails.
        """
        try:
            self.logger.info("Extracting spectral features...")
            
            # Spectral centroid - Represents the "center of mass" of the spectrum (brightness)
            # Higher values indicate brighter sounds with more high frequencies
            centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Spectral rolloff - Frequency below which 85% of the spectral energy is contained
            rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Compute spectrogram
            D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            
            # Spectral flux - How quickly the spectrum changes (timbral variations)
            # We calculate flux as the sum of squared differences between adjacent frames
            flux = np.sum(np.diff(D, axis=1)**2, axis=0)
            if len(flux) > 0:  # Add a 0 for the first frame to maintain dimensions
                flux = np.concatenate(([0], flux))
            
            # Bandwidth - Width of the frequency band
            bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Flatness - How noise-like the sound is
            flatness = librosa.feature.spectral_flatness(
                y=y, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Calculate means and standard deviations
            mean_centroid = float(np.mean(centroid))
            std_centroid = float(np.std(centroid))
            mean_rolloff = float(np.mean(rolloff))
            std_rolloff = float(np.std(rolloff))
            mean_flux = float(np.mean(flux))
            std_flux = float(np.std(flux))
            mean_bandwidth = float(np.mean(bandwidth))
            std_bandwidth = float(np.std(bandwidth))
            mean_flatness = float(np.mean(flatness))
            std_flatness = float(np.std(flatness))
            
            # Normalize to audio's Nyquist frequency
            normalized_mean_centroid = mean_centroid / (sr / 2)
            normalized_mean_rolloff = mean_rolloff / (sr / 2)
            normalized_mean_bandwidth = mean_bandwidth / (sr / 2)
            
            # Create spectral features dictionary
            spectral_features = {
                "centroid": {
                    "mean": mean_centroid,
                    "std": std_centroid,
                    "normalized_mean": normalized_mean_centroid
                },
                "rolloff": {
                    "mean": mean_rolloff,
                    "std": std_rolloff,
                    "normalized_mean": normalized_mean_rolloff
                },
                "flux": {
                    "mean": mean_flux,
                    "std": std_flux
                },
                "bandwidth": {
                    "mean": mean_bandwidth,
                    "std": std_bandwidth,
                    "normalized_mean": normalized_mean_bandwidth
                },
                "flatness": {
                    "mean": mean_flatness,
                    "std": std_flatness
                }
            }
            
            self.logger.info("Spectral features extraction complete")
            return spectral_features
            
        except Exception as e:
            error_msg = f"Spectral feature extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_mfccs(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract Mel-frequency cepstral coefficients (MFCCs) from the audio.
        
        MFCCs represent the timbre characteristics of the sound - the quality that
        distinguishes different types of sound production and different instruments.
        They capture the short-term power spectrum of the sound based on a linear 
        cosine transform of a log power spectrum on a nonlinear Mel scale of frequency.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing MFCC features.
            
        Raises:
            AudioProcessingError: If MFCC extraction fails.
        """
        try:
            self.logger.info("Extracting MFCCs...")
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )
            
            # Calculate delta (first-order difference) and delta-delta (second-order)
            # These represent the velocity and acceleration of MFCCs
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Calculate statistics for each coefficient
            mfcc_means = np.mean(mfccs, axis=1).tolist()
            mfcc_stds = np.std(mfccs, axis=1).tolist()
            mfcc_delta_means = np.mean(mfcc_delta, axis=1).tolist()
            mfcc_delta_stds = np.std(mfcc_delta, axis=1).tolist()
            mfcc_delta2_means = np.mean(mfcc_delta2, axis=1).tolist()
            mfcc_delta2_stds = np.std(mfcc_delta2, axis=1).tolist()
            
            # Create MFCC features dictionary
            mfcc_features = {
                "coefficients": {
                    "means": mfcc_means,
                    "stds": mfcc_stds
                },
                "delta": {
                    "means": mfcc_delta_means,
                    "stds": mfcc_delta_stds
                },
                "delta2": {
                    "means": mfcc_delta2_means,
                    "stds": mfcc_delta2_stds
                }
            }
            
            self.logger.info("MFCC extraction complete")
            return mfcc_features
            
        except Exception as e:
            error_msg = f"MFCC extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_chroma_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract chroma features from the audio.
        
        Chroma features represent the harmonic content of the audio by projecting
        the spectral content onto 12 bins representing the 12 semitones of the musical
        octave (C, C#, D, etc.). These features are particularly useful for analyzing
        the harmonic progression and structure of music.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing chroma features.
            
        Raises:
            AudioProcessingError: If chroma feature extraction fails.
        """
        try:
            self.logger.info("Extracting chroma features...")
            
            # Constant-Q chromagram - better for music analysis
            # Uses the Constant-Q Transform which has logarithmically-spaced frequency bins
            chroma_cq = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=self.hop_length
            )
            
            # STFT chromagram - uses the Short Time Fourier Transform
            chroma_stft = librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Calculate statistics
            mean_chroma_cq = np.mean(chroma_cq, axis=1).tolist()
            std_chroma_cq = np.std(chroma_cq, axis=1).tolist()
            mean_chroma_stft = np.mean(chroma_stft, axis=1).tolist()
            std_chroma_stft = np.std(chroma_stft, axis=1).tolist()
            
            # Calculate overall chroma variance (how much the harmonic content changes)
            chroma_cq_var = float(np.mean(np.var(chroma_cq, axis=1)))
            chroma_stft_var = float(np.mean(np.var(chroma_stft, axis=1)))
            
            # Calculate top chroma notes (most prominent pitches)
            top_chroma_cq = np.argsort(mean_chroma_cq)[-3:].tolist()
            top_chroma_stft = np.argsort(mean_chroma_stft)[-3:].tolist()
            
            # Map indices to note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            top_notes_cq = [note_names[i % 12] for i in top_chroma_cq]
            top_notes_stft = [note_names[i % 12] for i in top_chroma_stft]
            
            # Create chroma features dictionary
            chroma_features = {
                "cqt": {
                    "means": mean_chroma_cq,
                    "stds": std_chroma_cq,
                    "variance": chroma_cq_var,
                    "top_notes_indices": top_chroma_cq,
                    "top_notes": top_notes_cq
                },
                "stft": {
                    "means": mean_chroma_stft,
                    "stds": std_chroma_stft,
                    "variance": chroma_stft_var,
                    "top_notes_indices": top_chroma_stft,
                    "top_notes": top_notes_stft
                }
            }
            
            self.logger.info("Chroma features extraction complete")
            return chroma_features
            
        except Exception as e:
            error_msg = f"Chroma feature extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_rms_energy(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract RMS energy features from the audio.
        
        RMS (Root Mean Square) energy measures the amplitude/loudness variations in the audio,
        providing insights into the dynamics of the music. It can identify dramatic changes
        in volume, sustained loud or quiet passages, and overall dynamic range.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing RMS energy features.
            
        Raises:
            AudioProcessingError: If RMS energy extraction fails.
        """
        try:
            self.logger.info("Extracting RMS energy...")
            
            # Extract RMS energy over time
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Calculate overall statistics
            mean_rms = float(np.mean(rms))
            std_rms = float(np.std(rms))
            max_rms = float(np.max(rms))
            min_rms = float(np.min(rms))
            
            # Calculate dynamic range
            dynamic_range = max_rms - min_rms
            
            # Detect peaks (sudden increases in energy)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, height=mean_rms + std_rms)
            
            # Convert peak frames to timestamps
            peak_times = librosa.frames_to_time(
                peaks, sr=sr, hop_length=self.hop_length
            ).tolist()
            
            # Calculate energy curve slope (how quickly loudness changes)
            if len(rms) > 1:
                energy_slope = np.diff(rms)
                mean_slope = float(np.mean(np.abs(energy_slope)))
            else:
                mean_slope = 0.0
            
            # Create RMS energy features dictionary
            rms_features = {
                "mean": mean_rms,
                "std": std_rms,
                "max": max_rms,
                "min": min_rms,
                "dynamic_range": float(dynamic_range),
                "peak_count": len(peaks),
                "peak_times": peak_times[:10] if len(peak_times) > 10 else peak_times,  # Limit to 10 peaks
                "mean_slope": mean_slope
            }
            
            self.logger.info("RMS energy extraction complete")
            return rms_features
            
        except Exception as e:
            error_msg = f"RMS energy extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_zero_crossing_rate(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract zero crossing rate features from the audio.
        
        Zero crossing rate (ZCR) measures how often the audio signal crosses the zero axis,
        which is an indicator of the noisiness or "roughness" of the sound. High ZCR values
        typically indicate percussive sounds, consonants in speech, or noisy textures.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing zero crossing rate features.
            
        Raises:
            AudioProcessingError: If zero crossing rate extraction fails.
        """
        try:
            self.logger.info("Extracting zero crossing rate...")
            
            # Extract zero crossing rate over time
            zcr = librosa.feature.zero_crossing_rate(
                y, hop_length=self.hop_length
            )[0]
            
            # Calculate statistics
            mean_zcr = float(np.mean(zcr))
            std_zcr = float(np.std(zcr))
            median_zcr = float(np.median(zcr))
            
            # Detect high ZCR segments (potential percussive or noisy parts)
            high_zcr_threshold = mean_zcr + std_zcr
            high_zcr_frames = np.where(zcr > high_zcr_threshold)[0]
            
            # Convert frames to timestamps
            high_zcr_times = librosa.frames_to_time(
                high_zcr_frames, sr=sr, hop_length=self.hop_length
            ).tolist()
            
            # Calculate ZCR distribution by dividing the range into bins
            n_bins = 10
            hist, edges = np.histogram(zcr, bins=n_bins, density=True)
            
            # Create zero crossing rate features dictionary
            zcr_features = {
                "mean": mean_zcr,
                "std": std_zcr,
                "median": median_zcr,
                "high_zcr_segment_count": len(high_zcr_frames),
                "high_zcr_times": high_zcr_times[:10] if len(high_zcr_times) > 10 else high_zcr_times,  # Limit to 10
                "histogram": hist.tolist(),
                "histogram_edges": edges.tolist()
            }
            
            self.logger.info("Zero crossing rate extraction complete")
            return zcr_features
            
        except Exception as e:
            error_msg = f"Zero crossing rate extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def extract_spectral_contrast(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral contrast features from the audio.
        
        Spectral contrast represents the distribution of sound energy across frequency bands.
        It measures the difference between peaks and valleys in the spectrum, which helps
        distinguish between different instruments and vocals. High contrast indicates the 
        presence of both strong harmonic content and noise, typical in music with 
        multiple instruments.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing spectral contrast features.
            
        Raises:
            AudioProcessingError: If spectral contrast extraction fails.
        """
        try:
            self.logger.info("Extracting spectral contrast...")
            
            # Extract spectral contrast with 6 bands (default)
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Calculate statistics for each band
            mean_contrast = np.mean(contrast, axis=1).tolist()
            std_contrast = np.std(contrast, axis=1).tolist()
            
            # Calculate overall contrast (mean of all bands)
            overall_contrast = float(np.mean(mean_contrast))
            
            # Calculate contrast variance (how much the contrast changes over time)
            contrast_variance = float(np.mean(np.var(contrast, axis=1)))
            
            # Determine minimum and maximum contrast bands
            min_contrast_band = int(np.argmin(mean_contrast))
            max_contrast_band = int(np.argmax(mean_contrast))
            
            # Create spectral contrast features dictionary
            contrast_features = {
                "band_means": mean_contrast,
                "band_stds": std_contrast,
                "overall_contrast": overall_contrast,
                "contrast_variance": contrast_variance,
                "min_contrast_band": min_contrast_band,
                "max_contrast_band": max_contrast_band
            }
            
            self.logger.info("Spectral contrast extraction complete")
            return contrast_features
            
        except Exception as e:
            error_msg = f"Spectral contrast extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e
    
    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all extracted features to ensure consistency.
        
        This method applies appropriate normalization strategies to different feature types
        to ensure values are consistently scaled, making them more suitable for LLM interpretation.
        
        Args:
            features: Dictionary containing all extracted features.
            
        Returns:
            Dictionary with normalized features.
        """
        self.logger.info("Normalizing features...")
        
        # Create a copy to avoid modifying the original
        normalized = features.copy()
        
        # Add normalization metadata
        normalized["_metadata"] = {
            "normalized": True,
            "normalization_method": "mixed"
        }
        
        # We'll implement selective normalization based on feature type
        # Most features have already been normalized in their respective extraction methods
        # Here we'll add any additional normalization needed for consistency
        
        # For example, ensure all features with "means" and "stds" have similar formats
        for feature_type in normalized:
            if isinstance(normalized[feature_type], dict):
                # For numeric values that should be between 0-1
                if "confidence" in normalized[feature_type]:
                    normalized[feature_type]["confidence"] = min(1.0, max(0.0, normalized[feature_type]["confidence"]))
        
        self.logger.info("Feature normalization complete")
        return normalized
    
    # Existing methods remain unchanged
    # extract_tempo(), extract_key(), extract_basic_stats(), _categorize_tempo()
    
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
        
    def segment_audio(self, y: np.ndarray, sr: int, segment_duration: float = 1.0) -> List[Dict[str, Any]]:
        """
        Segment audio into time frames for temporal analysis.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            segment_duration: Duration of each segment in seconds (default: 1.0).
            
        Returns:
            List of dictionaries containing segmented audio data and timestamps.
            
        Raises:
            AudioProcessingError: If segmentation fails.
        """
        try:
            self.logger.info(f"Segmenting audio into {segment_duration}s frames...")
            
            # Calculate total number of samples in the audio
            total_samples = len(y)
            
            # Calculate number of samples per segment
            samples_per_segment = int(segment_duration * sr)
            
            # Calculate number of segments
            num_segments = max(1, int(total_samples / samples_per_segment))
            
            # Create segments
            segments = []
            for i in range(num_segments):
                start_sample = i * samples_per_segment
                end_sample = min(start_sample + samples_per_segment, total_samples)
                
                # Extract segment audio
                segment_y = y[start_sample:end_sample]
                
                # Calculate timestamp
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                # Store segment data
                segment = {
                    "index": i,
                    "waveform": segment_y,
                    "sample_rate": sr,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "n_samples": len(segment_y)
                }
                
                segments.append(segment)
            
            self.logger.info(f"Audio segmented into {len(segments)} segments")
            return segments
            
        except Exception as e:
            error_msg = f"Audio segmentation failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e

    def extract_segment_features(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features for a single audio segment.
        
        Args:
            segment: Dictionary containing segment audio data.
            
        Returns:
            Dictionary containing extracted features for the segment.
            
        Raises:
            AudioProcessingError: If feature extraction fails.
        """
        try:
            self.logger.debug(f"Extracting features for segment {segment['index']} (time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s)")
            
            # Extract segment data
            y = segment["waveform"]
            sr = segment["sample_rate"]
            
            # Initialize features dictionary
            features = {}
            
            # Extract basic statistics
            features["basic_stats"] = self.extract_basic_stats(y, sr)
            
            # Extract tempo and beat information
            features["tempo"] = self.extract_tempo(y, sr)
            
            # Extract key and mode
            features["key"] = self.extract_key(y, sr)
            
            # Extract spectral features
            features["spectral"] = self.extract_spectral_features(y, sr)
            
            # Extract MFCCs for timbre analysis
            features["mfccs"] = self.extract_mfccs(y, sr)
            
            # Extract chroma features for harmonic content
            features["chroma"] = self.extract_chroma_features(y, sr)
            
            # Extract RMS energy for dynamics
            features["rms_energy"] = self.extract_rms_energy(y, sr)
            
            # Extract zero crossing rate for texture
            features["zero_crossing_rate"] = self.extract_zero_crossing_rate(y, sr)
            
            # Extract spectral contrast for instrumentation
            features["spectral_contrast"] = self.extract_spectral_contrast(y, sr)
            
            # Add segment metadata
            features["metadata"] = {
                "segment_index": segment["index"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["duration"]
            }
            
            return features
            
        except Exception as e:
            error_msg = f"Segment feature extraction failed for segment {segment['index']}: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e

    def extract_temporal_features(self, y: np.ndarray, sr: int, segment_duration: float = 1.0) -> Dict[str, Any]:
        """
        Perform temporal analysis by extracting features over segmented audio.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            segment_duration: Duration of each segment in seconds (default: 1.0).
            
        Returns:
            Dictionary containing temporal features across segments.
            
        Raises:
            AudioProcessingError: If temporal analysis fails.
        """
        try:
            self.logger.info("Starting temporal feature analysis...")
            
            # Segment the audio
            segments = self.segment_audio(y, sr, segment_duration)
            
            # Extract features for each segment
            segment_features = []
            
            for segment in segments:
                features = self.extract_segment_features(segment)
                segment_features.append(features)
            
            # Analyze temporal variations and transitions
            tempo_variations = self.detect_tempo_variations(segment_features)
            key_changes = self.detect_key_changes(segment_features)
            energy_dynamics = self.analyze_energy_dynamics(segment_features)
            transitions = self.detect_significant_transitions(segment_features)
            
            # Combine into temporal features dictionary
            temporal_features = {
                "segments": segment_features,
                "segment_count": len(segment_features),
                "segment_duration": segment_duration,
                "tempo_variations": tempo_variations,
                "key_changes": key_changes,
                "energy_dynamics": energy_dynamics,
                "transitions": transitions
            }
            
            self.logger.info("Temporal feature analysis complete")
            return temporal_features
            
        except Exception as e:
            error_msg = f"Temporal feature analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e

    def detect_tempo_variations(self, segment_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze how tempo changes over time across segments.
        
        Args:
            segment_features: List of dictionaries containing segment features.
            
        Returns:
            Dictionary containing tempo variation analysis.
        """
        self.logger.info("Analyzing tempo variations...")
        
        # Extract tempo values from each segment
        tempos = [segment["tempo"]["bpm"] for segment in segment_features]
        tempo_times = [segment["metadata"]["start_time"] for segment in segment_features]
        
        # Calculate statistics
        mean_tempo = float(np.mean(tempos)) if tempos else 0
        std_tempo = float(np.std(tempos)) if tempos else 0
        min_tempo = float(np.min(tempos)) if tempos else 0
        max_tempo = float(np.max(tempos)) if tempos else 0
        tempo_range = max_tempo - min_tempo
        
        # Calculate tempo changes between segments
        tempo_changes = []
        if len(tempos) > 1:
            tempo_changes = np.diff(tempos).tolist()
        
        # Identify significant tempo changes (more than 10% change)
        significant_changes = []
        for i, change in enumerate(tempo_changes):
            if abs(change) > 0.1 * tempos[i]:
                significant_changes.append({
                    "segment_index": i,
                    "time": tempo_times[i+1],
                    "previous_tempo": tempos[i],
                    "new_tempo": tempos[i+1],
                    "change": change
                })
        
        # Create tempo variations dictionary
        tempo_variations = {
            "mean_tempo": mean_tempo,
            "std_tempo": std_tempo,
            "min_tempo": min_tempo,
            "max_tempo": max_tempo,
            "tempo_range": tempo_range,
            "tempo_stability": 1.0 - (std_tempo / mean_tempo if mean_tempo > 0 else 0),
            "tempo_by_segment": tempos,
            "tempo_changes": tempo_changes,
            "significant_changes": significant_changes,
            "significant_change_count": len(significant_changes)
        }
        
        self.logger.info(f"Tempo variation analysis complete: stability = {tempo_variations['tempo_stability']:.2f}, significant changes = {len(significant_changes)}")
        return tempo_variations

    def detect_key_changes(self, segment_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect changes in musical key across segments.
        
        Args:
            segment_features: List of dictionaries containing segment features.
            
        Returns:
            Dictionary containing key change analysis.
        """
        self.logger.info("Analyzing key changes...")
        
        # Extract key information from each segment
        keys = [segment["key"]["key"] for segment in segment_features]
        modes = [segment["key"]["mode"] for segment in segment_features]
        key_times = [segment["metadata"]["start_time"] for segment in segment_features]
        
        # Create combined key+mode strings
        key_modes = [f"{key} {mode}" for key, mode in zip(keys, modes)]
        
        # Detect key changes
        key_changes = []
        if len(key_modes) > 1:
            for i in range(1, len(key_modes)):
                if key_modes[i] != key_modes[i-1]:
                    key_changes.append({
                        "segment_index": i,
                        "time": key_times[i],
                        "previous_key": key_modes[i-1],
                        "new_key": key_modes[i]
                    })
        
        # Create key changes dictionary
        key_change_analysis = {
            "keys_by_segment": key_modes,
            "key_changes": key_changes,
            "key_change_count": len(key_changes),
            "most_common_key": max(set(key_modes), key=key_modes.count) if key_modes else None
        }
        
        self.logger.info(f"Key change analysis complete: detected {len(key_changes)} key changes")
        return key_change_analysis

    def analyze_silence_ratio(self, y: np.ndarray, sr: int, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Calculate the ratio of silence to sound and identify silent sections.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            threshold: Amplitude threshold below which audio is considered silent (default: 0.01).
            
        Returns:
            Dictionary containing silence analysis.
        """
        self.logger.info("Analyzing silence ratio...")
        
        # Calculate absolute amplitude
        amplitude = np.abs(y)
        
        # Identify silent samples
        is_silent = amplitude < threshold
        
        # Calculate overall silence ratio
        silence_ratio = float(np.mean(is_silent))
        
        # Find silent sections (consecutive silent samples)
        silent_sections = []
        if len(is_silent) > 0:
            # Find boundaries of silent sections
            transitions = np.diff(is_silent.astype(int))
            starts = np.where(transitions == 1)[0] + 1
            ends = np.where(transitions == -1)[0] + 1
            
            # Handle edge cases (silent at beginning or end)
            if is_silent[0]:
                starts = np.insert(starts, 0, 0)
            if is_silent[-1]:
                ends = np.append(ends, len(is_silent))
            
            # Create silent sections data
            for start, end in zip(starts, ends):
                duration = (end - start) / sr
                # Only include sections longer than 0.1 seconds
                if duration > 0.1:
                    silent_sections.append({
                        "start_time": start / sr,
                        "end_time": end / sr,
                        "duration": duration
                    })
        
        # Create silence analysis dictionary
        silence_analysis = {
            "silence_ratio": silence_ratio,
            "silent_sections": silent_sections,
            "silent_section_count": len(silent_sections),
            "total_silence_duration": sum(section["duration"] for section in silent_sections),
            "avg_silence_duration": np.mean([section["duration"] for section in silent_sections]) if silent_sections else 0
        }
        
        self.logger.info(f"Silence analysis complete: ratio = {silence_ratio:.2f}, identified {len(silent_sections)} silent sections")
        return silence_analysis

    def analyze_energy_dynamics(self, segment_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze energy dynamics (loudness changes) across segments.
        
        Args:
            segment_features: List of dictionaries containing segment features.
            
        Returns:
            Dictionary containing energy dynamics analysis.
        """
        self.logger.info("Analyzing energy dynamics...")
        
        # Extract RMS energy values from each segment
        energies = [segment["rms_energy"]["mean"] for segment in segment_features]
        energy_times = [segment["metadata"]["start_time"] for segment in segment_features]
        
        # Calculate statistics
        mean_energy = float(np.mean(energies)) if energies else 0
        std_energy = float(np.std(energies)) if energies else 0
        min_energy = float(np.min(energies)) if energies else 0
        max_energy = float(np.max(energies)) if energies else 0
        dynamic_range = max_energy - min_energy
        
        # Calculate energy changes between segments
        energy_changes = []
        if len(energies) > 1:
            energy_changes = np.diff(energies).tolist()
        
        # Identify significant energy changes (more than 20% change)
        significant_changes = []
        for i, change in enumerate(energy_changes):
            if abs(change) > 0.2 * energies[i] and energies[i] > 0:
                significant_changes.append({
                    "segment_index": i,
                    "time": energy_times[i+1],
                    "previous_energy": energies[i],
                    "new_energy": energies[i+1],
                    "change": change
                })
        
        # Create energy dynamics dictionary
        energy_dynamics = {
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "min_energy": min_energy,
            "max_energy": max_energy,
            "dynamic_range": dynamic_range,
            "energy_stability": 1.0 - (std_energy / mean_energy if mean_energy > 0 else 0),
            "energy_by_segment": energies,
            "energy_changes": energy_changes,
            "significant_changes": significant_changes,
            "significant_change_count": len(significant_changes)
        }
        
        self.logger.info(f"Energy dynamics analysis complete: stability = {energy_dynamics['energy_stability']:.2f}, significant changes = {len(significant_changes)}")
        return energy_dynamics

    def detect_significant_transitions(self, segment_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect significant transitions in the music by analyzing multiple feature changes.
        
        Args:
            segment_features: List of dictionaries containing segment features.
            
        Returns:
            Dictionary containing transition analysis.
        """
        self.logger.info("Detecting significant musical transitions...")
        
        # Initialize arrays to store transition scores for each segment
        if not segment_features:
            return {"transitions": [], "transition_count": 0}
        
        segment_count = len(segment_features)
        transition_scores = np.zeros(segment_count - 1)
        transition_times = [segment["metadata"]["start_time"] for segment in segment_features][1:]
        
        # Helper function to calculate normalized difference
        def normalized_diff(a, b, scale=1.0):
            if a == 0 and b == 0:
                return 0
            return abs(a - b) / (max(abs(a), abs(b)) + 1e-5) * scale
        
        # Calculate transition scores based on multiple features
        for i in range(segment_count - 1):
            current = segment_features[i]
            next_segment = segment_features[i+1]
            
            # Tempo change (weighted 0.2)
            tempo_diff = normalized_diff(
                current["tempo"]["bpm"], 
                next_segment["tempo"]["bpm"], 
                scale=0.2
            )
            
            # Energy change (weighted 0.25)
            energy_diff = normalized_diff(
                current["rms_energy"]["mean"], 
                next_segment["rms_energy"]["mean"], 
                scale=0.25
            )
            
            # Spectral change (weighted 0.2)
            spectral_diff = normalized_diff(
                current["spectral"]["centroid"]["mean"],
                next_segment["spectral"]["centroid"]["mean"],
                scale=0.2
            )
            
            # Harmonic change (weighted 0.2)
            # Check if keys are different
            key_diff = 0.2 if current["key"]["key"] != next_segment["key"]["key"] else 0
            mode_diff = 0.1 if current["key"]["mode"] != next_segment["key"]["mode"] else 0
            
            # Calculate overall transition score
            score = tempo_diff + energy_diff + spectral_diff + key_diff + mode_diff
            transition_scores[i] = min(1.0, score)  # Cap at 1.0
        
        # Find significant transitions (score > 0.4)
        significant_transitions = []
        for i, score in enumerate(transition_scores):
            if score > 0.4:
                significant_transitions.append({
                    "segment_index": i + 1,  # +1 because it's the start of the next segment
                    "time": transition_times[i],
                    "score": float(score),
                    "transition_type": "major" if score > 0.7 else "minor"
                })
        
        # Create transitions dictionary
        transitions = {
            "transition_scores": transition_scores.tolist(),
            "significant_transitions": significant_transitions,
            "transition_count": len(significant_transitions),
            "major_transition_count": sum(1 for t in significant_transitions if t["transition_type"] == "major"),
            "transition_times": [t["time"] for t in significant_transitions]
        }
        
        self.logger.info(f"Transition detection complete: found {len(significant_transitions)} significant transitions")
        return transitions

    def generate_audio_fingerprint(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Generate a comprehensive audio fingerprint combining all features.
        
        Args:
            y: Audio time series.
            sr: Sample rate.
            
        Returns:
            Dictionary containing the complete audio fingerprint.
            
        Raises:
            AudioProcessingError: If fingerprint generation fails.
        """
        try:
            self.logger.info("Generating comprehensive audio fingerprint...")
            
            # Extract static features
            static_features = self.extract_features({"waveform": y, "sample_rate": sr})
            
            # Extract temporal features
            temporal_features = self.extract_temporal_features(y, sr)
            
            # Calculate silence analysis
            silence_analysis = self.analyze_silence_ratio(y, sr)
            
            # Combine into comprehensive fingerprint
            fingerprint = {
                "static_features": static_features,
                "temporal_features": temporal_features,
                "silence_analysis": silence_analysis,
                "metadata": {
                    "duration": float(librosa.get_duration(y=y, sr=sr)),
                    "sample_rate": sr,
                    "n_samples": len(y),
                    "fingerprint_version": "1.0"
                }
            }
            
            self.logger.info("Audio fingerprint generation complete")
            return fingerprint
            
        except Exception as e:
            error_msg = f"Audio fingerprint generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E002") from e    