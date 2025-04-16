"""
Configuration parameters for the Music-to-Visual Emotion Interpreter System.
"""

# Audio processing configurations
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "hop_length": 512,
    "n_fft": 2048,
    "n_mfcc": 13,
}

# LLM configurations
LLM_CONFIG = {
    "model_name": "llama-3",
    "temperature": 0.7,
    "max_length": 2048,
}

# Visualization configurations
VISUALIZATION_CONFIG = {
    "resolution": (1280, 720),
    "fps": 30,
    "color_depth": 24,
}

# Output configurations
OUTPUT_CONFIG = {
    "video_codec": "libx264",
    "audio_codec": "aac",
    "max_processing_time": 300,  # 5 minutes in seconds
}

# Error codes
ERROR_CODES = {
    "E001": "Invalid MP3 file",
    "E002": "Feature extraction failed",
    "E003": "LLM inference timeout",
    "E004": "Visualization generation error",
    "E005": "Encoding failure",
}