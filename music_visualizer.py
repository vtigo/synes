#!/usr/bin/env python3
"""
Music-to-Visual Emotion Interpreter System

This application transforms MP3 audio files into visual representations (MP4)
based on emotional interpretation via an LLM. The system analyzes musical features
from the audio, processes them through an LLM to generate emotional interpretations,
and creates abstract/geometric visualizations that correspond to the emotional
journey of the music.

Usage:
    python music_visualizer.py --input <mp3_file_path> --output <mp4_file_path> [--model <model_path>]
"""

import os
import argparse
import time
import logging
import json
import numpy as np
from typing import Dict, Any

from audio_analyzer.input_processor import InputProcessor
from audio_analyzer.audio_processor import AudioProcessor
from audio_analyzer.feature_extractor import FeatureExtractor
from audio_analyzer.data_formatter import DataFormatter
from llm_interpreter.llm_processor import LLMProcessor
from llm_interpreter.prompt_manager import PromptManager
from llm_interpreter.response_parser import ResponseParser
from visualization_generator.graphics_engine import GraphicsEngine
from visualization_generator.render_pipeline import RenderPipeline
from visualization_generator.output_processor import OutputProcessor
from utils.error_handler import ErrorHandler, AudioProcessingError
from utils.file_utils import validate_file
from utils.logging_config import initialize_logging

import config


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MP3 audio files to emotional visual representations."
    )
    parser.add_argument("--input", required=True, help="Path to input MP3 file")
    parser.add_argument("--output", required=True, help="Path to output MP4 file")
    parser.add_argument("--model", help="Path to custom LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--extract-only", action="store_true", 
                        help="Only extract features without generating visualization")
    parser.add_argument("--save-features", help="Path to save extracted features as JSON")
    
    args = parser.parse_args()
    return vars(args)


def extract_audio_features(audio_data: Dict[str, Any], config_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Extract audio features from the provided audio data.
    
    Args:
        audio_data: Dictionary containing audio waveform and metadata.
        config_params: Optional configuration parameters for feature extraction.
        
    Returns:
        Dictionary containing extracted audio features.
        
    Raises:
        AudioProcessingError: If feature extraction fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting audio feature extraction...")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config=config_params)
    
    # Extract features
    features = feature_extractor.extract_features(audio_data)
    
    logger.info("Feature extraction completed successfully")
    
    return features


def save_features_to_json(features: Dict[str, Any], output_path: str) -> None:
    """
    Save extracted features to a JSON file.
    
    Args:
        features: Dictionary containing extracted features.
        output_path: Path where to save the JSON file.
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert NumPy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    # Convert and save to JSON
    json_features = convert_for_json(features)
    
    with open(output_path, 'w') as f:
        json.dump(json_features, f, indent=2)
    
    logger.info(f"Features saved to: {output_path}")


def display_feature_summary(features: Dict[str, Any]) -> None:
    """
    Display a summary of the extracted features.
    
    Args:
        features: Dictionary containing extracted features.
    """
    logger = logging.getLogger(__name__)
    
    # Extract and log key information
    tempo_info = features.get("tempo", {})
    key_info = features.get("key", {})
    basic_stats = features.get("basic_stats", {})
    
    # Display tempo information
    logger.info("=== Tempo Information ===")
    logger.info(f"BPM: {tempo_info.get('bpm', 'N/A'):.2f}")
    logger.info(f"Tempo Category: {tempo_info.get('tempo_category', 'N/A')}")
    logger.info(f"Beat Count: {tempo_info.get('beat_count', 'N/A')}")
    
    # Display key information
    logger.info("=== Key Information ===")
    logger.info(f"Key: {key_info.get('key', 'N/A')} {key_info.get('mode', 'N/A')}")
    logger.info(f"Confidence: {key_info.get('confidence', 'N/A'):.4f}")
    
    # Display basic statistics
    logger.info("=== Basic Audio Statistics ===")
    logger.info(f"Duration: {basic_stats.get('duration', 'N/A'):.2f} seconds")
    logger.info(f"RMS Energy: {basic_stats.get('rms', 'N/A'):.4f}")
    logger.info(f"Dynamic Range: {basic_stats.get('dynamic_range', 'N/A'):.4f}")
    logger.info(f"Silence Ratio: {basic_stats.get('silence_ratio', 'N/A'):.4f}")


def main():
    """Main execution function for the music visualizer."""
    # Parse arguments
    args = parse_arguments()
    input_file = args["input"]
    output_file = args["output"]
    model_path = args.get("model")
    verbose = args.get("verbose", False)
    extract_only = args.get("extract_only", False)
    save_features_path = args.get("save_features")
    
    # Initialize logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging_config = initialize_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Music-to-Visual Emotion Interpreter System")
    logger.info("-----------------------------------------")
    
    # Create error handler
    error_handler = ErrorHandler()
    
    try:
        start_time = time.time()
        
        # Initialize processors
        input_processor = InputProcessor()
        audio_processor = AudioProcessor()
        
        # Validate and load input file
        logger.info(f"Processing input file: {input_file}")
        
        try:
            # First get metadata without loading the full file
            metadata = input_processor.get_file_metadata(input_file)
            logger.info(f"File metadata: {metadata['file_name']}, "
                      f"Size: {metadata['file_size_mb']:.2f} MB, "
                      f"Duration: {metadata['duration']:.2f}s, "
                      f"Sample rate: {metadata['sample_rate']} Hz")
            
            # Then load the full audio file
            audio_data = input_processor.load_mp3_file(input_file)
            
            # Get basic audio stats using the audio processor
            audio_stats = audio_processor.get_audio_stats(audio_data)
            logger.info(f"Audio stats - Min: {audio_stats['min']:.4f}, "
                      f"Max: {audio_stats['max']:.4f}, "
                      f"RMS: {audio_stats['rms']:.4f}")
            
            # Extract audio features
            logger.info("Extracting audio features...")
            features = extract_audio_features(audio_data, config.AUDIO_CONFIG)
            
            # Display feature summary
            display_feature_summary(features)
            
            # Save features to JSON if requested
            if save_features_path:
                save_features_to_json(features, save_features_path)
            
            # If extract-only flag is set, stop here
            if extract_only:
                logger.info("Feature extraction completed successfully")
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")
                return 0
            
            # At this point, we would continue with LLM processing, visualization, etc.
            # but that's outside the scope of the current task
            logger.info("Audio features extracted successfully")
            
            # For now, just print a success message
            logger.info(f"Audio file processed successfully: {input_file}")
            logger.info(f"Output would be saved to: {output_file}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return 0
            
        except AudioProcessingError as e:
            error_handler.handle_error(e)
            logger.error(f"Audio processing error: {str(e)}")
            return 1
        
    except Exception as e:
        error_handler.handle_error(e)
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)