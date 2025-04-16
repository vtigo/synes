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
    
    args = parser.parse_args()
    return vars(args)


def main():
    """Main execution function for the music visualizer."""
    # Parse arguments
    args = parse_arguments()
    input_file = args["input"]
    output_file = args["output"]
    model_path = args.get("model")
    verbose = args.get("verbose", False)
    
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
            
            # At this point, we would continue with feature extraction, LLM processing, etc.
            # but that's outside the scope of the current task
            logger.info("Audio file successfully loaded and validated")
            
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