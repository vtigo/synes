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
from typing import Dict, Any

from audio_analyzer.audio_processor import AudioProcessor
from audio_analyzer.feature_extractor import FeatureExtractor
from audio_analyzer.data_formatter import DataFormatter
from llm_interpreter.llm_processor import LLMProcessor
from llm_interpreter.prompt_manager import PromptManager
from llm_interpreter.response_parser import ResponseParser
from visualization_generator.graphics_engine import GraphicsEngine
from visualization_generator.render_pipeline import RenderPipeline
from visualization_generator.output_processor import OutputProcessor
from utils.error_handler import ErrorHandler
from utils.file_utils import validate_file

import config


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MP3 audio files to emotional visual representations."
    )
    parser.add_argument("--input", required=True, help="Path to input MP3 file")
    parser.add_argument("--output", required=True, help="Path to output MP4 file")
    parser.add_argument("--model", help="Path to custom LLM model")
    
    args = parser.parse_args()
    return vars(args)


def main():
    """Main execution function for the music visualizer."""
    print("Music-to-Visual Emotion Interpreter System")
    print("-----------------------------------------")
    
    # Parse arguments
    args = parse_arguments()
    input_file = args["input"]
    output_file = args["output"]
    model_path = args.get("model")
    
    # Validate input file
    if not validate_file(input_file, file_type="mp3"):
        print(f"Error: {config.ERROR_CODES['E001']}")
        return 1
    
    try:
        start_time = time.time()
        
        # Initialize components
        audio_processor = AudioProcessor()
        feature_extractor = FeatureExtractor()
        data_formatter = DataFormatter()
        
        prompt_manager = PromptManager()
        llm_processor = LLMProcessor(model_path=model_path)
        response_parser = ResponseParser()
        
        graphics_engine = GraphicsEngine()
        render_pipeline = RenderPipeline()
        output_processor = OutputProcessor()
        
        # Process audio
        print("Loading and processing audio file...")
        audio_data = audio_processor.load_audio(input_file)
        
        print("Extracting audio features...")
        audio_features = feature_extractor.extract_features(audio_data)
        
        formatted_data = data_formatter.format_features(audio_features)
        
        # Generate LLM interpretation
        print("Generating emotional interpretation...")
        prompt = prompt_manager.create_prompt(formatted_data)
        llm_response = llm_processor.process(prompt)
        
        visualization_params = response_parser.parse_response(llm_response)
        
        # Generate visualization
        print("Creating visualization...")
        frames = graphics_engine.generate_frames(visualization_params)
        
        rendered_frames = render_pipeline.render(frames)
        
        print("Creating output video...")
        output_processor.create_video(rendered_frames, audio_data, output_file)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle_error(e)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)