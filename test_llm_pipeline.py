#!/usr/bin/env python3
"""
Test script for the LLM interpretation pipeline in the Music-to-Visual Emotion Interpreter System.

This script demonstrates how to use the LLM processor to convert formatted audio features
into visualization parameters. It can be used with either sample data or data from an
actual audio file.

Usage:
    python test_llm_pipeline.py [--input <json_path>] [--output <json_path>] [--model <model_path>]
"""

import os
import json
import argparse
import logging
import time
from typing import Dict, Any, Optional

from utils.logging_config import initialize_logging
from llm_interpreter.prompt_manager import PromptManager
from llm_interpreter.llm_processor import LLMProcessor
from utils.error_handler import AudioProcessingError, ErrorHandler


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the LLM interpretation pipeline."
    )
    parser.add_argument("--input", help="Path to JSON file with formatted audio features")
    parser.add_argument("--output", help="Path to save visualization parameters JSON")
    parser.add_argument("--model", help="Path to custom LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--timeout", type=int, default=120, help="LLM inference timeout in seconds")
    parser.add_argument("--fallback", action="store_true", help="Use fallback processing if LLM fails")
    
    return parser.parse_args()


def load_formatted_data(json_path: str) -> Dict[str, Any]:
    """
    Load formatted audio features from a JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        Dictionary containing formatted data.
        
    Raises:
        ValueError: If the file cannot be loaded or parsed.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        raise ValueError(f"Failed to load formatted data from {json_path}: {str(e)}")


def save_visualization_params(params: Dict[str, Any], output_path: str) -> None:
    """
    Save visualization parameters to a JSON file.
    
    Args:
        params: Dictionary containing visualization parameters.
        output_path: Path to save the JSON file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)


def main():
    """Main function for testing the LLM pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging_config = initialize_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    # Create error handler
    error_handler = ErrorHandler()
    
    try:
        logger.info("LLM Interpretation Pipeline Test")
        logger.info("-" * 40)
        
        # Initialize prompt manager and processor
        prompt_manager = PromptManager()
        llm_processor = LLMProcessor(
            model_path=args.model,
            timeout=args.timeout
        )
        
        # Get formatted data - either from input file or sample data
        formatted_data = None
        if args.input:
            logger.info(f"Loading formatted data from: {args.input}")
            formatted_data = load_formatted_data(args.input)
        else:
            logger.info("Using sample formatted data")
            formatted_data = prompt_manager._create_sample_data()
        
        # Log data summary
        summary = formatted_data.get("summary", {})
        logger.info(f"Processing audio data: {summary.get('duration', 'Unknown')} duration, "
                  f"{summary.get('tempo', {}).get('bpm', 'Unknown')} BPM, "
                  f"Key: {summary.get('key', 'Unknown')}")
        
        # Process data through LLM
        start_time = time.time()
        logger.info("Processing through LLM...")
        
        try:
            # Try regular processing
            visualization_params = llm_processor.process(formatted_data)
            success = True
        except AudioProcessingError as e:
            logger.error(f"LLM processing failed: {str(e)}")
            
            if args.fallback:
                logger.info("Using fallback processing")
                visualization_params = llm_processor.fallback_processing(formatted_data)
                success = False
            else:
                raise
        
        processing_time = time.time() - start_time
        
        # Log results
        if success:
            emotions = visualization_params.get("emotional_journey", [])
            frames = visualization_params.get("frames", [])
            colors = visualization_params.get("color_palette", [])
            
            logger.info(f"LLM processing successful in {processing_time:.2f} seconds")
            logger.info(f"Detected emotions: {', '.join(emotions[:5])}" + 
                      ("..." if len(emotions) > 5 else ""))
            logger.info(f"Generated {len(frames)} frames with {len(colors)} colors")
        else:
            logger.info(f"Using fallback visualization in {processing_time:.2f} seconds")
        
        # Save results if output path provided
        if args.output:
            logger.info(f"Saving visualization parameters to: {args.output}")
            save_visualization_params(visualization_params, args.output)
        
        return 0
        
    except Exception as e:
        error_handler.handle_error(e)
        logger.error(f"Test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())