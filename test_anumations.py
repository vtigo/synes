#!/usr/bin/env python3
"""
Test script for the visualization generator framework.

This script creates a simple test animation to verify that the visualization 
framework is working correctly.
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
import argparse
from typing import Dict, Any, List, Optional

# Import the visualization module
from visualization_generator.graphics_engine import Animation
from utils.logging_config import initialize_logging
from utils.error_handler import ErrorHandler, AudioProcessingError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the visualization framework."
    )
    parser.add_argument("--output", default="test_animation.mp4", 
                       help="Path to save the output video file (default: test_animation.mp4)")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Duration of the test animation in seconds (default: 5.0)")
    parser.add_argument("--width", type=int, default=800,
                       help="Width of the animation in pixels (default: 800)")
    parser.add_argument("--height", type=int, default=600,
                       help="Height of the animation in pixels (default: 600)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def create_test_visualization_parameters(duration: float, fps: int) -> Dict[str, Any]:
    """
    Create test visualization parameters.
    
    Args:
        duration: Duration of the animation in seconds.
        fps: Frames per second.
        
    Returns:
        Dictionary containing test visualization parameters.
    """
    # Calculate number of frames
    num_frames = int(duration * fps)
    
    # Colors for the animation
    colors = [
        "#FF0000", "#FF7F00", "#FFFF00", "#00FF00", 
        "#0000FF", "#4B0082", "#9400D3"
    ]
    
    # Shapes for the animation
    base_shapes = ["circle", "star", "rectangle", "polygon"]
    
    # Emotions for the animation
    emotions = ["happy", "calm", "excited", "peaceful", "melancholic", "energetic"]
    
    # Create frames
    frames = []
    
    for i in range(num_frames):
        # Calculate time
        t = i / (num_frames - 1) if num_frames > 1 else 0
        timestamp = t * duration
        
        # Choose emotion (cycle through emotions)
        emotion_index = int(t * len(emotions))
        emotion = emotions[emotion_index % len(emotions)]
        
        # Choose color (cycle through colors)
        color_index = int(t * len(colors))
        color = colors[color_index % len(colors)]
        
        # Choose shape (cycle through shapes)
        shape_index = int(t * len(base_shapes))
        shape_type = base_shapes[shape_index % len(base_shapes)]
        
        # Calculate animated parameters
        radius = 0.1 + 0.05 * np.sin(t * 2 * np.pi)
        x = 0.5 + 0.3 * np.sin(t * 2 * np.pi)
        y = 0.5 + 0.3 * np.cos(t * 2 * np.pi)
        angle = t * 360
        
        # Create shape data based on type
        if shape_type == "circle":
            shape_data = {
                "type": "circle",
                "center": (x, y),
                "radius": radius,
                "color": color
            }
        elif shape_type == "star":
            shape_data = {
                "type": "star",
                "center": (x, y),
                "radius": radius * 1.5,
                "points": 5,
                "color": color
            }
        elif shape_type == "rectangle":
            shape_data = {
                "type": "rectangle",
                "xy": (x - radius, y - radius),
                "width": radius * 2,
                "height": radius * 2,
                "angle": angle,
                "color": color
            }
        elif shape_type == "polygon":
            # Create a triangle
            shape_data = {
                "type": "polygon",
                "points": [
                    (x, y + radius * 1.5),
                    (x - radius, y - radius * 0.5),
                    (x + radius, y - radius * 0.5)
                ],
                "color": color
            }
        
        # Create frame
        frame = {
            "timestamp": timestamp,
            "dominant_emotion": emotion,
            "secondary_emotion": emotions[(emotion_index + 1) % len(emotions)],
            "shapes": [shape_data],
            "colors": [color],
            "movements": ["rotate"],
            "transitions": []
        }
        
        frames.append(frame)
    
    # Create visualization parameters
    visualization_params = {
        "emotional_journey": emotions,
        "color_palette": colors,
        "base_shapes": base_shapes,
        "frames": frames
    }
    
    return visualization_params


def create_animation_from_parameters(params: Dict[str, Any], width: int, height: int, fps: int) -> List[np.ndarray]:
    """
    Create an animation from visualization parameters.
    
    Args:
        params: Dictionary containing visualization parameters.
        width: Width of the animation in pixels.
        height: Height of the animation in pixels.
        fps: Frames per second.
        
    Returns:
        List of numpy arrays containing frames.
    """
    # Create animation object
    animation = Animation(width=width, height=height, fps=fps)
    
    # Generate frames
    frames = animation.generate_frames_from_parameters(params)
    
    # Close animation
    animation.close()
    
    return frames


def create_and_save_animation(output_path: str, frames: List[np.ndarray], fps: int) -> None:
    """
    Create and save an animation from frames.
    
    Args:
        output_path: Path to save the animation.
        frames: List of numpy arrays containing frames.
        fps: Frames per second.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating animation with {len(frames)} frames at {fps} fps")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create moviepy clip
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Write to file
    clip.write_videofile(output_path, fps=fps, codec='libx264', audio=False)
    
    logger.info(f"Animation saved to: {output_path}")


def main():
    """Main function to test the visualization framework."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging_config = initialize_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    # Create error handler
    error_handler = ErrorHandler()
    
    try:
        logger.info("Testing visualization framework")
        
        # Method 1: Using generate_test_animation
        logger.info("Method 1: Using generate_test_animation")
        animation = Animation(
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        # Create test visualization parameters
        params = create_test_visualization_parameters(args.duration, args.fps)
        
        # Generate frames
        frames = create_animation_from_parameters(params, args.width, args.height, args.fps)
        
        # Save animation
        create_and_save_animation(args.output, frames, args.fps)
        
        logger.info("Test completed successfully")
        
    except AudioProcessingError as e:
        error_handler.handle_error(e, "Error processing animation")
        logger.error(f"Animation processing error: {str(e)}")
    except Exception as e:
        error_handler.handle_error(e, "Unexpected error during animation generation")
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Clean up any resources
        plt.close('all')
        logger.info("Test completed")


if __name__ == "__main__":
    main()