"""
LLM processor for handling the LLM model and inference.
"""
import json
import logging
import time
from typing import Dict, Any, Optional, Union

from utils.error_handler import AudioProcessingError
from llm_interpreter.llm_interface import LLMInterface
from llm_interpreter.prompt_manager import PromptManager


class LLMProcessor:
    """
    Processes LLM inference requests.
    """
    
    def __init__(self, model_path: Optional[str] = None, timeout: int = 120):
        """
        Initialize the LLM processor.
        
        Args:
            model_path: Path or name of the model to use (default: None)
            timeout: Maximum time in seconds to wait for model inference (default: 120)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.timeout = timeout
        
        # Initialize the prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize the LLM interface
        self.llm_interface = LLMInterface(
            model_path=model_path,
            timeout=timeout
        )
        
        self.logger.info(f"Initialized LLM processor with model path: {model_path}")
    
    def process(self, formatted_data: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
        """
        Process formatted audio features through the LLM.
        
        Args:
            formatted_data: Dictionary containing formatted audio features.
            stream: Whether to stream the output (default: False).
            
        Returns:
            Dictionary containing visualization parameters from LLM.
            
        Raises:
            AudioProcessingError: If LLM processing fails.
        """
        self.logger.info("Processing audio features through LLM")
        
        start_time = time.time()
        
        try:
            # Create prompt using the prompt manager
            prompt = self.prompt_manager.create_prompt(formatted_data)
            
            # Process prompt through LLM
            llm_response = self.llm_interface.process(prompt, stream=stream)
            
            # Extract JSON from response
            visualization_params = self.llm_interface.extract_json_from_response(llm_response)
            
            if visualization_params is None:
                error_msg = "Failed to extract valid JSON from LLM response"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, "E003")
            
            # Validate the visualization parameters
            self._validate_visualization_params(visualization_params)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            visualization_params["_metadata"] = {
                "processing_time": processing_time,
                "response_length": len(llm_response),
                "model_path": self.model_path
            }
            
            self.logger.info(f"LLM processing complete in {processing_time:.2f} seconds")
            return visualization_params
            
        except AudioProcessingError:
            # Re-raise AudioProcessingErrors
            raise
            
        except Exception as e:
            error_msg = f"LLM processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003") from e
    
    def _validate_visualization_params(self, params: Dict[str, Any]) -> None:
        """
        Validate the visualization parameters from the LLM response.
        
        Args:
            params: Dictionary containing visualization parameters.
            
        Raises:
            AudioProcessingError: If the parameters are invalid.
        """
        # Check for required top-level fields
        required_fields = ["emotional_journey", "color_palette", "base_shapes", "frames"]
        
        for field in required_fields:
            if field not in params:
                error_msg = f"Missing required field in visualization parameters: {field}"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, "E003")
        
        # Check that frames is a non-empty list
        if not isinstance(params["frames"], list) or not params["frames"]:
            error_msg = "Frames must be a non-empty list"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003")
        
        # Check each frame for required fields
        for i, frame in enumerate(params["frames"]):
            if not isinstance(frame, dict):
                error_msg = f"Frame {i} is not a dictionary"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, "E003")
            
            frame_fields = ["timestamp", "dominant_emotion", "shapes", "colors"]
            for field in frame_fields:
                if field not in frame:
                    error_msg = f"Missing required field '{field}' in frame {i}"
                    self.logger.error(error_msg)
                    raise AudioProcessingError(error_msg, "E003")
        
        self.logger.debug("Visualization parameters validation successful")
    
    def fallback_processing(self, formatted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback visualization parameters if LLM processing fails.
        
        Args:
            formatted_data: Dictionary containing formatted audio features.
            
        Returns:
            Dictionary containing fallback visualization parameters.
        """
        self.logger.info("Generating fallback visualization parameters")
        
        # Extract basic info from formatted data
        summary = formatted_data.get("summary", {})
        duration = float(summary.get("duration", "180.0").split()[0])
        
        # Create basic emotion list
        emotions = summary.get("potential_emotions", ["neutral", "calm"])
        
        # Create basic color palette (blue-green gradient)
        color_palette = ["#1a237e", "#0d47a1", "#0288d1", "#00acc1", "#00897b", "#2e7d32"]
        
        # Create basic shapes
        base_shapes = ["circle", "square", "triangle"]
        
        # Create frames (one every 10 seconds)
        frames = []
        num_frames = max(3, int(duration / 10))
        
        for i in range(num_frames):
            timestamp = i * (duration / (num_frames - 1)) if num_frames > 1 else 0
            
            # Select emotion (cycle through available emotions)
            emotion_index = i % len(emotions)
            
            frame = {
                "timestamp": timestamp,
                "dominant_emotion": emotions[emotion_index],
                "secondary_emotion": emotions[(emotion_index + 1) % len(emotions)],
                "shapes": [
                    {
                        "type": base_shapes[i % len(base_shapes)],
                        "size": 0.5,
                        "position": [0.5, 0.5],
                        "rotation": i * 30
                    }
                ],
                "colors": [color_palette[i % len(color_palette)]],
                "movements": ["rotate"],
                "transitions": []
            }
            
            frames.append(frame)
        
        # Create fallback visualization parameters
        fallback_params = {
            "emotional_journey": emotions,
            "color_palette": color_palette,
            "base_shapes": base_shapes,
            "frames": frames,
            "_metadata": {
                "fallback": True,
                "reason": "LLM processing failed, using fallback parameters"
            }
        }
        
        self.logger.info(f"Generated fallback visualization with {len(frames)} frames")
        return fallback_params
    
    def test_processing(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test the LLM processing with sample data.
        
        Args:
            test_data: Optional formatted data to use for testing.
            
        Returns:
            Dictionary containing test results.
        """
        self.logger.info("Testing LLM processing")
        
        # Use provided data or get sample data from prompt manager
        formatted_data = test_data
        if formatted_data is None:
            formatted_data = self.prompt_manager._create_sample_data()
        
        try:
            # Process the formatted data
            visualization_params = self.process(formatted_data)
            
            # Create test results
            test_results = {
                "success": True,
                "emotional_journey": visualization_params.get("emotional_journey", []),
                "color_palette_size": len(visualization_params.get("color_palette", [])),
                "base_shapes": visualization_params.get("base_shapes", []),
                "frame_count": len(visualization_params.get("frames", [])),
                "processing_time": visualization_params.get("_metadata", {}).get("processing_time", 0)
            }
            
            self.logger.info(f"LLM processing test successful with {test_results['frame_count']} frames "
                          f"in {test_results['processing_time']:.2f} seconds")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"LLM processing test failed: {str(e)}")
            
            # Try fallback processing
            try:
                fallback_params = self.fallback_processing(formatted_data)
                
                return {
                    "success": False,
                    "error": str(e),
                    "fallback_success": True,
                    "fallback_frame_count": len(fallback_params.get("frames", []))
                }
            except Exception as fallback_error:
                return {
                    "success": False,
                    "error": str(e),
                    "fallback_success": False,
                    "fallback_error": str(fallback_error)
                }