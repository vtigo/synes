"""
Response processor for parsing LLM output into visualization instructions.
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple


class ResponseProcessor:
    """
    Parses LLM responses into structured visualization parameters.
    """
    
    def __init__(self):
        """Initialize the response processor."""
        self.logger = logging.getLogger(__name__)
        
        # Define required fields for validation
        self.required_top_level_fields = [
            "emotional_journey", 
            "color_palette", 
            "base_shapes", 
            "frames"
        ]
        
        self.required_frame_fields = [
            "timestamp", 
            "dominant_emotion", 
            "shapes", 
            "colors"
        ]
        
        # Define fallback values
        self.fallback_emotions = [
            "neutral", "calm", "energetic", "melancholic", 
            "joyful", "tense", "peaceful", "powerful"
        ]
        
        self.fallback_colors = [
            "#1a237e", "#0d47a1", "#0288d1", "#00acc1", 
            "#00897b", "#2e7d32", "#f57f17", "#e65100"
        ]
        
        self.fallback_shapes = ["circle", "square", "triangle", "rectangle"]
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse an LLM response into visualization parameters.
        
        Args:
            response: String containing the LLM response.
            
        Returns:
            Dictionary containing visualization parameters.
            
        Raises:
            ValueError: If the response cannot be parsed.
        """
        self.logger.info("Parsing LLM response")
        
        try:
            # Try to extract JSON from the response
            json_data = self._extract_json(response)
            
            if json_data is None:
                self.logger.warning("No valid JSON found in response, attempting alternate extraction")
                json_data = self._extract_json_with_regex(response)
            
            if json_data is None:
                self.logger.error("Failed to extract JSON from response")
                raise ValueError("Failed to extract valid JSON from LLM response")
            
            # Validate the structure
            self.validate_response_structure(json_data)
            
            # Normalize values
            normalized_data = self.normalize_emotional_values(json_data)
            
            self.logger.info("Successfully parsed LLM response")
            return normalized_data
            
        except ValueError:
            # Re-raise ValueErrors for validation failures
            raise
            
        except Exception as e:
            error_msg = f"Failed to parse LLM response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text.
        
        Args:
            text: String that may contain JSON.
            
        Returns:
            Parsed JSON as a dictionary, or None if extraction fails.
        """
        try:
            # First, try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to find a JSON object within the text
            try:
                # Look for text that might be JSON (between curly braces)
                start_index = text.find('{')
                end_index = text.rfind('}')
                
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_text = text[start_index:end_index + 1]
                    return json.loads(json_text)
                
                return None
                
            except (json.JSONDecodeError, ValueError):
                return None
    
    def _extract_json_with_regex(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON using regex for cases where normal extraction fails.
        
        Args:
            text: String that may contain JSON.
            
        Returns:
            Parsed JSON as a dictionary, or None if extraction fails.
        """
        try:
            # Look for patterns that indicate JSON objects
            # This is a more aggressive approach for malformed responses
            pattern = r'\{\s*"[^"]+"\s*:'
            matches = re.finditer(pattern, text)
            
            for match in matches:
                start_index = match.start()
                
                # Try to find the matching closing brace
                open_braces = 1
                for i in range(start_index + 1, len(text)):
                    if text[i] == '{':
                        open_braces += 1
                    elif text[i] == '}':
                        open_braces -= 1
                        
                    if open_braces == 0:
                        # Found a complete JSON object
                        potential_json = text[start_index:i + 1]
                        try:
                            return json.loads(potential_json)
                        except json.JSONDecodeError:
                            continue
            
            return None
            
        except Exception:
            return None
    
    def validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """
        Verify the parsed response contains all required fields.
        
        Args:
            response: Dictionary containing parsed LLM response.
            
        Returns:
            Boolean indicating if response is valid.
            
        Raises:
            ValueError: If the response is invalid.
        """
        self.logger.info("Validating response structure")
        
        # Check for required top-level fields
        missing_fields = [f for f in self.required_top_level_fields if f not in response]
        
        if missing_fields:
            error_msg = f"Missing required fields in response: {', '.join(missing_fields)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check that emotional_journey is a list
        if not isinstance(response["emotional_journey"], list):
            error_msg = "Field 'emotional_journey' must be a list"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check that color_palette is a list
        if not isinstance(response["color_palette"], list):
            error_msg = "Field 'color_palette' must be a list"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check that base_shapes is a list
        if not isinstance(response["base_shapes"], list):
            error_msg = "Field 'base_shapes' must be a list"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check that frames is a non-empty list
        if not isinstance(response["frames"], list) or not response["frames"]:
            error_msg = "Field 'frames' must be a non-empty list"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check each frame for required fields
        for i, frame in enumerate(response["frames"]):
            if not isinstance(frame, dict):
                error_msg = f"Frame {i} must be a dictionary"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check for required frame fields
            frame_missing_fields = [f for f in self.required_frame_fields if f not in frame]
            
            if frame_missing_fields:
                error_msg = f"Frame {i} is missing required fields: {', '.join(frame_missing_fields)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate frame field types
            if not isinstance(frame["timestamp"], (int, float)):
                error_msg = f"Frame {i}: 'timestamp' must be a number"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(frame["dominant_emotion"], str):
                error_msg = f"Frame {i}: 'dominant_emotion' must be a string"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(frame["shapes"], list):
                error_msg = f"Frame {i}: 'shapes' must be a list"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(frame["colors"], list):
                error_msg = f"Frame {i}: 'colors' must be a list"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        self.logger.info("Response structure validation successful")
        return True
    
    def normalize_emotional_values(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure emotion values are consistent and normalized.
        
        Args:
            response: Dictionary containing parsed LLM response.
            
        Returns:
            Dictionary with normalized emotion values.
        """
        self.logger.info("Normalizing emotional values")
        
        # Create a copy to avoid modifying the original
        normalized = response.copy()
        
        # Ensure emotional_journey contains only lowercase strings
        if "emotional_journey" in normalized:
            normalized["emotional_journey"] = [
                str(emotion).lower().strip() for emotion in normalized["emotional_journey"]
                if emotion
            ]
        
        # Normalize emotions in frames
        if "frames" in normalized:
            for frame in normalized["frames"]:
                # Normalize dominant_emotion
                if "dominant_emotion" in frame:
                    frame["dominant_emotion"] = str(frame["dominant_emotion"]).lower().strip()
                
                # Normalize secondary_emotion if present
                if "secondary_emotion" in frame:
                    frame["secondary_emotion"] = str(frame["secondary_emotion"]).lower().strip()
        
        # Normalize color values
        if "color_palette" in normalized:
            normalized["color_palette"] = [
                self._normalize_color(color) for color in normalized["color_palette"]
                if color
            ]
        
        # Normalize shape names
        if "base_shapes" in normalized:
            normalized["base_shapes"] = [
                str(shape).lower().strip() for shape in normalized["base_shapes"]
                if shape
            ]
        
        # Ensure timestamps are in ascending order
        if "frames" in normalized:
            normalized["frames"] = sorted(
                normalized["frames"], 
                key=lambda frame: frame.get("timestamp", 0)
            )
        
        self.logger.info("Emotional values normalization complete")
        return normalized
    
    def _normalize_color(self, color: Union[str, Dict[str, Any]]) -> str:
        """
        Normalize a color value to a valid hex color code.
        
        Args:
            color: Color value to normalize (string or dict).
            
        Returns:
            Normalized hex color code.
        """
        # If color is already a string
        if isinstance(color, str):
            color_str = color.strip()
            
            # Check if it's already a hex code
            if color_str.startswith('#'):
                # Ensure it's a valid hex code
                if len(color_str) in (4, 7, 9):  # #RGB, #RRGGBB, or #RRGGBBAA
                    return color_str
            
            # Handle named colors
            color_map = {
                "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF",
                "yellow": "#FFFF00", "purple": "#800080", "orange": "#FFA500",
                "black": "#000000", "white": "#FFFFFF", "gray": "#808080"
            }
            
            color_lower = color_str.lower()
            if color_lower in color_map:
                return color_map[color_lower]
            
            # Default to black if unrecognized
            return "#000000"
            
        # If color is a dictionary (e.g., {"r": 255, "g": 0, "b": 0})
        elif isinstance(color, dict):
            r = color.get("r", 0)
            g = color.get("g", 0)
            b = color.get("b", 0)
            
            # Convert to hex
            return f"#{r:02x}{g:02x}{b:02x}"
        
        # Default
        return "#000000"
    
    def get_fallback_response(self, duration: float = 180.0, num_frames: int = 20) -> Dict[str, Any]:
        """
        Generate a fallback visualization response when LLM output is invalid.
        
        Args:
            duration: Duration of the audio in seconds.
            num_frames: Number of frames to generate.
            
        Returns:
            Dictionary containing fallback visualization parameters.
        """
        self.logger.info("Generating fallback visualization response")
        
        # Generate frames
        frames = []
        for i in range(num_frames):
            # Calculate timestamp
            timestamp = i * (duration / (num_frames - 1)) if num_frames > 1 else 0
            
            # Select emotion (cycle through available emotions)
            emotion_index = i % len(self.fallback_emotions)
            secondary_index = (i + 1) % len(self.fallback_emotions)
            
            # Select color (cycle through available colors)
            color_index = i % len(self.fallback_colors)
            
            # Select shape (cycle through available shapes)
            shape_index = i % len(self.fallback_shapes)
            
            # Create frame
            frame = {
                "timestamp": timestamp,
                "dominant_emotion": self.fallback_emotions[emotion_index],
                "secondary_emotion": self.fallback_emotions[secondary_index],
                "shapes": [
                    {
                        "type": self.fallback_shapes[shape_index],
                        "size": 0.5,
                        "position": [0.5, 0.5],
                        "rotation": i * 15
                    }
                ],
                "colors": [self.fallback_colors[color_index]],
                "movements": ["rotate"],
                "transitions": []
            }
            
            frames.append(frame)
        
        # Create fallback response
        fallback_response = {
            "emotional_journey": self.fallback_emotions[:4],
            "color_palette": self.fallback_colors[:6],
            "base_shapes": self.fallback_shapes,
            "frames": frames,
            "_metadata": {
                "fallback": True,
                "reason": "Generated using fallback strategy due to invalid LLM response"
            }
        }
        
        self.logger.info(f"Generated fallback response with {num_frames} frames")
        return fallback_response
    
    def recover_from_malformed_response(self, response: str, duration: float = 180.0) -> Dict[str, Any]:
        """
        Attempt to recover useful information from a malformed response.
        
        Args:
            response: String containing the LLM response.
            duration: Duration of the audio in seconds.
            
        Returns:
            Dictionary containing recovered visualization parameters or fallback.
        """
        self.logger.info("Attempting to recover from malformed response")
        
        try:
            # Extract partial data using multiple approaches
            emotions = self._extract_emotions(response)
            colors = self._extract_colors(response)
            shapes = self._extract_shapes(response)
            frames = self._extract_frames(response)
            
            # If we couldn't extract any useful data, use complete fallback
            if not emotions and not colors and not shapes and not frames:
                self.logger.warning("Could not extract any useful data, using complete fallback")
                return self.get_fallback_response(duration)
            
            # Use fallback values for missing data
            if not emotions:
                emotions = self.fallback_emotions[:4]
            if not colors:
                colors = self.fallback_colors[:6]
            if not shapes:
                shapes = self.fallback_shapes
            
            # If we have no frames but have other data, generate frames
            if not frames:
                num_frames = max(10, int(duration / 20))  # One frame every 20 seconds by default
                frames = self._generate_frames_from_partial_data(
                    duration, num_frames, emotions, colors, shapes
                )
            
            # Create recovered response
            recovered_response = {
                "emotional_journey": emotions,
                "color_palette": colors,
                "base_shapes": shapes,
                "frames": frames,
                "_metadata": {
                    "recovered": True,
                    "reason": "Recovered from malformed response"
                }
            }
            
            self.logger.info("Successfully recovered partial data from malformed response")
            return recovered_response
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {str(e)}")
            return self.get_fallback_response(duration)
    
    def _extract_emotions(self, text: str) -> List[str]:
        """
        Extract emotion terms from text.
        
        Args:
            text: String to extract emotions from.
            
        Returns:
            List of extracted emotions.
        """
        # Common emotion words
        common_emotions = [
            "happy", "sad", "angry", "fearful", "disgusted", "surprised", 
            "calm", "excited", "tense", "relaxed", "content", "melancholic",
            "joyful", "anxious", "peaceful", "energetic", "nostalgic", "triumphant",
            "mysterious", "romantic", "playful", "solemn", "hopeful", "gloomy",
            "powerful", "tender", "aggressive", "serene", "passionate", "contemplative"
        ]
        
        # Look for emotions in the text
        found_emotions = []
        for emotion in common_emotions:
            if re.search(r'\b' + emotion + r'\b', text.lower()):
                found_emotions.append(emotion)
        
        # Limit to reasonable number
        return found_emotions[:10]
    
    def _extract_colors(self, text: str) -> List[str]:
        """
        Extract color codes from text.
        
        Args:
            text: String to extract colors from.
            
        Returns:
            List of extracted colors.
        """
        # Look for hex color codes
        hex_pattern = r'#[0-9a-fA-F]{3,8}\b'
        hex_colors = re.findall(hex_pattern, text)
        
        # Look for color names
        color_names = [
            "red", "green", "blue", "yellow", "purple", "orange",
            "black", "white", "gray", "pink", "brown", "cyan"
        ]
        
        named_colors = []
        for color in color_names:
            if re.search(r'\b' + color + r'\b', text.lower()):
                # Convert to hex
                color_map = {
                    "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF",
                    "yellow": "#FFFF00", "purple": "#800080", "orange": "#FFA500",
                    "black": "#000000", "white": "#FFFFFF", "gray": "#808080",
                    "pink": "#FFC0CB", "brown": "#A52A2A", "cyan": "#00FFFF"
                }
                named_colors.append(color_map.get(color.lower(), "#000000"))
        
        # Combine and limit to reasonable number
        all_colors = hex_colors + named_colors
        if not all_colors:
            return []
        
        return all_colors[:10]
    
    def _extract_shapes(self, text: str) -> List[str]:
        """
        Extract shape names from text.
        
        Args:
            text: String to extract shapes from.
            
        Returns:
            List of extracted shapes.
        """
        # Common shape names
        common_shapes = [
            "circle", "square", "triangle", "rectangle", "ellipse",
            "polygon", "star", "spiral", "wave", "line"
        ]
        
        # Look for shapes in the text
        found_shapes = []
        for shape in common_shapes:
            if re.search(r'\b' + shape + r'\b', text.lower()):
                found_shapes.append(shape)
        
        # Limit to reasonable number
        return found_shapes[:6]
    
    def _extract_frames(self, text: str) -> List[Dict[str, Any]]:
        """
        Attempt to extract frame data from text.
        
        Args:
            text: String to extract frames from.
            
        Returns:
            List of extracted frames.
        """
        frames = []
        
        # Look for frame patterns
        try:
            # Try to find timestamp patterns
            timestamp_pattern = r'\"timestamp\"\s*:\s*(\d+(\.\d+)?)'
            timestamps = re.findall(timestamp_pattern, text)
            
            if not timestamps:
                return []
            
            # For each timestamp, try to extract a frame
            for match in timestamps:
                timestamp = float(match[0])
                
                # Find the surrounding frame data (approximate)
                frame_start = text.find('{', max(0, text.find(f'timestamp":{timestamp}') - 50))
                if frame_start == -1:
                    continue
                
                # Find the end of this frame
                open_braces = 1
                frame_end = -1
                
                for i in range(frame_start + 1, min(frame_start + 1000, len(text))):
                    if text[i] == '{':
                        open_braces += 1
                    elif text[i] == '}':
                        open_braces -= 1
                        
                    if open_braces == 0:
                        frame_end = i + 1
                        break
                
                if frame_end == -1:
                    continue
                
                # Try to parse this frame
                try:
                    frame_text = text[frame_start:frame_end]
                    frame_data = json.loads(frame_text)
                    
                    # Ensure it has required fields
                    if "timestamp" in frame_data:
                        # Add defaults for missing fields
                        if "dominant_emotion" not in frame_data:
                            frame_data["dominant_emotion"] = "neutral"
                        
                        if "shapes" not in frame_data:
                            frame_data["shapes"] = [{"type": "circle", "size": 0.5}]
                        
                        if "colors" not in frame_data:
                            frame_data["colors"] = ["#0000FF"]
                        
                        frames.append(frame_data)
                except:
                    continue
            
            return frames
            
        except Exception:
            return []
    
    def _generate_frames_from_partial_data(
        self, duration: float, num_frames: int,
        emotions: List[str], colors: List[str], shapes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate frames using partial data extracted from a malformed response.
        
        Args:
            duration: Duration of the audio in seconds.
            num_frames: Number of frames to generate.
            emotions: List of emotions to use.
            colors: List of colors to use.
            shapes: List of shapes to use.
            
        Returns:
            List of generated frames.
        """
        frames = []
        
        for i in range(num_frames):
            # Calculate timestamp
            timestamp = i * (duration / (num_frames - 1)) if num_frames > 1 else 0
            
            # Select emotion (cycle through available emotions)
            emotion_index = i % len(emotions)
            secondary_index = (i + 1) % len(emotions)
            
            # Select color (cycle through available colors)
            color_index = i % len(colors)
            
            # Select shape (cycle through available shapes)
            shape_index = i % len(shapes)
            
            # Create frame
            frame = {
                "timestamp": timestamp,
                "dominant_emotion": emotions[emotion_index],
                "secondary_emotion": emotions[secondary_index],
                "shapes": [
                    {
                        "type": shapes[shape_index],
                        "size": 0.5,
                        "position": [0.5, 0.5],
                        "rotation": i * 15
                    }
                ],
                "colors": [colors[color_index]],
                "movements": ["rotate"],
                "transitions": []
            }
            
            frames.append(frame)
        
        return frames
    
    def extract_visualization_parameters(self, 
                                        response: Union[str, Dict[str, Any]],
                                        audio_duration: float = 180.0) -> Dict[str, Any]:
        """
        Extract visualization parameters from an LLM response with robust error handling.
        
        Args:
            response: String or dictionary containing LLM response.
            audio_duration: Duration of the audio in seconds (for fallback).
            
        Returns:
            Dictionary containing visualization parameters.
        """
        self.logger.info("Extracting visualization parameters")
        
        # If response is already a dictionary, validate and normalize it
        if isinstance(response, dict):
            try:
                self.validate_response_structure(response)
                return self.normalize_emotional_values(response)
            except ValueError as e:
                self.logger.warning(f"Invalid response structure: {str(e)}")
                try:
                    return self.recover_from_malformed_response(str(response), audio_duration)
                except Exception:
                    return self.get_fallback_response(audio_duration)
        
        # If response is a string, parse, validate, and normalize it
        try:
            parsed_response = self.parse_llm_response(response)
            return parsed_response
        except ValueError as e:
            self.logger.warning(f"Failed to parse response: {str(e)}")
            try:
                return self.recover_from_malformed_response(response, audio_duration)
            except Exception:
                return self.get_fallback_response(audio_duration)