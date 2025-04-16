"""
Prompt manager for creating fixed templates for LLM.
"""
import json
import logging
from typing import Dict, Any, Optional


class PromptManager:
    """
    Creates and manages prompts for the LLM.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.logger = logging.getLogger(__name__)
        
        # Base template without the JSON example
        self.base_template_part1 = """
You are an emotional interpreter for music. Analyze the following audio features
and create a visualization mapping that represents the emotional journey of this music
through abstract shapes and geometric patterns.

Audio Features:
- Tempo: {tempo_data}
- Key: {key_data}
- Timbre: {timbre_data}
- Harmony: {harmony_data}
- Dynamics: {dynamics_data}
- Texture: {texture_data}
{temporal_data}

Create a visualization that maps these features to visual elements including colors,
shapes, patterns, and movements. Your output should be a frame-by-frame description
of how the visualization evolves.

Output your response as a JSON object with the following structure:
"""

        # JSON example as a literal string (not processed by string formatting)
        self.json_example = """
{
  "emotional_journey": [list of primary emotions detected],
  "color_palette": [list of hex color codes],
  "base_shapes": [list of primary shapes to use],
  "frames": [
    {
      "timestamp": float,
      "dominant_emotion": string,
      "secondary_emotion": string,
      "shapes": [...],
      "colors": [...],
      "movements": [...],
      "transitions": [...]
    },
    ...
  ]
}
"""

        # Final part of the template
        self.base_template_part2 = """

Make sure your response is valid JSON. Don't include any explanations outside the JSON structure.
"""
    
    def create_prompt(self, formatted_data: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM using the formatted audio features.
        
        Args:
            formatted_data: Dictionary containing formatted audio features.
            
        Returns:
            String containing the prompt for the LLM.
            
        Raises:
            ValueError: If required data is missing.
        """
        self.logger.info("Creating prompt for LLM from formatted audio features")
        
        # Validate that required data is present
        self._validate_formatted_data(formatted_data)
        
        # Extract required components
        summary = formatted_data.get("summary", {})
        detailed = formatted_data.get("detailed_features", {})
        temporal = formatted_data.get("temporal", {})
        
        # Format tempo data
        tempo_data = self._format_tempo_section(detailed.get("tempo", {}))
        
        # Format key data
        key_data = self._format_key_section(detailed.get("key", {}))
        
        # Format timbre data
        timbre_data = self._format_timbre_section(detailed.get("timbre", {}))
        
        # Format harmony data
        harmony_data = self._format_harmony_section(detailed.get("harmony", {}))
        
        # Format dynamics data
        dynamics_data = self._format_dynamics_section(detailed.get("dynamics", {}))
        
        # Format texture data
        texture_data = self._format_texture_section(detailed.get("texture", {}))
        
        # Format temporal data if available
        temporal_data = self._format_temporal_section(temporal) if temporal else ""
        
        # Fill the template part 1 (everything before the JSON example)
        prompt_part1 = self.base_template_part1.format(
            tempo_data=tempo_data,
            key_data=key_data,
            timbre_data=timbre_data,
            harmony_data=harmony_data,
            dynamics_data=dynamics_data,
            texture_data=texture_data,
            temporal_data=temporal_data
        )
        
        # Combine all parts to form the complete prompt
        prompt = prompt_part1 + self.json_example + self.base_template_part2
        
        # Validate the completed prompt
        self._validate_completed_prompt(prompt)
        
        self.logger.info("Prompt created successfully")
        return prompt
    
    def _validate_formatted_data(self, formatted_data: Dict[str, Any]) -> None:
        """
        Validate that all required data is present in the formatted data.
        
        Args:
            formatted_data: Dictionary containing formatted audio features.
            
        Raises:
            ValueError: If required data is missing.
        """
        if not formatted_data:
            self.logger.error("Formatted data is empty")
            raise ValueError("Formatted data cannot be empty")
        
        # Check for summary
        if "summary" not in formatted_data:
            self.logger.error("Missing 'summary' in formatted data")
            raise ValueError("Formatted data must contain 'summary'")
        
        # Check for detailed features
        if "detailed_features" not in formatted_data:
            self.logger.error("Missing 'detailed_features' in formatted data")
            raise ValueError("Formatted data must contain 'detailed_features'")
        
        # Check for required detailed features
        detailed = formatted_data["detailed_features"]
        required_features = ["tempo", "key", "timbre"]
        
        for feature in required_features:
            if feature not in detailed:
                self.logger.error(f"Missing '{feature}' in detailed features")
                raise ValueError(f"Detailed features must contain '{feature}'")
    
    def _validate_completed_prompt(self, prompt: str) -> None:
        """
        Validate that the completed prompt is properly formatted.
        
        Args:
            prompt: The completed prompt string.
            
        Raises:
            ValueError: If the prompt is invalid.
        """
        # Check prompt length
        if len(prompt) < 100:
            self.logger.error("Prompt is too short, likely missing data")
            raise ValueError("Generated prompt is too short, may be missing data")
        
        # Check for all required sections
        required_sections = [
            "Audio Features:",
            "Tempo:",
            "Key:",
            "Timbre:",
            "Output your response as a JSON object"
        ]
        
        for section in required_sections:
            if section not in prompt:
                self.logger.error(f"Prompt is missing required section: {section}")
                raise ValueError(f"Prompt is missing required section: {section}")
        
        # Check for JSON structure example
        if "emotional_journey" not in prompt or "frames" not in prompt:
            self.logger.error("Prompt is missing JSON structure example")
            raise ValueError("Prompt is missing JSON structure example")
        
        self.logger.debug("Prompt validation successful")
    
    def _format_tempo_section(self, tempo_data: Dict[str, Any]) -> str:
        """Format tempo data for the prompt."""
        if not tempo_data:
            return "No tempo data available"
        
        return (f"{tempo_data.get('description', 'Unknown tempo')} at "
                f"{tempo_data.get('bpm', 'unknown')} BPM with "
                f"{tempo_data.get('stability', 'unknown stability')}")
    
    def _format_key_section(self, key_data: Dict[str, Any]) -> str:
        """Format key data for the prompt."""
        if not key_data:
            return "No key data available"
        
        emotional = ", ".join(key_data.get("emotional_qualities", ["unknown"]))
        return (f"{key_data.get('full_key', 'Unknown key')} with "
                f"{key_data.get('tonality_strength', 'unknown strength')} tonality. "
                f"Emotional qualities: {emotional}")
    
    def _format_timbre_section(self, timbre_data: Dict[str, Any]) -> str:
        """Format timbre data for the prompt."""
        if not timbre_data:
            return "No timbre data available"
        
        return timbre_data.get("description", "Unknown timbre characteristics")
    
    def _format_harmony_section(self, harmony_data: Dict[str, Any]) -> str:
        """Format harmony data for the prompt."""
        if not harmony_data:
            return "No harmony data available"
        
        top_notes = harmony_data.get("top_notes", "unknown")
        return (f"{harmony_data.get('complexity', 'Unknown harmonic complexity')}. "
                f"Prominent notes: {top_notes}")
    
    def _format_dynamics_section(self, dynamics_data: Dict[str, Any]) -> str:
        """Format dynamics data for the prompt."""
        if not dynamics_data:
            return "No dynamics data available"
        
        return (f"{dynamics_data.get('description', 'Unknown dynamics')}. "
                f"{dynamics_data.get('dynamic_peaks', 'Unknown peaks')}")
    
    def _format_texture_section(self, texture_data: Dict[str, Any]) -> str:
        """Format texture data for the prompt."""
        if not texture_data:
            return "No texture data available"
        
        return texture_data.get("description", "Unknown texture characteristics")
    
    def _format_temporal_section(self, temporal_data: Dict[str, Any]) -> str:
        """Format temporal data for the prompt."""
        if not temporal_data or "sections" not in temporal_data:
            return ""
        
        sections = temporal_data.get("sections", [])
        evolution = temporal_data.get("evolution", "No temporal evolution data")
        
        temporal_str = f"\nTemporal Evolution:\n{evolution}\n\nSections:"
        
        for i, section in enumerate(sections[:5]):  # Limit to 5 sections to keep prompt size reasonable
            start = section.get("start_time", "unknown")
            end = section.get("end_time", "unknown")
            key = section.get("key", "unknown key")
            tempo = section.get("tempo", "unknown tempo")
            dynamics = section.get("dynamics", "unknown dynamics")
            emotions = ", ".join(section.get("emotions", ["unknown"]))
            
            temporal_str += f"\n- Section {i+1} ({start}s to {end}s): {key}, {tempo}, {dynamics}. Emotions: {emotions}"
        
        if len(sections) > 5:
            temporal_str += f"\n- Plus {len(sections) - 5} more sections..."
        
        return temporal_str
    
    def test_prompt_generation(self, formatted_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Test prompt generation with sample data or provided data.
        
        Args:
            formatted_data: Optional formatted data to use for testing.
            
        Returns:
            Generated prompt string.
        """
        self.logger.info("Testing prompt generation")
        
        # Use provided data or create sample data
        if formatted_data is None:
            formatted_data = self._create_sample_data()
        
        # Generate prompt
        prompt = self.create_prompt(formatted_data)
        
        return prompt
    
    def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample formatted data for testing."""
        return {
            "summary": {
                "duration": "180.5 seconds",
                "tempo": {
                    "bpm": 120.5,
                    "description": "moderate and steady"
                },
                "key": "C major",
                "dynamics": "loud and prominent",
                "texture": "moderately textured",
                "complexity": "moderately complex",
                "potential_emotions": ["joyful", "energetic", "optimistic"]
            },
            "detailed_features": {
                "tempo": {
                    "description": "moderate and steady",
                    "bpm": 120.5,
                    "category": "moderate",
                    "beats": 450,
                    "stability": "very steady"
                },
                "key": {
                    "key": "C",
                    "mode": "major",
                    "full_key": "C major",
                    "tonality_strength": "clear and defined",
                    "emotional_qualities": ["bright", "positive", "uplifting"]
                },
                "timbre": {
                    "brightness": "bright and clear",
                    "complexity": "somewhat complex",
                    "description": "bright and clear, somewhat complex timbre"
                },
                "harmony": {
                    "complexity": "moderately simple with straightforward chord progressions",
                    "top_notes": "C, G, E"
                },
                "dynamics": {
                    "overall_level": "loud and prominent",
                    "dynamic_range": "moderate dynamic range with clear variations",
                    "dynamic_peaks": "several volume peaks creating emphasis",
                    "description": "loud and prominent with moderate dynamic range with clear variations"
                },
                "texture": {
                    "quality": "moderately textured",
                    "consistency": "mostly consistent texture with subtle variations",
                    "description": "moderately textured with mostly consistent texture with subtle variations"
                }
            },
            "temporal": {
                "sections": [
                    {
                        "start_time": 0.0,
                        "end_time": 45.5,
                        "key": "C major",
                        "tempo": "moderate and steady",
                        "dynamics": "moderate in volume",
                        "texture": "smooth and coherent",
                        "emotions": ["peaceful", "optimistic"]
                    },
                    {
                        "start_time": 45.5,
                        "end_time": 90.2,
                        "key": "C major",
                        "tempo": "moderate and steady",
                        "dynamics": "loud and prominent",
                        "texture": "moderately textured",
                        "emotions": ["joyful", "energetic"]
                    },
                    {
                        "start_time": 90.2,
                        "end_time": 180.5,
                        "key": "G major",
                        "tempo": "fast and energetic",
                        "dynamics": "very loud and powerful",
                        "texture": "rough and complex",
                        "emotions": ["triumphant", "excited"]
                    }
                ],
                "section_count": 3,
                "key_changes": [
                    {"time": 0.0, "key": "C major"},
                    {"time": 90.2, "key": "G major"}
                ],
                "tempo_changes": [
                    {"time": 0.0, "tempo": "moderate", "bpm": 120.5},
                    {"time": 90.2, "tempo": "fast", "bpm": 140.2}
                ],
                "energy_changes": [
                    {"time": 0.0, "energy": "medium", "rms": 0.15},
                    {"time": 45.5, "energy": "high", "rms": 0.25},
                    {"time": 90.2, "energy": "high", "rms": 0.35}
                ],
                "evolution": "The piece has a three-part structure. It stays in C major for two-thirds of the duration, then transitions to G major. The tempo changes from moderate to fast. The energy progresses from medium, then high, then high."
            }
        }