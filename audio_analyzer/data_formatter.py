"""
Data formatter for preparing extracted features for LLM input.
"""
from typing import Dict, Any, List, Optional
import logging
import numpy as np


class DataFormatter:
    """
    Formats extracted audio features for LLM processing.
    
    This class transforms technical audio features into a structured format
    optimized for LLM understanding of musical concepts and emotional connections.
    """
    
    def __init__(self):
        """Initialize the data formatter."""
        self.logger = logging.getLogger(__name__)
        
        # Define descriptive term mappings for various features
        self.tempo_terms = {
            "very_slow": "very slow and contemplative",
            "slow": "slow and relaxed",
            "moderate": "moderate and steady",
            "fast": "fast and energetic",
            "very_fast": "very fast and intense"
        }
        
        self.dynamics_terms = {
            "very_low": "very quiet and subtle",
            "low": "quiet and gentle",
            "moderate": "moderate in volume",
            "high": "loud and prominent",
            "very_high": "very loud and powerful"
        }
        
        self.texture_terms = {
            "very_smooth": "very smooth and flowing",
            "smooth": "smooth and coherent",
            "moderate": "moderately textured",
            "rough": "rough and complex",
            "very_rough": "very rough and chaotic"
        }
        
        # Define emotional connections to musical features
        self.major_emotions = ["joyful", "happy", "bright", "optimistic", "triumphant"]
        self.minor_emotions = ["melancholic", "sad", "dark", "introspective", "tense"]
        
        # Define required features for validation
        self.required_features = [
            "tempo", "key", "basic_stats", "spectral", 
            "mfccs", "chroma", "rms_energy"
        ]
    
    def validate_features(self, audio_features: Dict[str, Any]) -> bool:
        """
        Validate that all required features are present.
        
        Args:
            audio_features: Dictionary containing extracted audio features.
            
        Returns:
            Boolean indicating whether features are valid.
            
        Raises:
            ValueError: If required features are missing.
        """
        self.logger.info("Validating audio features...")
        
        # Check if all required feature categories are present
        missing_features = [f for f in self.required_features if f not in audio_features]
        
        if missing_features:
            error_msg = f"Missing required features: {', '.join(missing_features)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate specific required fields within features
        if "bpm" not in audio_features.get("tempo", {}):
            self.logger.error("Missing required field: tempo.bpm")
            raise ValueError("Missing required field: tempo.bpm")
            
        if "key" not in audio_features.get("key", {}):
            self.logger.error("Missing required field: key.key")
            raise ValueError("Missing required field: key.key")
            
        self.logger.info("Audio features validation successful")
        return True
    
    def categorize_value(self, value: float, category_map: Dict[str, str], 
                         thresholds: List[float]) -> str:
        """
        Categorize a numeric value into descriptive terms based on thresholds.
        
        Args:
            value: Numeric value to categorize.
            category_map: Dictionary mapping category keys to descriptive terms.
            thresholds: List of thresholds for categorization.
            
        Returns:
            Descriptive term for the value.
        """
        categories = list(category_map.keys())
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return category_map[categories[i]]
        return category_map[categories[-1]]
    
    def tempo_to_descriptive_term(self, bpm: float) -> str:
        """
        Convert BPM to a descriptive tempo term.
        
        Args:
            bpm: Tempo in beats per minute.
            
        Returns:
            Descriptive term for the tempo.
        """
        if bpm < 60:
            return self.tempo_terms["very_slow"]
        elif bpm < 90:
            return self.tempo_terms["slow"]
        elif bpm < 120:
            return self.tempo_terms["moderate"]
        elif bpm < 150:
            return self.tempo_terms["fast"]
        else:
            return self.tempo_terms["very_fast"]
    
    def dynamics_to_descriptive_term(self, rms: float) -> str:
        """
        Convert RMS energy to a descriptive dynamics term.
        
        Args:
            rms: RMS energy value.
            
        Returns:
            Descriptive term for the dynamics.
        """
        thresholds = [0.05, 0.1, 0.2, 0.3]
        return self.categorize_value(rms, self.dynamics_terms, thresholds)
    
    def texture_to_descriptive_term(self, zcr: float) -> str:
        """
        Convert zero crossing rate to a descriptive texture term.
        
        Args:
            zcr: Zero crossing rate value.
            
        Returns:
            Descriptive term for the texture.
        """
        thresholds = [0.02, 0.05, 0.1, 0.2]
        return self.categorize_value(zcr, self.texture_terms, thresholds)
    
    def suggest_emotions_from_features(self, audio_features: Dict[str, Any]) -> List[str]:
        """
        Suggest potential emotions based on audio features.
        
        Args:
            audio_features: Dictionary containing audio features.
            
        Returns:
            List of potential emotions.
        """
        potential_emotions = []
        
        # Get mode (major/minor)
        mode = audio_features.get("key", {}).get("mode", "")
        
        # Basic emotion connection based on mode
        if mode == "major":
            base_emotions = self.major_emotions
        else:
            base_emotions = self.minor_emotions
        
        # Get tempo category
        tempo_category = audio_features.get("tempo", {}).get("tempo_category", "moderate")
        
        # Modify emotions based on tempo
        if tempo_category in ["fast", "very_fast"]:
            if mode == "major":
                potential_emotions.extend(["excited", "energetic", "uplifting"])
            else:
                potential_emotions.extend(["anxious", "tense", "dramatic"])
        elif tempo_category in ["slow", "very_slow"]:
            if mode == "major":
                potential_emotions.extend(["peaceful", "serene", "content"])
            else:
                potential_emotions.extend(["somber", "reflective", "mournful"])
        
        # Add base emotions
        potential_emotions.extend(base_emotions[:2])  # Add a couple from the base set
        
        # Get dynamics (volume/energy)
        rms = audio_features.get("rms_energy", {}).get("mean", 0.1)
        
        # Modify emotions based on dynamics
        if rms > 0.2:  # High energy
            potential_emotions.extend(["intense", "powerful"])
        elif rms < 0.1:  # Low energy
            potential_emotions.extend(["delicate", "gentle", "intimate"])
        
        # Return unique emotions (no duplicates)
        return list(set(potential_emotions))
    
    def create_feature_summary(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a concise summary of key audio features.
        
        Args:
            audio_features: Dictionary containing extracted audio features.
            
        Returns:
            Dictionary containing summarized features.
        """
        self.logger.info("Creating feature summary...")
        
        # Validate features first
        self.validate_features(audio_features)
        
        # Extract key information
        bpm = audio_features["tempo"]["bpm"]
        key = audio_features["key"]["key"]
        mode = audio_features["key"]["mode"]
        duration = audio_features["basic_stats"]["duration"]
        
        # Convert to descriptive terms
        tempo_term = self.tempo_to_descriptive_term(bpm)
        
        # Get RMS energy and convert to descriptive term
        rms = audio_features["rms_energy"]["mean"]
        dynamics_term = self.dynamics_to_descriptive_term(rms)
        
        # Get zero crossing rate and convert to descriptive term
        zcr_mean = audio_features["zero_crossing_rate"]["mean"]
        texture_term = self.texture_to_descriptive_term(zcr_mean)
        
        # Calculate overall complexity based on spectral features
        spectral_cent = audio_features["spectral"]["centroid"]["normalized_mean"]
        spectral_contrast = audio_features["spectral_contrast"]["overall_contrast"]
        complexity_score = (spectral_cent + spectral_contrast) / 2
        
        if complexity_score < 0.3:
            complexity = "simple and minimal"
        elif complexity_score < 0.6:
            complexity = "moderately complex"
        else:
            complexity = "highly complex and elaborate"
        
        # Suggest potential emotions
        potential_emotions = self.suggest_emotions_from_features(audio_features)
        
        # Create summary dictionary
        summary = {
            "duration": f"{duration:.2f} seconds",
            "tempo": {
                "bpm": bpm,
                "description": tempo_term
            },
            "key": f"{key} {mode}",
            "dynamics": dynamics_term,
            "texture": texture_term,
            "complexity": complexity,
            "potential_emotions": potential_emotions
        }
        
        self.logger.info("Feature summary created successfully")
        return summary
    
    def generate_temporal_description(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a description of how the music evolves over time.
        
        Args:
            segments: List of dictionaries containing segment features.
            
        Returns:
            Dictionary containing temporal description.
        """
        self.logger.info("Generating temporal description...")
        
        if not segments:
            self.logger.warning("No segments provided for temporal description")
            return {
                "evolution": "No temporal data available",
                "sections": []
            }
        
        # Initialize description
        sections = []
        
        # Analyze key transitions
        key_changes = []
        last_key = None
        
        # Analyze tempo changes
        tempo_changes = []
        last_tempo = None
        
        # Analyze energy changes
        energy_changes = []
        last_energy = None
        
        # Process each segment
        for i, segment in enumerate(segments):
            # Get segment time
            start_time = segment["metadata"]["start_time"]
            
            # Get key information
            current_key = f"{segment['key']['key']} {segment['key']['mode']}"
            if last_key is None or current_key != last_key:
                key_changes.append({
                    "time": start_time,
                    "key": current_key
                })
                last_key = current_key
            
            # Get tempo information
            current_tempo = segment["tempo"]["tempo_category"]
            if last_tempo is None or current_tempo != last_tempo:
                tempo_changes.append({
                    "time": start_time,
                    "tempo": current_tempo,
                    "bpm": segment["tempo"]["bpm"]
                })
                last_tempo = current_tempo
            
            # Get energy information
            rms = segment["rms_energy"]["mean"]
            if rms < 0.1:
                current_energy = "low"
            elif rms < 0.2:
                current_energy = "medium"
            else:
                current_energy = "high"
                
            if last_energy is None or current_energy != last_energy:
                energy_changes.append({
                    "time": start_time,
                    "energy": current_energy,
                    "rms": rms
                })
                last_energy = current_energy
        
        # Find significant transitions as section boundaries
        section_boundaries = [0]  # Start with beginning of song
        
        # Add significant transitions from temporal features if available
        transitions = segments[0].get("transitions", {}).get("significant_transitions", [])
        if transitions:
            for transition in transitions:
                section_boundaries.append(transition["time"])
        else:
            # If no transitions data, use key and energy changes as section boundaries
            for change in key_changes[1:]:  # Skip the first one (beginning of song)
                section_boundaries.append(change["time"])
            
            for change in energy_changes[1:]:  # Skip the first one
                if change["time"] not in section_boundaries:
                    section_boundaries.append(change["time"])
        
        # Ensure section boundaries are unique and sorted
        section_boundaries = sorted(set(section_boundaries))
        
        # Create sections based on boundaries
        for i in range(len(section_boundaries)):
            start = section_boundaries[i]
            end = section_boundaries[i+1] if i+1 < len(section_boundaries) else None
            
            # Find segment that corresponds to this section
            section_segments = [s for s in segments if 
                               s["metadata"]["start_time"] >= start and 
                               (end is None or s["metadata"]["start_time"] < end)]
            
            if section_segments:
                # Use first segment in section for characteristics
                segment = section_segments[0]
                
                # Create section description
                section = {
                    "start_time": start,
                    "end_time": end,
                    "key": f"{segment['key']['key']} {segment['key']['mode']}",
                    "tempo": self.tempo_to_descriptive_term(segment["tempo"]["bpm"]),
                    "dynamics": self.dynamics_to_descriptive_term(segment["rms_energy"]["mean"]),
                    "texture": self.texture_to_descriptive_term(segment["zero_crossing_rate"]["mean"]),
                    "emotions": self.suggest_emotions_from_features(segment)
                }
                
                sections.append(section)
        
        # Create overall evolution description
        temporal_description = {
            "sections": sections,
            "section_count": len(sections),
            "key_changes": key_changes,
            "tempo_changes": tempo_changes,
            "energy_changes": energy_changes,
            "evolution": self.generate_evolution_summary(sections, key_changes, tempo_changes, energy_changes)
        }
        
        self.logger.info(f"Temporal description generated with {len(sections)} sections")
        return temporal_description
    
    def generate_evolution_summary(self, sections: List[Dict[str, Any]],
                                  key_changes: List[Dict[str, Any]],
                                  tempo_changes: List[Dict[str, Any]],
                                  energy_changes: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of how the music evolves over time.
        
        Args:
            sections: List of section descriptions.
            key_changes: List of key changes.
            tempo_changes: List of tempo changes.
            energy_changes: List of energy changes.
            
        Returns:
            String describing the evolution of the music.
        """
        if not sections:
            return "No temporal data available"
        
        summary = []
        
        # Describe overall structure
        if len(sections) == 1:
            summary.append("The piece maintains a consistent character throughout.")
        elif len(sections) == 2:
            summary.append("The piece has two distinct sections.")
        elif len(sections) == 3:
            summary.append("The piece has a three-part structure.")
        else:
            summary.append(f"The piece has a complex structure with {len(sections)} distinct sections.")
        
        # Describe key changes
        if len(key_changes) == 1:
            summary.append(f"It stays in {key_changes[0]['key']} throughout.")
        elif len(key_changes) == 2:
            summary.append(f"It transitions from {key_changes[0]['key']} to {key_changes[1]['key']}.")
        else:
            summary.append(f"It features {len(key_changes)} key changes, starting in {key_changes[0]['key']} and ending in {key_changes[-1]['key']}.")
        
        # Describe tempo changes
        if len(tempo_changes) == 1:
            summary.append(f"The tempo is consistently {tempo_changes[0]['tempo']}.")
        elif len(tempo_changes) == 2:
            summary.append(f"The tempo changes from {tempo_changes[0]['tempo']} to {tempo_changes[1]['tempo']}.")
        else:
            summary.append(f"It has {len(tempo_changes)} tempo changes, with notable shifts in pacing.")
        
        # Describe energy changes
        if len(energy_changes) == 1:
            summary.append(f"The energy level remains {energy_changes[0]['energy']} throughout.")
        elif len(energy_changes) > 3:
            summary.append(f"The piece features dynamic contrasts with {len(energy_changes)} changes in energy level.")
        else:
            energy_narrative = ", then ".join([f"{change['energy']}" for change in energy_changes])
            summary.append(f"The energy progresses from {energy_narrative}.")
        
        # Describe emotional journey if multiple sections
        if len(sections) > 1:
            first_emotions = sections[0]["emotions"][:1] if sections[0]["emotions"] else ["neutral"]
            last_emotions = sections[-1]["emotions"][:1] if sections[-1]["emotions"] else ["neutral"]
            
            if first_emotions != last_emotions:
                summary.append(f"Emotionally, it evolves from {', '.join(first_emotions)} to {', '.join(last_emotions)}.")
        
        return " ".join(summary)
    
    def format_features_for_llm(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the extracted features into a format suitable for LLM input.
        
        Args:
            audio_features: Dictionary containing extracted audio features.
            
        Returns:
            Dictionary with formatted features ready for LLM input.
        """
        self.logger.info("Formatting features for LLM input...")
        
        # Validate features first
        self.validate_features(audio_features)
        
        # Create feature summary
        summary = self.create_feature_summary(audio_features)
        
        # Generate temporal description if temporal features are available
        temporal_description = {}
        if "temporal_features" in audio_features:
            segments = audio_features["temporal_features"].get("segments", [])
            temporal_description = self.generate_temporal_description(segments)
        
        # Create comprehensive formatted data
        formatted_data = {
            "summary": summary,
            "temporal": temporal_description,
            "detailed_features": {
                "tempo": self.format_tempo_features(audio_features["tempo"]),
                "key": self.format_key_features(audio_features["key"]),
                "timbre": self.format_timbre_features(audio_features),
                "harmony": self.format_harmony_features(audio_features),
                "dynamics": self.format_dynamics_features(audio_features),
                "texture": self.format_texture_features(audio_features)
            }
        }
        
        self.logger.info("Features formatted successfully for LLM input")
        return formatted_data
    
    def format_tempo_features(self, tempo_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format tempo features for LLM understanding.
        
        Args:
            tempo_features: Dictionary containing tempo features.
            
        Returns:
            Dictionary with formatted tempo features.
        """
        bpm = tempo_features.get("bpm", 0)
        tempo_category = tempo_features.get("tempo_category", "moderate")
        beat_count = tempo_features.get("beat_count", 0)
        avg_beat_interval = tempo_features.get("avg_beat_interval", 0)
        
        # Format tempo stability
        if "std_beat_interval" in tempo_features and avg_beat_interval > 0:
            std_beat_interval = tempo_features["std_beat_interval"]
            stability_ratio = std_beat_interval / avg_beat_interval if avg_beat_interval else 0
            
            if stability_ratio < 0.05:
                stability = "extremely steady and metronomic"
            elif stability_ratio < 0.1:
                stability = "very steady"
            elif stability_ratio < 0.2:
                stability = "moderately steady"
            elif stability_ratio < 0.3:
                stability = "somewhat unsteady"
            else:
                stability = "highly variable or rubato"
        else:
            stability = "unknown stability"
        
        # Create formatted tempo data
        formatted_tempo = {
            "description": self.tempo_to_descriptive_term(bpm),
            "bpm": bpm,
            "category": tempo_category,
            "beats": beat_count,
            "stability": stability
        }
        
        return formatted_tempo
    
    def format_key_features(self, key_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format key features for LLM understanding.
        
        Args:
            key_features: Dictionary containing key features.
            
        Returns:
            Dictionary with formatted key features.
        """
        key = key_features.get("key", "C")
        mode = key_features.get("mode", "major")
        confidence = key_features.get("confidence", 0.5)
        
        # Describe key confidence
        if confidence > 0.8:
            confidence_desc = "very strong and clear"
        elif confidence > 0.6:
            confidence_desc = "clear and defined"
        elif confidence > 0.4:
            confidence_desc = "somewhat clear"
        elif confidence > 0.2:
            confidence_desc = "ambiguous"
        else:
            confidence_desc = "very ambiguous or atonal"
        
        # Create formatted key data
        formatted_key = {
            "key": key,
            "mode": mode,
            "full_key": f"{key} {mode}",
            "tonality_strength": confidence_desc
        }
        
        # Add emotional associations based on key
        if mode == "major":
            formatted_key["emotional_qualities"] = ["bright", "positive", "uplifting"]
        else:
            formatted_key["emotional_qualities"] = ["dark", "serious", "introspective"]
        
        return formatted_key
    
    def format_timbre_features(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format timbre features for LLM understanding.
        
        Args:
            audio_features: Dictionary containing all audio features.
            
        Returns:
            Dictionary with formatted timbre features.
        """
        # Extract MFCC features (related to timbre)
        mfccs = audio_features.get("mfccs", {})
        spectral = audio_features.get("spectral", {})
        
        # Describe timbre based on spectral centroid
        if "centroid" in spectral:
            centroid = spectral["centroid"].get("normalized_mean", 0.5)
            
            if centroid < 0.3:
                brightness = "dark and warm"
            elif centroid < 0.5:
                brightness = "balanced and neutral"
            elif centroid < 0.7:
                brightness = "bright and clear"
            else:
                brightness = "very bright and sharp"
        else:
            brightness = "unknown brightness"
        
        # Describe complexity based on spectral contrast
        if "spectral_contrast" in audio_features:
            contrast = audio_features["spectral_contrast"].get("overall_contrast", 0.5)
            
            if contrast < 0.3:
                complexity = "simple and pure"
            elif contrast < 0.5:
                complexity = "somewhat complex"
            elif contrast < 0.7:
                complexity = "complex and rich"
            else:
                complexity = "very complex and dense"
        else:
            complexity = "unknown complexity"
        
        # Create formatted timbre data
        formatted_timbre = {
            "brightness": brightness,
            "complexity": complexity,
            "description": f"{brightness}, {complexity} timbre"
        }
        
        return formatted_timbre
    
    def format_harmony_features(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format harmony features for LLM understanding.
        
        Args:
            audio_features: Dictionary containing all audio features.
            
        Returns:
            Dictionary with formatted harmony features.
        """
        # Extract chroma features (related to harmony)
        chroma = audio_features.get("chroma", {})
        
        # Calculate harmonic complexity from chroma variance
        if "cqt" in chroma:
            variance = chroma["cqt"].get("variance", 0.5)
            
            if variance < 0.1:
                complexity = "very simple, possibly using only a few chords"
            elif variance < 0.2:
                complexity = "moderately simple with straightforward chord progressions"
            elif variance < 0.3:
                complexity = "moderately complex with interesting chord progressions"
            else:
                complexity = "harmonically complex with rich chord progressions"
        else:
            complexity = "unknown harmonic complexity"
        
        # Get top notes if available
        top_notes = chroma.get("cqt", {}).get("top_notes", [])
        
        # Create formatted harmony data
        formatted_harmony = {
            "complexity": complexity,
            "top_notes": ", ".join(top_notes) if top_notes else "unknown",
        }
        
        return formatted_harmony
    
    def format_dynamics_features(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format dynamics features for LLM understanding.
        
        Args:
            audio_features: Dictionary containing all audio features.
            
        Returns:
            Dictionary with formatted dynamics features.
        """
        # Extract RMS energy features (related to dynamics)
        rms_energy = audio_features.get("rms_energy", {})
        
        # Get mean energy
        mean_rms = rms_energy.get("mean", 0.1)
        
        # Get dynamic range
        dynamic_range = rms_energy.get("dynamic_range", 0.1)
        
        # Describe dynamics based on mean RMS
        dynamics_desc = self.dynamics_to_descriptive_term(mean_rms)
        
        # Describe dynamic range
        if dynamic_range < 0.1:
            range_desc = "very consistent with minimal dynamic variation"
        elif dynamic_range < 0.2:
            range_desc = "somewhat consistent dynamics with some variation"
        elif dynamic_range < 0.3:
            range_desc = "moderate dynamic range with clear variations"
        elif dynamic_range < 0.4:
            range_desc = "wide dynamic range with significant contrasts"
        else:
            range_desc = "extremely wide dynamic range with dramatic contrasts"
        
        # Get peak count (sudden increases in volume)
        peak_count = rms_energy.get("peak_count", 0)
        
        # Describe peaks
        if peak_count == 0:
            peaks_desc = "no significant volume peaks"
        elif peak_count < 3:
            peaks_desc = "a few distinct volume peaks"
        elif peak_count < 10:
            peaks_desc = "several volume peaks creating emphasis"
        else:
            peaks_desc = "many volume peaks creating a highly dynamic sound"
        
        # Create formatted dynamics data
        formatted_dynamics = {
            "overall_level": dynamics_desc,
            "dynamic_range": range_desc,
            "dynamic_peaks": peaks_desc,
            "description": f"{dynamics_desc} with {range_desc}"
        }
        
        return formatted_dynamics
    
    def format_texture_features(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format texture features for LLM understanding.
        
        Args:
            audio_features: Dictionary containing all audio features.
            
        Returns:
            Dictionary with formatted texture features.
        """
        # Extract zero crossing rate features (related to texture)
        zcr = audio_features.get("zero_crossing_rate", {})
        
        # Get mean zero crossing rate
        mean_zcr = zcr.get("mean", 0.1)
        
        # Describe texture based on zero crossing rate
        texture_desc = self.texture_to_descriptive_term(mean_zcr)
        
        # Get standard deviation of zero crossing rate
        std_zcr = zcr.get("std", 0.01)
        
        # Describe texture consistency
        if std_zcr < 0.01:
            consistency = "very consistent texture throughout"
        elif std_zcr < 0.02:
            consistency = "mostly consistent texture with subtle variations"
        elif std_zcr < 0.05:
            consistency = "moderately varied texture"
        else:
            consistency = "highly varied texture with significant changes"
        
        # Create formatted texture data
        formatted_texture = {
            "quality": texture_desc,
            "consistency": consistency,
            "description": f"{texture_desc} with {consistency}"
        }
        
        return formatted_texture