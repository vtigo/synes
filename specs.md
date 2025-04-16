# Comprehensive Specification: Music-to-Visual Emotion Interpreter System

## 1. System Overview

This CLI application transforms MP3 audio files into visual representations (MP4) based on emotional interpretation via an LLM. The system analyzes musical features from the audio, processes them through an LLM to generate emotional interpretations, and creates abstract/geometric visualizations that correspond to the emotional journey of the music.

## 2. Technical Architecture

### 2.1 Core Components

1. **Audio Analysis Engine**
   - Input processor: Handles MP3 file reading
   - Feature extractor: Processes audio features using Librosa
   - Data formatter: Prepares extracted features for LLM input

2. **LLM Interpretation Engine**
   - Model: Llama 3 (local implementation)
   - Prompt manager: Creates fixed templates for LLM 
   - Response processor: Parses LLM output into visualization instructions

3. **Visualization Generator**
   - Graphics engine: Matplotlib animation system
   - Rendering pipeline: Converts LLM instructions to visual elements
   - Output processor: Combines visuals with original audio using MoviePy

### 2.2 Data Flow

```
[MP3 Input] → [Audio Analysis] → [Feature Extraction] → [LLM Processing] 
→ [Visualization Mapping] → [Frame Generation] → [Video Rendering] → [MP4 Output]
```

## 3. Technical Requirements

### 3.1 Audio Processing Requirements

- **Library**: Librosa
- **Features to Extract**:
  - Tempo and beat detection (BPM, beat positions)
  - Key and mode detection
  - Spectral features (centroid, rolloff, flux)
  - Chroma features (harmonic content)
  - MFCCs (timbre characteristics)
  - Onset detection (articulation)
  - RMS energy (dynamics)
  - Zero crossing rate (texture)
  - Spectral contrast (instrumentation detection)
  - Tempo variation measurements
  - Silence ratio analysis

### 3.2 LLM Implementation

- **Model**: Llama 3 (8B or 13B parameters recommended)
- **Inference**: Local deployment
- **Input Format**: Fixed JSON template containing all audio features
- **Output Format**: Structured JSON with visualization parameters
- **Memory Requirements**: At least 16GB RAM
- **Processing Mode**: Analyze entire track before visualization

### 3.3 Visualization Engine

- **Libraries**: Matplotlib for generation, MoviePy for encoding
- **Resolution**: 1280x720 (720p)
- **Frame Rate**: 30fps
- **Color Depth**: 24-bit
- **Visual Elements**:
  - Abstract shapes
  - Geometric patterns
  - Color palettes
  - Movement patterns
  - Texture variations

### 3.4 System Requirements

- **Language**: Python 3.9+
- **Dependencies**: Librosa, Matplotlib, MoviePy, PyTorch/Transformers (for Llama)
- **Processing Time**: Maximum 5 minutes per song
- **Output Format**: MP4 container with H.264 video codec and original audio

## 4. Functional Specifications

### 4.1 Command Line Interface

```
music_visualizer.py --input <mp3_file_path> --output <mp4_file_path> [--model <model_path>]
```

### 4.2 Audio Analysis Process

1. Load MP3 file and convert to raw waveform
2. Extract all specified audio features
3. Segment analysis for temporal features
4. Normalize all features to consistent scales
5. Generate comprehensive audio fingerprint

### 4.3 LLM Prompt Template Structure

```
You are an emotional interpreter for music. Analyze the following audio features
and create a visualization mapping that represents the emotional journey of this music
through abstract shapes and geometric patterns.

Audio Features:
- Tempo: {tempo_data}
- Key: {key_data}
- Timbre: {timbre_data}
- [All other extracted features]

Create a visualization that maps these features to visual elements including colors,
shapes, patterns, and movements. Your output should be a frame-by-frame description
of how the visualization evolves.

Output your response as a JSON object with the following structure:
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
```

### 4.4 Visualization Mapping

- Tempo → Animation speed
- Key → Base shape configuration
- Timbre → Texture and opacity
- Frequency spectrum → Color distribution
- Dynamics → Size variations
- Harmony → Pattern complexity
- Rhythm → Movement patterns

## 5. Error Handling

### 5.1 Input Validation

- Validate MP3 file format and integrity
- Check minimum audio quality requirements (bit rate, sample rate)
- Ensure file is not DRM protected or corrupted

### 5.2 Processing Errors

- Handle Librosa extraction failures with specific error codes
- Implement timeout protocols for LLM inference
- Create fallback visualization parameters if LLM fails

### 5.3 Output Validation

- Verify MP4 encoding success
- Validate audio-visual synchronization
- Implement frame dropping strategy for performance issues

### 5.4 Error Codes

| Code | Description | Recovery Action |
|------|-------------|----------------|
| E001 | Invalid MP3 file | Request new file |
| E002 | Feature extraction failed | Try simplified extraction |
| E003 | LLM inference timeout | Use cached responses/fallback |
| E004 | Visualization generation error | Use template visuals |
| E005 | Encoding failure | Retry with lower resolution |

## 6. Testing Strategy

### 6.1 Unit Testing

- Test each module independently:
  - Audio processing accuracy
  - Feature extraction validation
  - LLM response parsing
  - Visualization generation

### 6.2 Integration Testing

- Test full pipeline with various music genres
- Validate processing time constraints
- Verify emotional consistency between similar songs

### 6.3 Performance Testing

- Process various length audio files
- Monitor memory usage during LLM inference
- Validate frame generation speed

### 6.4 Test Dataset

Create a test suite with:
- Short samples (30s) representing distinct emotions
- Full-length songs across multiple genres
- Edge cases (extremely fast/slow tempo, unusual harmonics)

## 7. Implementation Plan

### 7.1 Phase 1: Audio Analysis Implementation

1. Implement MP3 loading and validation
2. Develop feature extraction pipeline
3. Create feature normalization system
4. Test with sample audio files

### 7.2 Phase 2: LLM Integration

1. Set up local Llama 3 deployment
2. Implement prompt template system
3. Create response parsing module
4. Test with sample audio features

### 7.3 Phase 3: Visualization Engine

1. Develop base visualization framework
2. Implement shape and color generation
3. Create animation system for transitions
4. Test with sample LLM outputs

### 7.4 Phase 4: Integration and Optimization

1. Connect all system components
2. Optimize processing pipeline
3. Implement error handling
4. Create CLI interface

## 8. Technical Dependencies

- **Python Libraries**:
  - Librosa >= 0.9.2
  - NumPy >= 1.20.0
  - Matplotlib >= 3.5.0
  - MoviePy >= 1.0.3
  - PyTorch >= 1.13.0
  - Transformers >= 4.25.0
  - JSON >= 2.0.9
  - ArgParse >= 1.4.0

- **External Tools**:
  - FFMPEG (for MoviePy)
  - CUDA (optional, for GPU acceleration)
