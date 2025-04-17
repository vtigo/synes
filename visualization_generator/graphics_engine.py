"""
Graphics engine for creating visual frames using Matplotlib.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.colors import hex2color, to_rgba
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from utils.error_handler import AudioProcessingError


class Animation:
    """
    Base animation class using matplotlib's animation framework.
    
    This class handles the generation of visual frames based on emotional
    interpretation of music. It provides methods for creating shapes, managing
    color palettes, and generating animations.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, dpi: int = 100,
                fps: int = 30, background_color: str = "#000000"):
        """
        Initialize the animation framework.
        
        Args:
            width: Width of the animation in pixels (default: 1280).
            height: Height of the animation in pixels (default: 720).
            dpi: Dots per inch (default: 100).
            fps: Frames per second (default: 30).
            background_color: Background color as hex code (default: "#000000").
        """
        self.logger = logging.getLogger(__name__)
        
        # Set dimensions and timing
        self.width = width
        self.height = height
        self.dpi = dpi
        self.fps = fps
        self.background_color = background_color
        
        # Derived properties
        self.aspect_ratio = width / height
        self.ms_per_frame = 1000 / fps  # Milliseconds per frame
        
        # Initialize figure and axes
        self.fig, self.ax = plt.subplots(
            figsize=(width/dpi, height/dpi),
            dpi=dpi
        )
        
        # Configure axes
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Set background color
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        
        # Store for frames
        self.frames = []
        self.current_frame_objects = []
        
        # Store for color palette
        self.color_palette = []
        self.emotion_color_map = {}
        
        self.logger.info(f"Animation initialized with dimensions {width}x{height}, {fps} fps")
    
    def set_color_palette(self, colors: List[str]) -> None:
        """
        Set the color palette for the animation.
        
        Args:
            colors: List of hex color codes.
        """
        try:
            # Validate colors
            validated_colors = []
            for color in colors:
                try:
                    # Convert to RGB and back to validate
                    rgb = hex2color(color)
                    validated_colors.append(color)
                except ValueError:
                    self.logger.warning(f"Invalid color code: {color}, skipping")
            
            self.color_palette = validated_colors
            self.logger.info(f"Color palette set with {len(validated_colors)} colors")
            
        except Exception as e:
            error_msg = f"Failed to set color palette: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def map_emotions_to_colors(self, emotions: List[str]) -> None:
        """
        Map emotions to colors in the palette.
        
        Args:
            emotions: List of emotion names.
        """
        try:
            if not self.color_palette:
                self.logger.warning("No color palette set, using default colors")
                self.color_palette = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
            
            # Map emotions to colors
            self.emotion_color_map = {}
            
            for i, emotion in enumerate(emotions):
                # Use modulo to cycle through palette if there are more emotions than colors
                color_index = i % len(self.color_palette)
                self.emotion_color_map[emotion] = self.color_palette[color_index]
            
            self.logger.info(f"Mapped {len(emotions)} emotions to colors")
            
        except Exception as e:
            error_msg = f"Failed to map emotions to colors: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def get_color_for_emotion(self, emotion: str) -> str:
        """
        Get the color associated with an emotion.
        
        Args:
            emotion: Name of the emotion.
            
        Returns:
            Hex color code for the emotion.
        """
        emotion = emotion.lower().strip() if isinstance(emotion, str) else str(emotion)
        
        if emotion in self.emotion_color_map:
            return self.emotion_color_map[emotion]
        
        # If emotion not in map, return the first color in the palette or a default
        if self.color_palette:
            return self.color_palette[0]
        
        return "#FFFFFF"  # Default to white if no palette
    
    def create_circle(self, center: Tuple[float, float], radius: float, 
                      color: str, alpha: float = 1.0, zorder: int = 1) -> patches.Circle:
        """
        Create a circle shape.
        
        Args:
            center: (x, y) coordinates of the center, normalized from 0 to 1.
            radius: Radius of the circle, normalized from 0 to 1.
            color: Color of the circle as hex code or name.
            alpha: Transparency of the circle (default: 1.0).
            zorder: Drawing order (default: 1).
            
        Returns:
            Matplotlib Circle patch.
        """
        try:
            circle = patches.Circle(
                center, radius, 
                facecolor=color, 
                alpha=alpha, 
                zorder=zorder
            )
            
            self.ax.add_patch(circle)
            self.current_frame_objects.append(circle)
            
            return circle
            
        except Exception as e:
            error_msg = f"Failed to create circle: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def create_rectangle(self, xy: Tuple[float, float], width: float, height: float, 
                       angle: float = 0, color: str = "#FFFFFF", 
                       alpha: float = 1.0, zorder: int = 1) -> patches.Rectangle:
        """
        Create a rectangle shape.
        
        Args:
            xy: (x, y) coordinates of the bottom-left corner, normalized from 0 to 1.
            width: Width of the rectangle, normalized from 0 to 1.
            height: Height of the rectangle, normalized from 0 to 1.
            angle: Rotation angle in degrees (default: 0).
            color: Color of the rectangle as hex code or name (default: "#FFFFFF").
            alpha: Transparency of the rectangle (default: 1.0).
            zorder: Drawing order (default: 1).
            
        Returns:
            Matplotlib Rectangle patch.
        """
        try:
            rectangle = patches.Rectangle(
                xy, width, height, 
                angle=angle, 
                facecolor=color, 
                alpha=alpha, 
                zorder=zorder
            )
            
            self.ax.add_patch(rectangle)
            self.current_frame_objects.append(rectangle)
            
            return rectangle
            
        except Exception as e:
            error_msg = f"Failed to create rectangle: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def create_line(self, start: Tuple[float, float], end: Tuple[float, float], 
                  color: str = "#FFFFFF", width: float = 1.0, 
                  alpha: float = 1.0, zorder: int = 1) -> plt.Line2D:
        """
        Create a line shape.
        
        Args:
            start: (x, y) coordinates of the start point, normalized from 0 to 1.
            end: (x, y) coordinates of the end point, normalized from 0 to 1.
            color: Color of the line as hex code or name (default: "#FFFFFF").
            width: Width of the line in points (default: 1.0).
            alpha: Transparency of the line (default: 1.0).
            zorder: Drawing order (default: 1).
            
        Returns:
            Matplotlib Line2D object.
        """
        try:
            line = plt.Line2D(
                [start[0], end[0]], [start[1], end[1]],
                color=color,
                linewidth=width,
                alpha=alpha,
                zorder=zorder
            )
            
            self.ax.add_line(line)
            self.current_frame_objects.append(line)
            
            return line
            
        except Exception as e:
            error_msg = f"Failed to create line: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def create_polygon(self, points: List[Tuple[float, float]], color: str = "#FFFFFF", 
                     alpha: float = 1.0, zorder: int = 1) -> patches.Polygon:
        """
        Create a polygon shape.
        
        Args:
            points: List of (x, y) coordinates of the vertices, normalized from 0 to 1.
            color: Color of the polygon as hex code or name (default: "#FFFFFF").
            alpha: Transparency of the polygon (default: 1.0).
            zorder: Drawing order (default: 1).
            
        Returns:
            Matplotlib Polygon patch.
        """
        try:
            polygon = patches.Polygon(
                points,
                facecolor=color,
                alpha=alpha,
                zorder=zorder
            )
            
            self.ax.add_patch(polygon)
            self.current_frame_objects.append(polygon)
            
            return polygon
            
        except Exception as e:
            error_msg = f"Failed to create polygon: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def create_star(self, center: Tuple[float, float], radius: float, points: int = 5, 
                  color: str = "#FFFFFF", alpha: float = 1.0, 
                  zorder: int = 1) -> patches.Polygon:
        """
        Create a star shape.
        
        Args:
            center: (x, y) coordinates of the center, normalized from 0 to 1.
            radius: Outer radius of the star, normalized from 0 to 1.
            points: Number of points in the star (default: 5).
            color: Color of the star as hex code or name (default: "#FFFFFF").
            alpha: Transparency of the star (default: 1.0).
            zorder: Drawing order (default: 1).
            
        Returns:
            Matplotlib Polygon patch representing a star.
        """
        try:
            # Calculate vertices for the star
            inner_radius = radius / 2  # Inner radius is half of outer radius
            theta = np.linspace(0, 2*np.pi, 2*points, endpoint=False)
            
            # Alternate between inner and outer radius
            r = np.ones(2*points)
            r[1::2] = inner_radius / radius
            
            # Convert to cartesian coordinates
            x = center[0] + radius * r * np.cos(theta)
            y = center[1] + radius * r * np.sin(theta)
            
            # Create polygon vertices
            vertices = np.column_stack([x, y])
            
            # Create polygon
            star = patches.Polygon(
                vertices,
                facecolor=color,
                alpha=alpha,
                zorder=zorder
            )
            
            self.ax.add_patch(star)
            self.current_frame_objects.append(star)
            
            return star
            
        except Exception as e:
            error_msg = f"Failed to create star: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def create_shape(self, shape_type: str, params: Dict[str, Any]) -> Any:
        """
        Create a shape based on type and parameters.
        
        Args:
            shape_type: Type of shape to create.
            params: Dictionary of parameters for the shape.
            
        Returns:
            Matplotlib patch or artist.
            
        Raises:
            AudioProcessingError: If the shape type is unknown.
        """
        try:
            shape_type = shape_type.lower().strip()
            
            if shape_type == "circle":
                return self.create_circle(
                    center=params.get("center", (0.5, 0.5)),
                    radius=params.get("radius", 0.1),
                    color=params.get("color", "#FFFFFF"),
                    alpha=params.get("alpha", 1.0),
                    zorder=params.get("zorder", 1)
                )
            
            elif shape_type == "rectangle":
                return self.create_rectangle(
                    xy=params.get("xy", (0.4, 0.4)),
                    width=params.get("width", 0.2),
                    height=params.get("height", 0.2),
                    angle=params.get("angle", 0),
                    color=params.get("color", "#FFFFFF"),
                    alpha=params.get("alpha", 1.0),
                    zorder=params.get("zorder", 1)
                )
            
            elif shape_type == "line":
                return self.create_line(
                    start=params.get("start", (0.3, 0.3)),
                    end=params.get("end", (0.7, 0.7)),
                    color=params.get("color", "#FFFFFF"),
                    width=params.get("width", 1.0),
                    alpha=params.get("alpha", 1.0),
                    zorder=params.get("zorder", 1)
                )
            
            elif shape_type == "polygon":
                return self.create_polygon(
                    points=params.get("points", [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]),
                    color=params.get("color", "#FFFFFF"),
                    alpha=params.get("alpha", 1.0),
                    zorder=params.get("zorder", 1)
                )
            
            elif shape_type == "star":
                return self.create_star(
                    center=params.get("center", (0.5, 0.5)),
                    radius=params.get("radius", 0.1),
                    points=params.get("points", 5),
                    color=params.get("color", "#FFFFFF"),
                    alpha=params.get("alpha", 1.0),
                    zorder=params.get("zorder", 1)
                )
            
            else:
                error_msg = f"Unknown shape type: {shape_type}"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, "E004")
                
        except AudioProcessingError:
            # Re-raise AudioProcessingError
            raise
            
        except Exception as e:
            error_msg = f"Failed to create shape: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def clear_frame(self) -> None:
        """Clear all objects from the current frame."""
        for obj in self.current_frame_objects:
            try:
                obj.remove()
            except:
                pass
        
        self.current_frame_objects = []
    
    def save_frame(self) -> np.ndarray:
        """
        Save the current frame as a numpy array.
        
        Returns:
            Numpy array containing the frame image data.
        """
        try:
            # Draw the figure to update
            self.fig.canvas.draw()
            
            # Get the ARGB buffer from the figure
            w, h = int(self.fig.get_figwidth() * self.dpi), int(self.fig.get_figheight() * self.dpi)
            buffer = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            buffer = buffer.reshape((h, w, 4))
            
            # Convert ARGB to RGB by removing alpha channel
            rgb_buffer = buffer[:, :, 1:4]
            
            return rgb_buffer.copy()
            
        except Exception as e:
            error_msg = f"Failed to save frame: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def generate_frames_from_parameters(self, visualization_params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate frames based on visualization parameters.
        
        Args:
            visualization_params: Dictionary containing visualization parameters.
            
        Returns:
            List of numpy arrays, each containing a frame image.
            
        Raises:
            AudioProcessingError: If frame generation fails.
        """
        try:
            self.logger.info("Generating frames from visualization parameters")
            
            # Extract parameters
            emotional_journey = visualization_params.get("emotional_journey", [])
            color_palette = visualization_params.get("color_palette", [])
            frames_data = visualization_params.get("frames", [])
            
            # Set color palette
            self.set_color_palette(color_palette)
            
            # Map emotions to colors
            self.map_emotions_to_colors(emotional_journey)
            
            # Generate frames
            generated_frames = []
            
            for frame_data in frames_data:
                # Clear previous frame
                self.clear_frame()
                
                # Extract frame information
                timestamp = frame_data.get("timestamp", 0)
                dominant_emotion = frame_data.get("dominant_emotion", "neutral")
                shapes_data = frame_data.get("shapes", [])
                colors = frame_data.get("colors", [])
                
                self.logger.debug(f"Generating frame at timestamp: {timestamp:.2f}s, emotion: {dominant_emotion}")
                
                # If no colors specified in frame, use emotion color
                if not colors:
                    colors = [self.get_color_for_emotion(dominant_emotion)]
                
                # Create shapes
                for shape_data in shapes_data:
                    if not isinstance(shape_data, dict):
                        continue
                    
                    shape_type = shape_data.get("type", "circle")
                    
                    # Set color for shape (use first color if none specified)
                    if "color" not in shape_data and colors:
                        shape_data["color"] = colors[0]
                    
                    # Create the shape
                    self.create_shape(shape_type, shape_data)
                
                # If no shapes in frame data, create a default shape
                if not shapes_data:
                    self.create_circle(
                        center=(0.5, 0.5),
                        radius=0.2,
                        color=colors[0] if colors else "#FFFFFF"
                    )
                
                # Save frame
                frame = self.save_frame()
                generated_frames.append(frame)
            
            self.logger.info(f"Generated {len(generated_frames)} frames")
            return generated_frames
            
        except AudioProcessingError:
            # Re-raise AudioProcessingError
            raise
            
        except Exception as e:
            error_msg = f"Failed to generate frames: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def generate_test_animation(self, duration: float = 5.0) -> List[np.ndarray]:
        """
        Generate a simple test animation to verify the framework.
        
        Args:
            duration: Duration of the animation in seconds (default: 5.0).
            
        Returns:
            List of numpy arrays, each containing a frame image.
        """
        try:
            self.logger.info(f"Generating test animation ({duration}s)")
            
            # Set test color palette
            self.set_color_palette([
                "#FF0000", "#00FF00", "#0000FF", 
                "#FFFF00", "#FF00FF", "#00FFFF"
            ])
            
            # Map test emotions
            self.map_emotions_to_colors([
                "happy", "sad", "angry", "calm", "excited", "peaceful"
            ])
            
            # Calculate number of frames
            num_frames = int(duration * self.fps)
            
            # Generate frames
            frames = []
            
            for i in range(num_frames):
                # Clear previous frame
                self.clear_frame()
                
                # Calculate time and parameters
                t = i / (num_frames - 1) if num_frames > 1 else 0
                
                # Animated parameters
                radius = 0.1 + 0.05 * np.sin(t * 2 * np.pi)
                x = 0.5 + 0.3 * np.sin(t * 2 * np.pi)
                y = 0.5 + 0.3 * np.cos(t * 2 * np.pi)
                angle = t * 360
                
                # Get color (cycle through palette)
                color_index = int(t * len(self.color_palette))
                color = self.color_palette[color_index % len(self.color_palette)]
                
                # Create shapes
                self.create_circle((x, y), radius, color)
                
                self.create_rectangle(
                    (0.5 - radius/2, 0.5 - radius/2),
                    radius, radius, angle, color
                )
                
                self.create_star(
                    (0.5, 0.5), 0.3, 5, "#FFFFFF", 0.5
                )
                
                # Save frame
                frame = self.save_frame()
                frames.append(frame)
            
            self.logger.info(f"Generated {len(frames)} test frames")
            return frames
            
        except Exception as e:
            error_msg = f"Failed to generate test animation: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E004")
    
    def close(self) -> None:
        """Close the animation and release resources."""
        plt.close(self.fig)
        self.logger.info("Animation closed and resources released")