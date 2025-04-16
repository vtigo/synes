# In llm_interface.py
"""
LLM interface for handling the Llama 3 model and inference.
"""
import logging
import time
import json
from typing import Dict, Any, Optional

# Import the Ollama library - make sure to install it first with: pip install ollama
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from utils.error_handler import AudioProcessingError


class LLMInterface:
    """
    Interface for the Llama 3 model and inference using Ollama.
    """
    
    def __init__(self, model_path: Optional[str] = None, timeout: int = 60, 
                 temperature: float = 0.7, max_new_tokens: int = 4096):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path or name of the model to use (default: llama3).
            timeout: Maximum time in seconds to wait for model inference (default: 60).
            temperature: Sampling temperature for generation (default: 0.7).
            max_new_tokens: Maximum number of tokens to generate (default: 4096).
            
        Raises:
            AudioProcessingError: If the dependencies cannot be loaded.
        """
        self.logger = logging.getLogger(__name__)
        
        # Store parameters
        self.model_name = model_path or "llama3"
        self.timeout = timeout
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Check if Ollama is installed
        if not HAS_OLLAMA:
            self.logger.warning("Ollama package not found. Install with 'pip install ollama'")
        
        self.logger.info(f"Initialized LLM interface with model: {self.model_name}")
    
    def load_model(self) -> bool:
        """
        Check if Ollama service is running and the model is available.
        
        Returns:
            Boolean indicating if the model is available.
            
        Raises:
            AudioProcessingError: If Ollama service is not available.
        """
        try:
            if not HAS_OLLAMA:
                error_msg = "Ollama package is required but not installed"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, "E003")
            
            # List available models to check connection
            models = ollama.list()
            
            # Check if our model is in the list
            model_exists = any(model.get('name', '').startswith(self.model_name) 
                              for model in models.get('models', []))
            
            if not model_exists:
                self.logger.warning(f"Model '{self.model_name}' not found locally. "
                                  f"You may need to pull it with 'ollama pull {self.model_name}'")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Ollama service: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003") from e
    
    def process(self, prompt: str, stream: bool = False) -> str:
        """
        Process a prompt through the LLM.
        
        Args:
            prompt: String containing the prompt for the LLM.
            stream: Whether to stream the output (default: False).
            
        Returns:
            String containing the LLM response.
            
        Raises:
            AudioProcessingError: If LLM processing fails or times out.
        """
        self.logger.info("Processing prompt through Ollama")
        
        # Check if Ollama is installed
        if not HAS_OLLAMA:
            error_msg = "Ollama package is required but not installed"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003")
        
        try:
            # Try to ensure the model is available
            self.load_model()
            
            start_time = time.time()
            
            # Process with Ollama
            if stream:
                # Handle streaming response
                response_text = ""
                stream = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=True,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_new_tokens
                    }
                )
                
                for chunk in stream:
                    response_text += chunk.get('response', '')
                    
                    # Check for timeout
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError("Model inference timed out")
            else:
                # Handle non-streaming response
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_new_tokens
                    }
                )
                response_text = response.get('response', '')
            
            processing_time = time.time() - start_time
            self.logger.info(f"LLM processing complete in {processing_time:.2f} seconds")
            
            return response_text
            
        except TimeoutError:
            error_msg = "LLM inference timed out"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003")
            
        except Exception as e:
            error_msg = f"LLM processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, "E003") from e
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response.
        
        Args:
            response: String containing the LLM response.
            
        Returns:
            Dictionary containing parsed JSON or None if extraction fails.
        """
        self.logger.info("Extracting JSON from LLM response")
        
        try:
            # Try to find JSON object in the response
            # Look for opening and closing braces
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start == -1 or json_end == -1 or json_end <= json_start:
                self.logger.warning("No valid JSON found in response")
                return None
            
            # Extract the JSON part
            json_str = response[json_start:json_end+1]
            
            # Parse the JSON
            parsed_json = json.loads(json_str)
            
            self.logger.info("Successfully extracted JSON from response")
            return parsed_json
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from response: {str(e)}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error extracting JSON from response: {str(e)}")
            return None