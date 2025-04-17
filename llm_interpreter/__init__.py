"""
LLM interpreter module for translating audio features into emotional interpretations.

This module handles the interaction with the Llama 3 LLM model to convert technical
audio features into emotional interpretations and visualization instructions.
"""

from llm_interpreter.prompt_manager import PromptManager
from llm_interpreter.llm_processor import LLMProcessor
from llm_interpreter.llm_interface import LLMInterface
from llm_interpreter.response_processor import ResponseParser

__all__ = ['PromptManager', 'LLMProcessor', 'LLMInterface', 'ResponseParser']