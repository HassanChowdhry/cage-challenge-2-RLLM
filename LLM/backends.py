"""
Modular LLM backend system for supporting multiple providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.0, max_tokens: int = 200):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Convenience method to call generate directly."""
        return self.generate(prompt, **kwargs)


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: str = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: str = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ""


class GoogleGeminiBackend(LLMBackend):
    """Google Gemini API backend."""
    
    def __init__(self, model_name: str = "gemini-pro", api_key: str = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                **kwargs
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            return ""


class LocalHFBackend(LLMBackend):
    """Local HuggingFace model backend."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Local model generation error: {e}")
            return ""


def create_backend(backend_type: str, **kwargs) -> LLMBackend:
    """Factory function to create LLM backends."""
    backend_map = {
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "gemini": GoogleGeminiBackend,
        "local": LocalHFBackend,
    }
    
    if backend_type not in backend_map:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backend_map.keys())}")
    
    return backend_map[backend_type](**kwargs) 