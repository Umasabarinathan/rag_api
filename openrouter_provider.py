import os
from openai import OpenAI


class OpenRouterProvider:
    """OpenRouter provider for OpenAI client integration."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        Initialize the OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            model_name: Model name to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        # Create OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/Umasabarinathan/yalix-chat-buddy",
                "X-Title": "RAG Implementation with OpenRouter"
            }
        )
    
    def get_client(self) -> OpenAI:
        """Get the configured OpenAI client."""
        return self.client
    
    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model_name


def create_openrouter_provider() -> OpenRouterProvider:
    """
    Create an OpenRouter provider from environment variables.
    
    Returns:
        OpenRouterProvider instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL_NAME")
    
    # Validate required environment variables
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env file")
    if not base_url:
        raise ValueError("OPENAI_API_BASE not set in .env file")
    if not model_name:
        raise ValueError("OPENAI_MODEL_NAME not set in .env file")
    
    return OpenRouterProvider(api_key, base_url, model_name)