import logging
from LLM.backend.gemini import GeminiBackend
from LLM.backend.huggingface import LocalHFBackend
from LLM.backend.model import LLMBackend

logger = logging.getLogger(__name__)

def create_backend(backend_type: str, hyperparams: dict | None = {}) -> LLMBackend:
    backend_map = {
        "local": LocalHFBackend,
        "gemini": GeminiBackend,
    }

    logger.info(f"Creating LLM backend '{backend_type}' with hyper-parameters: {hyperparams}")
    return backend_map[backend_type](hyperparams)