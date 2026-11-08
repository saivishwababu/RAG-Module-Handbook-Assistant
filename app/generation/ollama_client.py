"""Ollama API client with exact model tag discovery and robust streaming."""

import json
import logging
from typing import List, Dict, Any, Optional, Iterator
import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if local Ollama server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """
        List all installed Ollama models with complete tags.
        Does NOT strip tags, preventing collision between e.g. llama3.2:1b and llama3.2:3b.
        """
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Failed to query Ollama models: {e}")
        return []

    def select_best_model(self, preferred_models: List[str]) -> Optional[str]:
        """
        Match preferred models against installed models using exact tags first,
        then fallback to family prefix if safe.
        """
        installed = self.list_installed_models()
        if not installed:
            return None

        # Pass 1: Exact tag match (e.g. 'llama3.2:latest' == 'llama3.2:latest')
        for pref in preferred_models:
            for inst in installed:
                if pref == inst:
                    return inst
                # Handle implied ':latest'
                if not pref.endswith(":latest") and f"{pref}:latest" == inst:
                    return inst
                if pref.endswith(":latest") and pref.replace(":latest", "") == inst:
                    return inst

        # Pass 2: Base name match (e.g. 'llama3.2' matches 'llama3.2:3b')
        for pref in preferred_models:
            pref_base = pref.split(":")[0]
            for inst in installed:
                if inst.split(":")[0] == pref_base:
                    return inst

        # Pass 3: Return first installed model
        return installed[0] if installed else None

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Iterator[str] | str:
        """Execute chat inference against Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            },
        }

        try:
            if stream:
                return self._stream_response(payload)
            else:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. Ensure 'ollama serve' is running."
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama inference timed out after {self.timeout}s.") from e

    def _stream_response(self, payload: Dict[str, Any]) -> Iterator[str]:
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
