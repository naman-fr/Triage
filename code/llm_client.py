"""
LLM Client Module — Multi-provider wrapper for deterministic and multimodal LLM calls.
Supports Google Gemini, OpenAI, Anthropic, and Groq.
"""

import json
import time
import re
import base64
import requests
from typing import Optional, Tuple, List, Dict

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    LLM_TEMPERATURE, LLM_SEED, LLM_MAX_TOKENS, LLM_MODELS,
    get_llm_provider,
)

IMAGE_URL_PATTERN = re.compile(
    r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?\S+)?|data:image/(?:png|jpg|jpeg|gif|webp);base64,\S+)',
    re.IGNORECASE
)


def _extract_and_download_images(text: str) -> Tuple[str, List[Dict]]:
    """
    Extracts image URLs or Base64 payloads from the text prompt,
    downloads and encodes them, and returns (cleaned_text, image_blocks).
    Used for multimodal analysis of screenshots, error logs, etc.
    """
    cleaned_text = text
    image_blocks = []
    
    matches = IMAGE_URL_PATTERN.findall(text)
    for match in matches:
        cleaned_text = cleaned_text.replace(match, "")
        try:
            if match.startswith("data:image/"):
                header, base64_data = match.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data.strip()
                    }
                })
            else:
                res = requests.get(match, timeout=10)
                res.raise_for_status()
                content_type = res.headers.get("Content-Type", "image/png")
                if "image" not in content_type:
                    content_type = "image/png"
                img_data = base64.b64encode(res.content).decode("utf-8")
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": img_data
                    }
                })
        except Exception as e:
            print(f"[LLM Client] Multimodal error processing image {match[:50]}: {e}")
            
    return cleaned_text.strip(), image_blocks


class LLMClient:
    """
    Unified LLM client that auto-detects the available provider
    and provides a consistent interface for structured output generation.
    Supports multimodal input handling for visual support tickets.
    """

    def __init__(self):
        self.provider = get_llm_provider()
        self.model = LLM_MODELS[self.provider]
        self._client = None
        self._init_client()
        print(f"[LLM] Using provider: {self.provider} ({self.model})")

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": LLM_TEMPERATURE,
                    "max_output_tokens": LLM_MAX_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions
            user_prompt: User message/query
            max_retries: Number of retries on failure

        Returns:
            Raw string response from the LLM
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "google":
                    return self._generate_google(system_prompt, user_prompt)
                elif self.provider == "openai":
                    return self._generate_openai(system_prompt, user_prompt)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(system_prompt, user_prompt)
                elif self.provider == "groq":
                    return self._generate_groq(system_prompt, user_prompt)
            except Exception as e:
                print(f"[LLM] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return ""

    def _generate_google(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Google Gemini."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._client.generate_content(full_prompt)
        return response.text

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Anthropic Claude (with multimodal image support)."""
        cleaned_prompt, image_blocks = _extract_and_download_images(user_prompt)
        
        if image_blocks:
            # Construct content as a list of blocks for multimodal sonnet
            content = image_blocks + [{"type": "text", "text": cleaned_prompt}]
        else:
            content = user_prompt

        response = self._client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return response.content[0].text

    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Groq."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Generate a structured JSON response from the LLM.

        Returns:
            Parsed JSON dictionary
        """
        raw = self.generate(system_prompt, user_prompt)

        # Try to parse JSON from the response
        try:
            # Handle responses wrapped in markdown code blocks
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = _extract_json(raw)
            if json_match:
                return json_match

            # Return a safe default
            print(f"[LLM] Warning: Could not parse JSON from response")
            return {}


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
