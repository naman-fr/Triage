"""
LLM Client Module — Multi-provider wrapper for deterministic and multimodal LLM calls.
Supports Google Gemini, OpenAI, Anthropic, Groq, and Mock fallback.
Includes robust API error handlers, session-level provider blacklisting, and fallback chaining.
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
    Extracts image URLs, Base64 payloads, or local file paths from the text prompt,
    downloads or reads and encodes them, and returns (cleaned_text, image_blocks).
    Used for multimodal analysis of screenshots, error logs, etc.
    """
    from pathlib import Path
    cleaned_text = text
    image_blocks = []
    
    # 1. Search for markdown images: ![alt](path)
    markdown_pattern = re.compile(r'!\[.*?\]\((.*?)\)', re.IGNORECASE)
    md_matches = markdown_pattern.findall(text)
    
    # 2. Search for raw http/https links or base64 data URIs
    raw_matches = IMAGE_URL_PATTERN.findall(text)
    
    # 3. Combine matches and filter duplicates
    all_targets = list(set(md_matches + raw_matches))
    
    # 4. If no matches were found, look for plain word paths ending with image extensions
    if not all_targets:
        plain_path_pattern = re.compile(r'(\b\S+\.(?:png|jpg|jpeg|gif|webp|svg)\b)', re.IGNORECASE)
        all_targets = list(set(plain_path_pattern.findall(text)))

    for target in all_targets:
        target_str = target.strip()
        if not target_str:
            continue
            
        # Clean target_str of surrounding quotes if any
        target_str = target_str.strip('"\'()')
        
        try:
            # Case A: Base64 data URI
            if target_str.startswith("data:image/"):
                header, base64_data = target_str.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data.strip()
                    }
                })
                cleaned_text = cleaned_text.replace(target, "")
                
            # Case B: Remote URL
            elif target_str.startswith("http://") or target_str.startswith("https://"):
                res = requests.get(target_str, timeout=10)
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
                cleaned_text = cleaned_text.replace(target, "")
                
            # Case C: Local file path
            else:
                # Resolve paths (absolute or relative to current repository root)
                local_path = Path(target_str)
                if not local_path.is_absolute():
                    # Check relative to current working dir or repo root
                    from config import REPO_ROOT
                    candidate_paths = [
                        Path.cwd() / local_path,
                        REPO_ROOT / local_path,
                        local_path
                    ]
                    for cp in candidate_paths:
                        if cp.exists() and cp.is_file():
                            local_path = cp
                            break
                            
                if local_path.exists() and local_path.is_file():
                    # Determine media type based on suffix
                    suffix = local_path.suffix.lower().lstrip(".")
                    media_type = f"image/{suffix}" if suffix != "svg" else "image/svg+xml"
                    if suffix in ["jpg", "jpeg"]:
                        media_type = "image/jpeg"
                    
                    with open(local_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                        
                    image_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data
                        }
                    })
                    cleaned_text = cleaned_text.replace(target, "")
                    print(f"[LLM Client] Successfully loaded local visual attachment: {local_path}")
        except Exception as e:
            print(f"[LLM Client] Multimodal error processing target {target_str[:50]}: {e}")
            
    return cleaned_text.strip(), image_blocks


class LLMClient:
    """
    Unified LLM client that auto-detects the available provider,
    manages fallback providers in case of API errors (e.g. rate limit, billing),
    and falls back to a deterministic Mock generator if all else fails.
    """

    # Class-level state to remember permanently failed providers across instances
    disabled_providers = set()

    def __init__(self):
        self.provider = get_llm_provider()
        self.model = LLM_MODELS[self.provider]
        self._client = None
        self._init_client()
        print(f"[LLM] Primary provider: {self.provider} ({self.model})")

    def _init_client(self):
        """Initialize the default client."""
        if self.provider == "google" and "google" not in self.disabled_providers:
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
        elif self.provider == "openai" and "openai" not in self.disabled_providers:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "anthropic" and "anthropic" not in self.disabled_providers:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        elif self.provider == "groq" and "groq" not in self.disabled_providers:
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)
        else:
            self._client = "mock"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a response from the LLM.
        Iterates through candidate providers in priority order if failures occur (e.g. credit/billing limits).
        """
        # Determine candidate chain: Primary -> Others (with keys) -> Mock
        providers_to_try = []
        if self.provider not in self.disabled_providers:
            providers_to_try.append(self.provider)

        all_options = ["anthropic", "openai", "groq", "google"]
        for opt in all_options:
            if opt not in providers_to_try and opt not in self.disabled_providers:
                # Add only if key exists in env
                if opt == "anthropic" and ANTHROPIC_API_KEY:
                    providers_to_try.append(opt)
                elif opt == "openai" and OPENAI_API_KEY:
                    providers_to_try.append(opt)
                elif opt == "groq" and GROQ_API_KEY:
                    providers_to_try.append(opt)
                elif opt == "google" and GOOGLE_API_KEY:
                    providers_to_try.append(opt)
        if "mock" not in providers_to_try:
            providers_to_try.append("mock")

        for prov in providers_to_try:
            prov_model = LLM_MODELS[prov]
            for attempt in range(max_retries):
                try:
                    if prov == "google":
                        import google.generativeai as genai
                        genai.configure(api_key=GOOGLE_API_KEY)
                        client = genai.GenerativeModel(
                            model_name=prov_model,
                            generation_config={
                                "temperature": LLM_TEMPERATURE,
                                "max_output_tokens": LLM_MAX_TOKENS,
                                "response_mime_type": "application/json",
                            },
                        )
                        full_prompt = f"{system_prompt}\n\n{user_prompt}"
                        response = client.generate_content(full_prompt)
                        return response.text
                    elif prov == "openai":
                        from openai import OpenAI
                        client = OpenAI(api_key=OPENAI_API_KEY)
                        response = client.chat.completions.create(
                            model=prov_model,
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
                    elif prov == "anthropic":
                        from anthropic import Anthropic
                        client = Anthropic(api_key=ANTHROPIC_API_KEY)
                        cleaned_prompt, image_blocks = _extract_and_download_images(user_prompt)
                        if image_blocks:
                            content = image_blocks + [{"type": "text", "text": cleaned_prompt}]
                        else:
                            content = user_prompt
                        response = client.messages.create(
                            model=prov_model,
                            max_tokens=LLM_MAX_TOKENS,
                            system=system_prompt,
                            messages=[
                                {"role": "user", "content": content},
                            ],
                            temperature=LLM_TEMPERATURE,
                        )
                        return response.content[0].text
                    elif prov == "groq":
                        from groq import Groq
                        client = Groq(api_key=GROQ_API_KEY)
                        response = client.chat.completions.create(
                            model=prov_model,
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
                    elif prov == "mock":
                        return self._generate_mock(system_prompt, user_prompt)
                except Exception as e:
                    print(f"[LLM] {prov} error (attempt {attempt + 1}/{max_retries}): {e}")
                    # If it is a billing/auth error, blacklist this provider for the session
                    err_msg = str(e).lower()
                    if any(phrase in err_msg for phrase in ["credit balance", "billing", "invalid api key", "unauthorized", "api_key_invalid"]):
                        print(f"[LLM] Permanent error detected for provider '{prov}'. Disabling it for this session.")
                        LLMClient.disabled_providers.add(prov)
                        break  # Stop retrying this provider and switch immediately to fallback
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            # Log provider failure and switch to the next fallback
            print(f"[LLM] Fallback: Switching from '{prov}' to next candidate.")

        return ""

    def _generate_mock(self, system_prompt: str, user_prompt: str) -> str:
        """Simulate LLM response using simple heuristics when API keys are missing."""
        status = "replied"
        product_area = "general_support"
        request_type = "product_issue"
        risk_level = "low"
        actions_taken = []
        
        user_prompt_lower = user_prompt.lower()
        
        if "visa" in user_prompt_lower:
            product_area = "visa_support"
        elif "claude" in user_prompt_lower:
            product_area = "claude_support"
        elif "devplatform" in user_prompt_lower or "hackerrank" in user_prompt_lower:
            product_area = "devplatform_support"
            
        is_injection = "prompt injection" in user_prompt_lower or "[system override]" in user_prompt_lower or "dan mode" in user_prompt_lower or "override safety" in user_prompt_lower
        has_pii = "pii detected" in user_prompt_lower
        
        if is_injection:
            response = (
                "⚠️ Mock Mode: I've detected that this message contains instructions "
                "attempting to override my normal operation. I cannot comply with such requests. "
                "Please configure active API keys (e.g. ANTHROPIC_API_KEY) in your .env to enable live mode."
            )
            request_type = "invalid"
            justification = "Mock agent detected potential prompt injection attack."
        else:
            if "refund" in user_prompt_lower:
                response = (
                    "⚠️ Mock Mode: I understand you are requesting a refund. "
                    "According to our support policy, refunds can only be processed for eligible cases after identity verification. "
                    "I will initiate identity verification to proceed."
                )
                actions_taken = [
                    {"action": "verify_identity", "parameters": {}}
                ]
                justification = "Mock agent handling refund request: verification initiated."
            elif "compromise" in user_prompt_lower or "hacked" in user_prompt_lower:
                response = (
                    "⚠️ Mock Mode: It appears your account may have been compromised. "
                    "For safety, I am locking your account and escalating to a human specialist."
                )
                status = "escalated"
                risk_level = "high"
                actions_taken = [
                    {"action": "lock_account", "parameters": {}},
                    {"action": "escalate_to_human", "parameters": {"priority": "urgent", "department": "security"}}
                ]
                justification = "Mock agent handling account compromise: locking and escalating."
            else:
                response = (
                    "⚠️ Mock Mode: Thank you for contacting support! No active API keys "
                    "were detected in your environment. Here is a simulated response based on the retrieved local documentation. "
                    "Please set your keys in the .env file to enable live LLM processing."
                )
                justification = "Mock mode active. Grounded on local document indexing."

        mock_output = {
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
            "confidence_score": 0.95 if not is_injection else 0.99,
            "risk_level": risk_level,
            "actions_taken": actions_taken,
            "source_documents": ""
        }
        return json.dumps(mock_output)

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

        try:
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
            json_match = _extract_json(raw)
            if json_match:
                return json_match

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
