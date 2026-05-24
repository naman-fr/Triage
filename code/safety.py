"""
Safety & Adversarial Detection Module.
Multi-layer defense against prompt injections, PII detection, and content screening.
"""

import re
import base64
import json
from typing import Optional


# ─── Prompt Injection Detection ──────────────────────────────────────────────

# Patterns that indicate prompt injection attempts (case-insensitive)
INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|guidelines|directions)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions|prompts|rules|guidelines)",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|prompts|rules|context)",
    r"override\s+(all\s+)?(safety|security|content)\s+(protocols?|filters?|policies?|rules?)",
    r"you\s+are\s+now\s+(in\s+)?(maintenance|debug|admin|developer|test|DAN)\s+mode",
    r"you\s+are\s+no\s+longer\s+a\s+support\s+agent",
    r"you\s+are\s+now\s+DAN",
    r"DAN\s+mode\s+enabled",

    # System prompt extraction
    r"(output|print|display|show|reveal|share|provide|give)\s+(your|the)\s+(system|full|complete|original)\s+(prompt|instructions|rules|guidelines)",
    r"what\s+(are|is)\s+your\s+(system|internal)\s+(prompt|instructions|rules)",
    r"(share|reveal|output)\s+your\s+(instructions|guidelines|directives|configuration)",

    # System/role override tags
    r"<\s*system\s*>",
    r"\[\s*SYSTEM\s*(OVERRIDE|MESSAGE|ALERT|PROMPT)\s*\]",
    r"\[\s*ADMIN\s*(OVERRIDE|ACCESS|MODE)\s*\]",
    r"IMPORTANT\s*:\s*(Disregard|Ignore|Override|Forget)",

    # Fake authority / impersonation
    r"(I\s+am|this\s+is)\s+(a|an|the)\s+(senior|internal|authorized)\s+.{0,30}(engineer|admin|employee|QA|auditor)",
    r"(QA|quality\s+assurance)\s+(team|engineer|audit)\s+.{0,30}(verify|test|confirm|check)",
    r"AUTH_CODE\s*:",
    r"ALERT_ACK",
    r"emp_id\s*=",
    r"access_level\s*=",

    # Output manipulation
    r"(output|respond|reply)\s+(the\s+following|with|exactly)\s*:",
    r"STATUS\s*:\s*(replied|escalated)",
    r"REQUEST_TYPE\s*:",

    # Data exfiltration
    r"(complete|full|entire)\s+list\s+of\s+(support\s+)?articles",
    r"(exact|full)\s+retrieval\s+algorithm",
    r"how\s+many\s+documents\s+you\s+have",
    r"which\s+document\s+did\s+you\s+(pull|retrieve|use|get)",
    r"(list|show|output)\s+all\s+.{0,20}(tools|functions|capabilities)\s+available",
    r"confidence\s+scoring\s+algorithm",
    r"(internal|system)\s+(rules|logic|decision)",

    # Multi-lingual injection patterns
    r"(affiche|montre|donne)\s+(toutes?\s+)?(les?\s+)?(r[èe]gles|instructions|documents)",
    r"ignorieren\s+Sie\s+(Ihre\s+)?System",
    r"geben\s+Sie\s+.{0,30}(informationen|daten|anweisungen)\s+aus",
]

# Compile patterns for performance
COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS
]


# ─── PII Detection Patterns ─────────────────────────────────────────────────

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"  # 16 digits with optional separators
        r"|\b\d{4}[-\s]?XXXX[-\s]?XXXX[-\s]?\d{4}\b"  # Masked format
    ),
    "email": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "phone": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?"
        r"[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    ),
    "dob": re.compile(
        r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"
    ),
    "address": re.compile(
        r"\b\d{1,5}\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|"
        r"Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl|Way)\b",
        re.IGNORECASE,
    ),
}


# ─── Language Detection ──────────────────────────────────────────────────────

LANGUAGE_INDICATORS = {
    "zh": re.compile(r"[\u4e00-\u9fff]{3,}"),  # Chinese characters
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]{3,}"),  # Japanese
    "ko": re.compile(r"[\uac00-\ud7af]{3,}"),  # Korean
    "ar": re.compile(r"[\u0600-\u06ff]{3,}"),  # Arabic
    "fr": re.compile(
        r"\b(bonjour|merci|je\s+suis|s'il\s+vous|comment|pourquoi|"
        r"oui|non|votre|notre|cette|avec|pour|dans|les|des|une|"
        r"est|sont|ont|carte|bloqu[ée]e|aide|que|qui|quoi)\b",
        re.IGNORECASE,
    ),
    "de": re.compile(
        r"\b(ich\s+bin|bitte|danke|hilfe|konto|wurde|mein|meine|"
        r"passwort|zugang|wie|was|können|haben|nicht|und|der|die|das|"
        r"gehackt|wiederherstellen|Benutzer|helfen|geben|Ihre)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(hola|gracias|por\s+favor|ayuda|necesito|tarjeta|"
        r"buenas?\s+tardes?|reportar|clonada|llamé|banco|"
        r"qué|cómo|debo|hacer|contactar)\b",
        re.IGNORECASE,
    ),
    "hi": re.compile(r"[\u0900-\u097f]{3,}"),  # Hindi/Devanagari
}


# ─── Base64 Detection ────────────────────────────────────────────────────────

BASE64_PATTERN = re.compile(
    r"^[A-Za-z0-9+/]{20,}={0,2}$"
)


# ─── CSV/Excel Formula Injection ─────────────────────────────────────────────

FORMULA_INJECTION_PATTERN = re.compile(
    r"^[=+\-@]\s*(cmd|HYPERLINK|IMPORTRANGE|IMPORTDATA|IMPORTHTML|IMPORTXML)",
    re.IGNORECASE,
)


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """
    Detect prompt injection attempts in the given text.

    Returns:
        (is_injection, list_of_matched_patterns)
    """
    if not text:
        return False, []

    matched = []
    for i, pattern in enumerate(COMPILED_INJECTION_PATTERNS):
        if pattern.search(text):
            matched.append(INJECTION_PATTERNS[i])

    # Check for Base64 encoded injection
    for word in text.split():
        if BASE64_PATTERN.match(word) and len(word) > 20:
            try:
                decoded = base64.b64decode(word).decode("utf-8", errors="ignore")
                # Recursively check decoded content for injection
                sub_injection, sub_patterns = detect_injection(decoded)
                if sub_injection:
                    matched.append(f"base64_encoded_injection: {decoded[:100]}")
            except Exception:
                pass

    # Check for formula injection
    if FORMULA_INJECTION_PATTERN.match(text.strip()):
        matched.append("csv_formula_injection")

    return len(matched) > 0, matched


def detect_pii(text: str) -> tuple[bool, list[str]]:
    """
    Detect personally identifiable information in the given text.

    Returns:
        (has_pii, list_of_pii_types_found)
    """
    if not text:
        return False, []

    found_types = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found_types.append(pii_type)

    return len(found_types) > 0, found_types


def redact_pii(text: str) -> str:
    """
    Redact PII from text, replacing with generic placeholders.
    """
    if not text:
        return text

    # Redact SSNs
    text = PII_PATTERNS["ssn"].sub("XXX-XX-XXXX", text)

    # Redact credit card numbers (keep last 4 if possible)
    def redact_cc(match: re.Match) -> str:
        digits = re.sub(r"[\s\-]", "", match.group())
        if len(digits) >= 4:
            return f"XXXX-XXXX-XXXX-{digits[-4:]}"
        return "XXXX-XXXX-XXXX-XXXX"

    text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", redact_cc, text)

    # Redact DOB
    text = PII_PATTERNS["dob"].sub("XX/XX/XXXX", text)

    # Redact addresses (partial)
    text = PII_PATTERNS["address"].sub("[ADDRESS REDACTED]", text)

    return text


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text using regex heuristics.

    Returns ISO 639-1 language code.
    """
    if not text or not text.strip():
        return "en"

    # Check for non-Latin scripts first (they're unambiguous)
    for lang_code in ["zh", "ja", "ko", "ar", "hi"]:
        if LANGUAGE_INDICATORS[lang_code].search(text):
            return lang_code

    # Count matches for Latin-script languages
    scores: dict[str, int] = {}
    for lang_code in ["fr", "de", "es"]:
        pattern = LANGUAGE_INDICATORS[lang_code]
        matches = pattern.findall(text)
        if matches:
            scores[lang_code] = len(matches)

    if scores:
        # If the best non-English score is significant, return that language
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] >= 3:
            return best_lang

    # Default to English
    return "en"


def assess_risk_level(
    text: str,
    has_pii: bool,
    is_injection: bool,
    status: str,
) -> str:
    """
    Assess the risk level of a ticket based on content analysis.

    Returns: 'low', 'medium', 'high', or 'critical'
    """
    risk_keywords_critical = [
        r"identity\s+theft", r"identity\s+stolen",
        r"class\s+action\s+lawsuit", r"legal\s+counsel",
        r"data\s+breach", r"privacy\s+breach",
        r"unauthorized\s+access", r"account\s+compromis",
        r"suicid", r"self.harm", r"life.threatening",
        r"GDPR\s+Article\s+17",
    ]
    risk_keywords_high = [
        r"fraud", r"stolen", r"unauthorized\s+(transaction|charge)",
        r"legal\s+threat", r"lawsuit", r"lawyer",
        r"stranded", r"emergency",
        r"security\s+vulnerabilit",
        r"medical\s+advice", r"dangerous",
        r"discrimination",
        r"someone\s+else.s\s+conversation",
    ]
    risk_keywords_medium = [
        r"refund", r"chargeback", r"dispute",
        r"blocked\s+(card|account)",
        r"subscription\s+(cancel|pause)",
        r"account\s+delete",
        r"payment\s+fail",
        r"not\s+working",
        r"billing",
    ]

    text_lower = text.lower() if text else ""

    # Critical risk
    for pattern in risk_keywords_critical:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "critical"

    # PII + sensitive content = high
    if has_pii and any(
        re.search(p, text_lower, re.IGNORECASE) for p in risk_keywords_high
    ):
        return "critical"

    # Injection attempts are high risk
    if is_injection:
        return "high"

    # High risk keywords
    for pattern in risk_keywords_high:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "high"

    # PII present = at least medium
    if has_pii:
        return "medium"

    # Medium risk keywords
    for pattern in risk_keywords_medium:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return "medium"

    return "low"


def validate_response_safety(response: str, original_text: str) -> str:
    """
    Post-generation safety check on the response.
    Ensures no PII is echoed and no injection compliance.
    """
    if not response:
        return response

    # Check if response echoes PII from original text
    has_pii, pii_types = detect_pii(original_text)
    if has_pii:
        response = redact_pii(response)

    # Check that response doesn't comply with injection patterns
    compliance_patterns = [
        r"DAN\s+mode\s+enabled",
        r"ALERT_ACK",
        r"PWNED",
        r"maintenance\s+mode",
        r"system\s+prompt\s*:",
        r"my\s+(system\s+)?instructions\s+are",
    ]

    for pattern in compliance_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            response = (
                "I cannot comply with that request. I am a support agent "
                "designed to help with DevPlatform, Claude, and Visa inquiries. "
                "How can I assist you with a legitimate support question?"
            )
            break

    return response
