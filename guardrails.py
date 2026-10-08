import os
import re


SECRET_NAMES = ("GROQ_API_KEY", "SUPABASE_SERVICE_KEY")
SECRET_PATTERN = re.compile(r"(?:gsk_|sb_secret_)[A-Za-z0-9_-]{15,}")
OVERRIDE_PATTERN = re.compile(
    r"\bignore\s+(?:all\s+)?(?:previous|prior|system|your)\s+(?:instructions|rules|prompts)\b"
    r"|\b(?:reveal|show|print|give|display|expose)\b.{0,80}"
    r"\b(?:api[ _-]?keys?|secret[ _-]?keys?|system[ _-]?prompt|service[ _-]?key|\.env)\b",
    re.IGNORECASE | re.DOTALL,
)


class GuardrailError(ValueError):
    """A request or response failed a guardrail."""


class InputGuard:
    def check(self, question):
        if not isinstance(question, str):
            raise GuardrailError("Please enter your question as text.")
        question = question.strip()
        if len(question) < 3:
            raise GuardrailError("Please enter a longer question.")
        if len(question) > 2000:
            raise GuardrailError("Please keep your question below 2,000 characters.")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in question):
            raise GuardrailError("Please remove unusual control characters.")
        if SECRET_PATTERN.search(question):
            raise GuardrailError("Remove API keys or secrets from your question.")
        for name in SECRET_NAMES:
            secret = os.getenv(name)
            if secret and secret in question:
                raise GuardrailError("Remove API keys or secrets from your question.")
        if OVERRIDE_PATTERN.search(question):
            raise GuardrailError(
                "I can answer NECA questions, but cannot reveal keys or override my rules."
            )
        return question


class OutputGuard:
    def check(self, result):
        answer = result.get("answer")
        sources = result.get("sources")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 6000:
            raise GuardrailError("The answer failed the output check. Please try again.")
        if not isinstance(sources, list):
            raise GuardrailError("The answer had invalid source references.")
        if result.get("status") == "answered" and not sources:
            raise GuardrailError("The answer was missing its sources.")
        # Citation numbers are validated by chat.format_model_answer.
        # These checks do not prove that every claim is factually supported.
        visible = answer + " " + str(sources)
        if SECRET_PATTERN.search(visible):
            raise GuardrailError("The answer was blocked because it contained a secret.")
        for name in SECRET_NAMES:
            secret = os.getenv(name)
            if secret and secret in visible:
                raise GuardrailError("The answer was blocked because it contained a secret.")
        return result


def safe_error(error):
    message = str(error)
    for name in SECRET_NAMES:
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[hidden]")
    return SECRET_PATTERN.sub("[hidden]", message)

