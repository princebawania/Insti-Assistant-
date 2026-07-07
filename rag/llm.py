"""
LLM wrapper. Default provider is Google Gemini (free tier). The rest of the
codebase only calls `generate(prompt, system)`, so swapping providers is a
one-file change. OpenAI and Anthropic paths are included for convenience.
"""
from __future__ import annotations

import os

import config

_DEFAULT_MODELS = {
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


class LLMError(RuntimeError):
    pass


def _model_name() -> str:
    return config.LLM_MODEL or _DEFAULT_MODELS.get(config.LLM_PROVIDER, "")


def generate(prompt: str, system: str = "") -> str:
    """Send a single prompt to the configured LLM and return its text reply."""
    provider = config.LLM_PROVIDER

    if provider == "gemini":
        return _gemini(prompt, system)
    if provider == "openai":
        return _openai(prompt, system)
    if provider == "anthropic":
        return _anthropic(prompt, system)
    raise LLMError(f"Unknown LLM_PROVIDER: {provider!r}")


# ---------- Gemini (default, free tier) ----------
def _gemini(prompt: str, system: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://aistudio.google.com/app/apikey"
        )
    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        _model_name(),
        system_instruction=system or None,
        generation_config={"temperature": config.LLM_TEMPERATURE},
    )
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# ---------- OpenAI ----------
def _openai(prompt: str, system: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=_model_name(),
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------- Anthropic ----------
def _anthropic(prompt: str, system: str) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set.")
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=_model_name(),
        max_tokens=1024,
        temperature=config.LLM_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()
