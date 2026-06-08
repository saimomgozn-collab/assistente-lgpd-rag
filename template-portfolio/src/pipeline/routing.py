"""Model routing cheap-first com fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo."""
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    query_lower = query.lower()
    complex_keywords = [
        "explique", "compare", "analise", "projete", 
        "diferença", "resuma", "por que", "quais os impactos"
    ]

    # Verifica necessidade de processamento profundo
    if any(word in query_lower for word in complex_keywords):
        return RouteDecision(
            model=premium_model, 
            complexity="complex", 
            reason="Palavras-chave analiticas detectadas"
        )

    # Verifica queries curtas e diretas
    if len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(
            model=cheap_model, 
            complexity="simple", 
            reason="Query curta e direta"
        )

    # Fallback padrao
    return RouteDecision(
        model=cheap_model, 
        complexity="simple", 
        reason="Fallback padrao (cheap-first)"
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()