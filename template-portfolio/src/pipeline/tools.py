"""Function-calling / tool-use — registro de tools usadas pelo agente."""

from __future__ import annotations

import json
from typing import Any, Callable

# Dicionario para a demo (artigos mais consultados da LGPD)
LGPD_ARTICLES = {
    5: "Art. 5º Para os fins desta Lei, considera-se:\nI - dado pessoal: informação relacionada a pessoa natural identificada ou identificável;\nII - dado pessoal sensível: dado pessoal sobre origem racial ou étnica, convicção religiosa, opinião política, filiação a sindicato ou a organização de caráter religioso, filosófico ou político, dado referente à saúde ou à vida sexual, dado genético ou biométrico, quando vinculado a uma pessoa natural;\nIII - dado anonimizado: dado relativo a titular que não possa ser identificado, considerando a utilização de meios técnicos razoáveis e disponíveis na ocasião de seu tratamento;",
    7: "Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:\nI - mediante o fornecimento de consentimento pelo titular;\nII - para o cumprimento de obrigação legal ou regulatória pelo controlador;\nIII - pela administração pública, para o tratamento e uso compartilhado de dados necessários à execução de políticas públicas previstas em leis e regulamentos ou respaldadas em contratos, convênios ou instrumentos congêneres;",
    18: "Art. 18. O titular dos dados pessoais tem o direito a obter do controlador, em relação aos dados do titular por ele tratados, a qualquer momento e mediante requisição:\nI - confirmação da existência de tratamento;\nII - acesso aos dados;\nIII - correção de dados incompletos, inexatos ou desatualizados;\nIV - anonimização, bloqueio ou eliminação de dados desnecessários, excessivos ou tratados em desconformidade com o disposto nesta Lei;"
}

def cite_article(article_number: int) -> str:
    """Retorna o texto integral de um artigo especifico da LGPD."""
    article = LGPD_ARTICLES.get(article_number)
    if article:
        return article
    return f"Artigo {article_number} não está no cache rápido. Consulte o documento completo recuperado no contexto."


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": "Retorna o texto exato de um artigo da Lei Geral de Proteção de Dados (LGPD). Use sempre que o usuário perguntar o que diz um artigo específico da lei.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "O número do artigo da LGPD a ser consultado (ex: 5, 7, 18).",
                    },
                },
                "required": ["article_number"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"