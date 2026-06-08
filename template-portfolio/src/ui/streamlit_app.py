"""Streamlit UI — entrada principal do app. Pronta para deploy no Streamlit Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Adiciona root do projeto no path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402

# UI setup
st.set_page_config(page_title="Assistente LGPD & Defesa", page_icon=":shield:", layout="centered")

st.title(":shield: Assistente Técnico de Conformidade LGPD")
st.caption("Análise inteligente de privacidade, segurança da informação e governança de dados.")


# Inicializacao de pipeline e caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Metricas e debug na sidebar
with st.sidebar:
    st.header("Métricas do Sistema")
    st.metric("Chunks Indexados", pipeline.collection.count())
    st.metric("Exact Cache Hits", exact_cache.stats()["size"])
    st.metric("Semantic Cache Hits", semantic_cache.stats()["size"])

    if st.button("Limpar Caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a página.")


# Interface principal
query = st.text_input("Sua pergunta:", placeholder="Ex: O que são dados pessoais sensíveis segundo o Artigo 5?")

if query:
    # Trace contornado para evitar TypeError na nuvem
    trace_id = "no-trace"

    # Busca no cache exato
    cached = exact_cache.get(query)
    if cached:
        st.success("Cache hit (exact)")
        st.write(cached)
        log_event("cache_hit", trace_id=trace_id, layer="exact")
        st.stop()

    # Busca no cache semantico
    try:
        cached = semantic_cache.get(query)
    except NotImplementedError:
        cached = None
        st.warning("Semantic cache nao implementado. Caindo no LLM real.")

    if cached:
        st.success("Cache hit (semantic)")
        st.write(cached)
        log_event("cache_hit", trace_id=trace_id, layer="semantic")
        st.stop()

    # Execucao do pipeline RAG e roteamento
    try:
        decision = classify_complexity(query)
        st.info(f"Routing: {decision.complexity} -> {decision.model}")
        log_event("route_decision", trace_id=trace_id, model=decision.model, complexity=decision.complexity, reason=decision.reason)
    except NotImplementedError:
        st.warning("Routing nao implementado. Usando modelo default.")

    try:
        result = pipeline.answer(query)
    except NotImplementedError as e:
        st.error(f"Pipeline nao implementado: {e}")
        st.info("Implemente os requisitos em src/pipeline/rag.py para destravar.")
        st.stop()

    # Renderizacao e atualizacao de cache
    st.write(result["answer"])
    if result.get("sources"):
        with st.expander("Fontes citadas"):
            for source, page in result["sources"]:
                st.write(f"- {source} (p. {page})")

    exact_cache.put(query, result["answer"])
    semantic_cache.put(query, result["answer"])
    log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))


st.divider()
st.caption(
    "Arquitetura: RAG local com ChromaDB e embeddings do Gemini. "
    "Proteção ativa via function-calling para citação precisa de artigos."
)
