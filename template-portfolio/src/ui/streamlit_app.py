"""Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Path setup
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402

def aplicar_design_premium():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f19;
        }
        
        div[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }
        
        div[data-testid="stSidebar"] div.stMarkdown {
            padding: 0.2rem 0;
        }
        
        /* Logo lateral LGPD */
        .sidebar-logo {
            text-align: center;
            padding: 10px 0 30px 0;
            border-bottom: 1px solid #1f2937;
            margin-bottom: 20px;
        }
        
        .sidebar-logo h1 {
            color: #fbbf24;
            font-weight: 900;
            margin: 0;
            font-size: 2.8rem;
            letter-spacing: 2px;
            line-height: 1.2;
        }
        
        .sidebar-logo p {
            color: #9ca3af;
            margin: 0;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        /* Cards de Metricas */
        .metric-card {
            background-color: #1f2937;
            border-left: 3px solid #3b82f6;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        
        .metric-title {
            color: #9ca3af;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 5px;
        }
        
        .metric-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* Header Principal Customizado */
        .main-header-container {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            padding-top: 10px;
        }

        .shield-icon {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .shield-icon span {
            font-size: 3.5rem;
            line-height: 1;
        }

        .header-text h1 {
            margin: 0;
            font-size: 2.4rem;
            color: #f8fafc;
            font-weight: 800;
        }

        .header-text p {
            margin: 5px 0 0 0;
            color: #94a3b8;
            font-size: 1.05rem;
        }

        .stTextInput img {
            max-width: 100%;
        }
        
        .architecture-box {
            background-color: #111827;
            border-left: 4px solid #fbbf24;
            border-radius: 4px;
            padding: 15px;
            margin-top: 40px;
            color: #9ca3af;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Config UI
st.set_page_config(page_title="Assistente LGPD", page_icon="🛡️", layout="centered")

aplicar_design_premium()

# Header customizado via HTML para controle total do design
st.markdown(
    """
    <div class="main-header-container">
        <div class="shield-icon">
            <span>🛡️</span>
        </div>
        <div class="header-text">
            <h1>Assistente Técnico LGPD</h1>
            <p>Análise inteligente de privacidade, segurança da informação e governança de dados.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Cache & Pipeline init
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

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <h1>LGPD</h1>
            <p>Conformidade</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Métricas do Sistema")
    
    chunks_count = pipeline.collection.count()
    exact_hits = exact_cache.stats()["size"]
    semantic_hits = semantic_cache.stats()["size"]
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Chunks Indexados</div>
            <div class="metric-value">{chunks_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Exact Cache Hits</div>
            <div class="metric-value">{exact_hits}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Semantic Cache Hits</div>
            <div class="metric-value">{semantic_hits}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Limpar Caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a página.")

# Main UI
query = st.text_input("Sua pergunta:", placeholder="Ex: O que são dados pessoais sensíveis segundo o Artigo 5?")

if query:
    trace_id = "no-trace"

    # Exact cache lookup
    cached = exact_cache.get(query)
    if cached:
        st.success("Cache hit (exact)")
        st.write(cached)
        log_event("cache_hit", trace_id=trace_id, layer="exact")
        st.stop()

    # Semantic cache lookup
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

    # RAG & Routing
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

    # Output & cache update
    st.write(result["answer"])
    if result.get("sources"):
        with st.expander("Fontes citadas"):
            for source, page in result["sources"]:
                st.write(f"- {source} (p. {page})")

    exact_cache.put(query, result["answer"])
    semantic_cache.put(query, result["answer"])
    log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))

# Footer
st.markdown(
    """
    <div class="architecture-box">
        <strong>Arquitetura:</strong> RAG local com ChromaDB e embeddings do Gemini. 
        Proteção ativa via function-calling para citação precisa de artigos.
    </div>
    """,
    unsafe_allow_html=True,
)
