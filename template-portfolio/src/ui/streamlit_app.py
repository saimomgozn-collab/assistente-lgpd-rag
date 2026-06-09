"""Streamlit UI - Assistente de Conformidade LGPD - Versão Premium"""

from __future__ import annotations
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# PATH E SETUP
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
load_dotenv()

# Imports 
from src.observability.trace import log_event
from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity

# INICIAL
st.set_page_config(page_title="Assistente LGPD", page_icon="🛡️", layout="wide")

def aplicar_design_premium():
    """Injeta CSS customizado para um visual corporativo e moderno."""
    st.markdown("""
        <style>
        .stApp { background-color: #0b0f19; }
        div[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
        
        .sidebar-brand { text-align: center; padding: 2rem 0; background: linear-gradient(180deg, #1e3a8a 0%, #0b0f19 100%); margin-bottom: 20px; }
        .sidebar-brand h1 { color: #fbbf24; font-size: 2rem; margin: 0; }
        .sidebar-brand p { color: #60a5fa; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; }
        
        .metric-card { background-color: #1f2937; border-left: 4px solid #fbbf24; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        </style>
    """, unsafe_allow_html=True)

aplicar_design_premium()

# PIPELINE E CACHE
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))

@st.cache_resource
def get_exact_cache():
    return ExactCache()

@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)

# objetos
pipeline = get_pipeline()
exact_cache = get_exact_cache()
semantic_cache = get_semantic_cache()

# SIDEBAR (METRICAS)
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <h1>🛡️ LGPD</h1>
            <p>Compliance Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("Dashboard")
    chunks = pipeline.collection.count()
    st.markdown(f'<div class="metric-card">Chunks Indexados: <b>{chunks}</b></div>', unsafe_allow_html=True)
    
    if st.button("Limpar Caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.rerun()

# PRINCIPAL (CHAT)
tab1, tab2 = st.tabs(["💬 Chat com a Lei", "⚙️ Arquitetura"])

with tab1:
    st.title("Assistente Técnico LGPD")
    
    # Input de Chat
    query = st.chat_input("Pergunte sobre artigos, prazos ou governança...")
    
    if query:
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            # Cache e RAG
            cached = exact_cache.get(query) or semantic_cache.get(query)
            
            if cached:
                st.write(cached)
            else:
                # pipeline real
                try:
                    result = pipeline.answer(query)
                    st.write(result["answer"])
                    # Salva cache
                    exact_cache.put(query, result["answer"])
                    semantic_cache.put(query, result["answer"])
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

with tab2:
    st.subheader("Configurações do Sistema")
    st.info("Arquitetura: RAG local com ChromaDB e embeddings do Gemini.")
