"""Streamlit UI - Assistente de Conformidade LGPD """

from __future__ import annotations
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# path setup
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
load_dotenv()

# imports 
from src.observability.trace import log_event
from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity

# inicial config
st.set_page_config(page_title="Assistente LGPD", page_icon="🛡️", layout="wide")

def aplicar_design_premium():
    """Injeta CSS customizado para um visual corporativo e moderno."""
    st.markdown("""
        <style>
        .stApp { background-color: #0b0f19; }
        div[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
        
        /* emblema 3d */
        .sidebar-brand { 
            text-align: center; 
            padding: 2rem 1rem; 
            background: rgba(17, 24, 39, 0.5); 
            border: 1px solid rgba(251, 191, 36, 0.2);
            border-radius: 20px;
            margin: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 0 15px rgba(251, 191, 36, 0.1);
            backdrop-filter: blur(10px);
        }
        .sidebar-brand h1 { 
            color: #fbbf24; 
            font-size: 2.5rem; 
            margin: 0; 
            text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
        }
        .sidebar-brand p { 
            color: #ffffff; 
            font-size: 0.7rem; 
            letter-spacing: 4px; 
            text-transform: uppercase; 
            margin-top: 5px;
            opacity: 0.8;
        }
        
        .metric-card { background-color: #1f2937; border-left: 4px solid #fbbf24; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .source-tag { font-size: 0.8rem; color: #60a5fa; background: #1e3a8a; padding: 2px 8px; border-radius: 4px; margin-right: 5px; }
        </style>
    """, unsafe_allow_html=True)

aplicar_design_premium()

# pipeline cache
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

# sidebar (metricas)
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <h1>🛡️</h1>
            <h1 style="font-size: 1.5rem;">LGPD</h1>
            <p>COMPLIANCE</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("Dashboard")
    chunks = pipeline.collection.count()
    st.markdown(f'<div class="metric-card">Chunks Indexados: <b>{chunks}</b></div>', unsafe_allow_html=True)
    
    if st.button("Limpar Caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.rerun()

# principal (chat)
tab1, tab2 = st.tabs(["💬 Chat com a Lei", "⚙️ Arquitetura"])

with tab1:
    st.title("Assistente Técnico LGPD")
    
    # input chat
    query = st.chat_input("Pergunte sobre artigos, prazos ou governança...")
    
    if query:
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            # cache rag
            cached = exact_cache.get(query) or semantic_cache.get(query)
            
            if cached:
                st.write(cached)
            else:
                # pipeline real
                try:
                    result = pipeline.answer(query)
                    st.write(result["answer"])
                    
                    # mostra citacoes
                    if result.get("sources"):
                        st.markdown("---")
                        st.caption("Fontes consultadas:")
                        for source, page in result["sources"]:
                            st.markdown(f"<span class='source-tag'>{source}</span> (p. {page})", unsafe_allow_html=True)
                    
                    # salva cache
                    exact_cache.put(query, result["answer"])
                    semantic_cache.put(query, result["answer"])
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

with tab2:
    st.subheader("Configuracoes do Sistema")
    st.info("Arquitetura: RAG local com ChromaDB e embeddings do Gemini.")
