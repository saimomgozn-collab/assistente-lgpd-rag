"""Streamlit UI - Versão Premium LGPD"""
# ... (manter os imports que você já tem)

def aplicar_design_premium():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; }
        
        /* Sidebar e Logo */
        div[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
        
        .sidebar-brand {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(180deg, #1e3a8a 0%, #0b0f19 100%);
            margin-bottom: 20px;
        }
        
        .sidebar-brand h1 { color: #fbbf24; font-size: 2rem; margin: 0; }
        .sidebar-brand p { color: #60a5fa; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; }

        /* Estilo para as Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; color: #9ca3af; }
        
        /* Cards */
        .metric-card {
            background-color: #1f2937;
            border-left: 4px solid #fbbf24;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Configuração
st.set_page_config(page_title="Assistente LGPD", page_icon="⚖️", layout="wide")
aplicar_design_premium()

# Sidebar Refinada
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <h1>🛡️ LGPD</h1>
            <p>Compliance Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Abas na sidebar para métricas
    tab_metrica = st.tabs(["📊 Métricas", "⚙️ Configs"])[0]
    with tab_metrica:
        chunks_count = pipeline.collection.count()
        st.markdown(f"""
            <div class="metric-card">
                <div style="color:#9ca3af; font-size:0.7rem;">CHUNKS INDEXADOS</div>
                <div style="color:white; font-size:1.5rem; font-weight:bold;">{chunks_count}</div>
            </div>
        """, unsafe_allow_html=True)

# Main Page com Tabs (O pulo do gato para organizar)
tab1, tab2 = st.tabs(["💬 Chat com a Lei", "📝 Documentação"])

with tab1:
    st.title("⚖️ Conformidade LGPD")
    query = st.chat_input("Pergunte sobre artigos, prazos ou governança...")
    if query:
        # ... (seu código de lógica de RAG continua aqui)
        st.chat_message("user").write(query)
        # ... renderização da resposta
        st.chat_message("assistant").write(result["answer"])

with tab2:
    st.subheader("Arquitetura do Sistema")
    st.info("RAG local com ChromaDB e embeddings do Gemini.")
