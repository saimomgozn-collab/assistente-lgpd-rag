from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


def _make_client() -> tuple[OpenAI, str]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def ingest_and_index(self) -> int:
        """Le PDFs de corpus_dir, faz chunking e indexa em Chroma com throttle."""
        docs: list[dict] = []
        
        # Ingestao de PDFs (ignora arquivos ocultos)
        for path in self.corpus_dir.glob("*.pdf"):
            if path.name.startswith("."):
                continue
            reader = PdfReader(path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "text": text,
                        "source": path.name,
                        "page": idx + 1
                    })

        # Chunking Recursivo
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        ids, documents, metadatas = [], [], []

        for doc in docs:
            split_texts = splitter.split_text(doc["text"])
            for text in split_texts:
                ids.append(str(uuid.uuid4()))
                documents.append(text)
                metadatas.append({
                    "source": doc["source"],
                    "page": doc["page"]
                })

        # Indexacao controlada por lotes e tempo para evitar estouro de cota
        if ids:
            batch_size = 20
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i : i + batch_size],
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size]
                )
                time.sleep(5)  # Pausa de seguranca para reset de limite por minuto

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        results = self.collection.query(query_texts=[query], n_results=k)
        hits = []
        
        if results and results["documents"] and len(results["documents"]) > 0:
            for idx in range(len(results["documents"][0])):
                hits.append({
                    "text": results["documents"][0][idx],
                    "source": results["metadatas"][0][idx]["source"],
                    "page": results["metadatas"][0][idx]["page"],
                    "distance": results["distances"][0][idx] if results["distances"] else 0.0
                })
        return hits

    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate."""
        hits = self.retrieve(question, k=k)

        # Montagem do contexto estruturado
        context_blocks = []
        for h in hits:
            context_blocks.append(f"[{h['source']}:{h['page']}]\n{h['text']}")
        context = "\n\n".join(context_blocks)

        # Chat completion via provider unificado
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "sources": [(h["source"], h["page"]) for h in hits]
        }


PROMPT_TEMPLATE = """Voce e um assistente tecnico. Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa se o banco estiver vazio."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
