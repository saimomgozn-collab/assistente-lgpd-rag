from __future__ import annotations

import os
import uuid
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


def _make_client() -> OpenAI:
    """Inicializa cliente OpenAI-compatible."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local e Embeddings Locais (Offline)."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs_local_v1",  # Nome novo para evitar conflitos de cache
        llm_model: str | None = None,
    ) -> None:
        self.client = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        
        # MAGIA AQUI: Sem passar 'embedding_function', o Chroma baixa automaticamente 
        # um modelo de IA gratuito e roda direto na CPU da nuvem, sem limites.
        self.collection = chroma.get_or_create_collection(
            name=collection_name
        )

    def ingest_and_index(self) -> int:
        """Le PDFs e indexa LOCALMENTE sem gastar API e sem Rate Limits."""
        docs: list[dict] = []
        
        # Ingestao de PDFs
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

        # Chunking
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

        # Indexacao em massa na velocidade da luz (sem API, sem restricoes)
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

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

        context_blocks = []
        for h in hits:
            context_blocks.append(f"[{h['source']}:{h['page']}]\n{h['text']}")
        context = "\n\n".join(context_blocks)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Somente AQUI a API gasta cota, apenas com o chat final.
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
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
