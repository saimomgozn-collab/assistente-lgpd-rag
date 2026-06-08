# Assistente Técnico de Conformidade LGPD

> Sistema especialista para análise de privacidade e segurança da informação, blindado contra alucinações jurídicas através de function-calling e RAG.

**Live demo:** [Insira o link gerado pelo Streamlit Cloud aqui]

## Problem statement

1. **Qual problema resolve?** Evita o risco de interpretações jurídicas incorretas geradas por IA genérica ao lidar com a proteção de dados.
2. **Para quem?** Profissionais de TI, DPOs (Data Protection Officers) e gestores de tecnologia estruturando defesas cibernéticas.
3. **Por que LLM + RAG + Tool-use?** Modelos base tendem a "alucinar" incisos e parágrafos de leis. O RAG ancora a resposta na documentação oficial da ANPD, e a ferramenta (tool-use) garante a citação literal de artigos críticos da LGPD.

## Arquitetura

```mermaid
flowchart LR
    USER([User]) --> UI[Streamlit UI]
    UI --> CACHE{Exact cache?}
    CACHE -->|hit| RESP[Response]
    CACHE -->|miss| SEM{Semantic cache?}
    SEM -->|hit| RESP
    SEM -->|miss| CLS[Classify complexity]
    CLS -->|simple| CHEAP[Cheap LLM]
    CLS -->|complex| ORCH[Orchestrator]
    ORCH --> RAG[(Chroma RAG)]
    ORCH --> TOOL[Custom tool]
    RAG --> PREMIUM[Premium LLM]
    TOOL --> PREMIUM
    PREMIUM --> RESP
