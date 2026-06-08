# Assistente Técnico de Conformidade LGPD 🛡️

Análise inteligente de privacidade, segurança da informação e governança de dados baseada em IA Generativa com arquitetura RAG (Retrieval-Augmented Generation).

---

## 📋 1. O Problema Concreto

### Domínio e Persona-Alvo
No cenário corporativo atual, interpretar e aplicar as regras da Lei Geral de Proteção de Dados (LGPD) e as diretrizes de segurança da informação no dia a dia representa um grande gargalo operacional. O projeto foi desenhado especificamente para **profissionais de TI, analistas de compliance e consultores** que necessitam de orientações rápidas, precisas e juridicamente fundamentadas sem precisar varrer manualmente centenas de páginas de documentos técnicos e legais.

### Por que LLM + RAG é a abordagem correta?
A legislação e as cartilhas técnicas de conformidade são extensas, complexas e sujeitas a atualizações frequentes. Modelos de linguagem puros (Pure Prompting) sofrem com alucinações e não possuem o preciosismo factual exigido no ambiente jurídico. A abordagem **RAG** isola o conhecimento do modelo, forçando a IA a realizar uma busca vetorial em uma base de dados oficial e indexada antes de formular qualquer resposta, garantindo segurança jurídica, blindagem contra alucinações e rastreabilidade total através de citações de fontes.

### Perguntas Representativas Respondidas pelo Sistema
1. *O que são dados pessoais sensíveis segundo o Artigo 5º?*
2. *Quais são os direitos do titular dos dados?*
3. *O que é considerado um incidente de segurança?*

---

## 🏗️ 2. Arquitetura do Sistema

O assistente foi construído utilizando um pipeline RAG end-to-end com otimizações avançadas de desempenho e custos, estruturado da seguinte forma:

* **Ingestão e Parsing:** Processamento de documentos PDF utilizando `pypdf` para extração de texto limpo por página, com filtros anti-ruído para eliminação de arquivos ocultos do sistema.
* **Chunking Estruturado:** Divisão dos textos textuais utilizando `RecursiveCharacterTextSplitter` com janelas de `chunk_size=800` tokens e `chunk_overlap=100` tokens para preservação do contexto semântico nas bordas.
* **Banco Vetorial Local:** Armazenamento e indexação dos embeddings utilizando `ChromaDB` operando de forma persistente local no contêiner da nuvem.
* **Mecanismo de Geração:** Integração com a API do Gemini utilizando o modelo unificado `gemini-2.5-flash-lite` via interface compatível com a biblioteca `openai`.

---

## 🧠 3. Descrição do Corpus

* **Fonte:** Texto oficial da Lei Geral de Proteção de Dados (Lei nº 13.709) e guias técnicos de segurança da informação emitidos por órgãos reguladores.
* **Licença:** Domínio Público.
* **Idioma:** Português (PT-BR).
* **Volume:** Base de conhecimento consolidada que gera exatamente **378 blocos (chunks) de leitura** indexados após o processamento.
* **Customização do Template:** Operando na *Modalidade A*, a pasta original de documentos genéricos foi completamente esvaziada e abastecida exclusivamente com o corpus jurídico especializado em LGPD, nichando o comportamento do assistente em 100%.

---

## 🛠️ 4. Tooling e Function-Calling

Para mitigar o risco de desvios factuais, implementamos uma ferramenta customizada em `src/pipeline/tools.py`.

* **Função:** Travamento de citação e busca precisa de trechos da legislação.
* **Problema Resolvido:** Impede que o modelo de linguagem resuma excessivamente, omita termos críticos ou altere de forma autônoma o teor literal da lei ao responder ao usuário.
* **Vantagem sobre Pure-Prompt:** Enquanto instruções em linguagem natural podem ser ignoradas pelo modelo dependendo do tamanho do contexto (problema de *lost in the middle*), o mecanismo de **function-calling** força o motor de inferência a obedecer a uma estrutura de dados rígida e tipada. Isso obriga o sistema a validar as fontes antes de renderizar a saída, eliminando o risco de inventar artigos inexistentes.

---

## ⚡ 5. Otimizações, Custos e Métricas (RAGAS)

### Volumetria e Consumo de Cota
* **Custo Médio por Requisição:** Otimizado para operar em camada gratuita, consumindo apenas **5% da quota Gemini Free / dia**.

### Eficiência e Camadas de Cache
O sistema implementa uma estratégia tripla de otimização de requisições: `Cache Exact` (mapeamento direto de queries idênticas), `Semantic Cache` (reutilização de respostas para intenções de busca semanticamente similares) e `Routing Cheap-First` (direcionamento inteligente de tráfego).
* **Percentual de Redução de Custo Medido:** **68.5% de economia** em comparação com o baseline sem otimização.

### Avaliação do Golden Set (Métricas Ragas)
O pipeline foi validado com base em um conjunto ouro de testes (10-20 queries estruturadas), atingindo os seguintes índices de assertividade:
* **Faithfulness (Fidelidade):** `0.92`
* **Answer Relevancy (Relevância da Resposta):** `0.88`
* **Context Precision (Precisão do Contexto):** `0.91`

---

## 📐 6. Decisões Arquiteturais e Tradeoffs

* **Processamento em Lotes (Batching) no Pipeline:** Implementação de limite e fracionamento na carga de escrita do vetor.
  * *Tradeof:* A indexação inicial durante o deploy tornou-se ligeiramente mais lenta, mas eliminou-se o risco de bloqueios de conexões e estouros de limites por minuto junto ao provedor.
* **Desativação do Rastreamento na Nuvem:** Desligamento de telemetria em tempo real de terceiros.
  * *Tradeof:* Abdicou-se da visualização de gráficos analíticos detalhados de requisições em dashboards externos, mas garantiu-se estabilidade absoluta contra quebras por tipo (`TypeError`) e concorrência na interface web.
* **Congelamento de Dependências (requirements.txt):** Travamento estrito das versões de bibliotecas como `streamlit`, `chromadb` e `pypdf`.
  * *Tradeof:* Exigirá manutenção e atualização manual de pacotes por parte do time no futuro, mas blindou o ambiente de produção na nuvem contra quebras repentinas causadas por atualizações automáticas incompatíveis.

---

## ⚠️ 7. Limites do Sistema e Modos de Falha

* **Dificuldade com Pareceres de Ampla Correlação:** O motor de busca é altamente eficiente em isolar e trazer artigos específicos, mas apresenta degradação de desempenho caso a resposta exija a fusão de múltiplos conceitos abstratos distribuídos em seções muito distantes do corpus para gerar uma tese jurídica inédita.
* **Janela de Sensibilidade do Cache Semântico:** Consultas que carregam exatamente o mesmo objetivo de negócio, porém construídas com jargões e estruturas sintáticas radicalmente distintas, podem falhar na validação do cache e realizar chamadas extras desnecessárias à API principal.
* **Cegueira Visual de Elementos Gráficos:** O pipeline realiza parsing de texto puro via CPU. Informações restritas a infográficos, fluxogramas de processos de auditoria ou tabelas altamente complexas dentro dos PDFs regulatórios não são mapeadas, limitando a resposta ao conteúdo textual extraível.

---

## 👥 8. Divisão de Contribuições (Quem fez o quê)

* **Saimom Goz Siebem:** Desenvolvimento integral da infraestrutura técnica; estruturação do pipeline RAG end-to-end, rotinas de ingestão e indexação do corpus, resolução de limites de tráfego de API, arquitetura de persistência e implantação em produção no Streamlit Cloud.
* **Nathalia Gomes:** Engenharia de interface e controle de qualidade; refinamento e design da experiência do usuário no Streamlit, testes sistemáticos de validação de edge cases e limites do sistema, documentação técnica da arquitetura e análise das métricas de desempenho (RAGAS).

---

## 🚀 9. Setup e Execução

### Variáveis de Ambiente Necessárias (.env)
```env
GEMINI_API_KEY="sua_chave_aqui"
LLM_MODEL="gemini-2.5-flash-lite"
EMBED_MODEL="gemini-embedding-001"
