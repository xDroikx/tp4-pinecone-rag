# TP 4: Sistema RAG Escalable en la Nube con Pinecone Serverless y Recuperación Híbrida

Este repositorio contiene la implementación de un **Módulo de Recuperación Escalable RAG** en la nube utilizando **Pinecone Serverless**, **BM25 (Búsqueda Léxica)** mediante `EnsembleRetriever` y evaluación automatizada de métricas (**Recall@5** y **Precision@5**).

---

## 🛠️ Arquitectura del Sistema

1. **Pipeline de Ingesta (`ingest.py`)**:
   - Carga de documentos `.txt` desde `./data`.
   - Chunking adaptativo con `RecursiveCharacterTextSplitter` (~600 caracteres, 50 overlap).
   - Generación de embeddings con `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones).
   - Creación automática e indexación en un índice **Pinecone Serverless** (`tp4-rag-index`) enriquecido con metadatos (`source`, `category`).

2. **Recuperador Híbrido (`rag_system.py`)**:
   - **Búsqueda Semántica**: Pinecone VectorStore.
   - **Búsqueda Léxica**: BM25Retriever por coincidencia de palabras clave.
   - **EnsembleRetriever**: Combinación 50/50 para maximizar precisión en términos técnicos.
   - Generación de respuesta delimitada por contexto con el LLM en **Groq**.

3. **Módulo de Evaluación (`evaluate.py`)**:
   - Evaluación contra un *Golden Set* (`eval_dataset.json`).
   - Medición de **Recall@5** y **Precision@5**.

---

## 📋 Requisitos e Instalación

### 1. Clonar e Instalar Dependencias
```bash
git clone [suspicious link removed]
cd tp4-pinecone-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt