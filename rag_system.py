import os
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Intentar importar EnsembleRetriever; si falla por versiones de LangChain, usamos la clase de respaldo
try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    class EnsembleRetriever:
        def __init__(self, retrievers, weights=None):
            self.retrievers = retrievers
            self.weights = weights or [0.5, 0.5]

        def invoke(self, query: str):
            all_docs = []
            seen = set()
            for retriever in self.retrievers:
                docs = retriever.invoke(query)
                for doc in docs:
                    key = (doc.page_content, doc.metadata.get("source"))
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            return all_docs

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_system")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tp4-rag-index")


class RAGSystem:
    def __init__(self, data_dir: str = "./data", top_k: int = 5):
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 1. Recuperador Vectorial (Pinecone)
        logger.info("🌲 Conectando a Pinecone VectorStore...")
        self.vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # 2. Recuperador Léxico (BM25)
        logger.info("📚 Inicializando BM25 Retriever local...")
        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            file_name = os.path.basename(chunk.metadata.get("source", "desconocido"))
            chunk.metadata["source"] = file_name
            chunk.metadata["category"] = "tecnico"

        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = self.top_k

        # 3. EnsembleRetriever Híbrido (50% vectorial + 50% BM25)
        logger.info("🔀 Creando EnsembleRetriever (Vectorial 50% + BM25 50%)...")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )

        # 4. Configurar LLM y Prompt
        model_name = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0
        )

        prompt_template = """Eres un asistente técnico RAG. Responde ÚNICAMENTE utilizando el siguiente contexto recuperado.
Si la respuesta no se encuentra explícitamente en el contexto, di "No dispongo de esa información en el contexto proporcionado."

Contexto:
{context}

Pregunta: {question}
Respuesta:"""

        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def get_relevant_documents(self, query: str):
        """Devuelve los top-k documentos recuperados por la búsqueda híbrida."""
        docs = self.hybrid_retriever.invoke(query)
        return docs[:self.top_k]

    def query(self, question: str):
        """Ejecuta la consulta completa y devuelve la respuesta junto a sus fuentes."""
        docs = self.get_relevant_documents(question)
        context = "\n\n".join([f"[{d.metadata.get('source', 'N/A')}] {d.page_content}" for d in docs])
        response = self.chain.invoke({"context": context, "question": question})
        sources = list(set([d.metadata.get("source", "desconocido") for d in docs]))
        return {
            "respuesta": response,
            "fuentes": sources,
            "documentos": docs
        }


if __name__ == "__main__":
    rag = RAGSystem()
    res = rag.query("¿Cómo funciona la búsqueda híbrida y BM25 en Pinecone?")
    print("\n--- Respuesta del Sistema ---")
    print(res["respuesta"])
    print("Fuentes consultadas:", res["fuentes"])