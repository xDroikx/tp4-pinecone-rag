import os
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tp4-rag-index")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def get_embeddings():
    """Usa el modelo local MiniLM (genera vectores de 384 dimensiones)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def init_pinecone_index(dimension: int = 384):
    """Verifica si el índice existe en Pinecone Serverless; de lo contrario, lo crea."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        logger.info(f"🌲 Creando índice Serverless en Pinecone: '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("✅ Índice en Pinecone creado exitosamente.")
    else:
        logger.info(f"⚡ El índice '{INDEX_NAME}' ya existe en Pinecone.")

    return pc


def run_ingestion(data_dir: str = "./data"):
    if not PINECONE_API_KEY:
        raise ValueError("❌ No se encontró PINECONE_API_KEY en el archivo .env")

    # 1. Asegurar la existencia del índice en la nube
    init_pinecone_index(dimension=384)

    # 2. Cargar documentos
    logger.info(f"📂 Cargando documentos desde '{data_dir}'...")
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    if not docs:
        raise FileNotFoundError(f"No se encontraron archivos .txt en '{data_dir}'.")

    # 3. Aplicar Chunking
    logger.info(f"✂️ Aplicando chunking a {len(docs)} documentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    # 4. Enriquecer con metadatos explícitos
    for chunk in chunks:
        file_name = os.path.basename(chunk.metadata.get("source", "desconocido"))
        chunk.metadata["source"] = file_name
        chunk.metadata["category"] = "tecnico"

    # 5. Indexar en Pinecone VectorStore
    logger.info("☁️ Subiendo vectores y metadatos a Pinecone Serverless...")
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    logger.info("✅ Ingesta en Pinecone completada.")
    return vectorstore, chunks


if __name__ == "__main__":
    run_ingestion()