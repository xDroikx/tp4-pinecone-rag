import os
import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import RespuestaRAG
from ingest import run_ingestion

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_chain")

PROMPT_TEMPLATE = """
Eres un asistente técnico experto y veraz. Responde únicamente basándote en el siguiente CONTEXTO proporcionado.
Si la respuesta no se encuentra en el contexto, debes responder estrictamente: "No dispongo de esa información en el contexto proporcionado."
No utilices conocimientos previos fuera del contexto provisto.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""


def get_llm_model():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

    kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


async def get_rag_response(query: str) -> RespuestaRAG:
    vectorstore = run_ingestion()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    logger.info(f"🔍 Buscando información para: '{query}'...")
    docs_recuperados = await retriever.ainvoke(query)
    
    contexto_texto = "\n\n".join([doc.page_content for doc in docs_recuperados])
    fuentes_detectadas = list(set([os.path.basename(doc.metadata.get("source", "desconocido")) for doc in docs_recuperados]))
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = get_llm_model()
    structured_model = model.with_structured_output(RespuestaRAG)
    
    chain = prompt | structured_model
    
    logger.info("🤖 Generando respuesta con el LLM...")
    resultado: RespuestaRAG = await chain.ainvoke({
        "contexto": contexto_texto,
        "pregunta": query
    })
    
    if "No dispongo de esa información" not in resultado.respuesta:
        resultado.fuentes = fuentes_detectadas
    else:
        resultado.fuentes = []
        
    return resultado