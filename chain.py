import os
import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import EntidadesTecnicas

load_dotenv()

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_extraccion")

# 1. Definición del Prompt Modular (LCEL)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un ingeniero especialista en arquitectura de software y análisis de logs. "
               "Tu tarea es analizar el texto provisto y extraer las entidades técnicas con precisión. "
               "Identificá tecnologías mencionadas, evaluá la criticidad (baja, media, alta) "
               "y generá un resumen técnico breve."),
    ("human", "{texto}")
])

def get_model():
    """Fábrica para instanciar el cliente compatible con OpenAI / Groq con temperature=0."""
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

def build_chain():
    """
    Construye la Cadena LCEL:
    prompt | model.with_structured_output(EntidadesTecnicas) + .with_retry()
    """
    model = get_model()
    
    # Forzamos la salida estructurada mapeada al esquema Pydantic
    structured_model = model.with_structured_output(EntidadesTecnicas)

    # Composición con el operador pipe | y reintentos automáticos
    chain = (prompt | structured_model).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    return chain

async def process_text(text: str) -> EntidadesTecnicas:
    """Función asíncrona principal. Ejecuta la cadena mediante .ainvoke()."""
    chain = build_chain()
    logger.info(f"Ejecutando cadena LCEL asíncrona mediante .ainvoke() ({len(text)} caracteres)...")
    
    try:
        resultado = await chain.ainvoke({"texto": text})
        logger.info("✅ Validación Pydantic exitosa.")
        return resultado
    except Exception as e:
        logger.error(f"❌ Falló tras agotar reintentos (.with_retry): {e}")
        raise