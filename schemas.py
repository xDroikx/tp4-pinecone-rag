from pydantic import BaseModel, Field
from typing import List


class RespuestaRAG(BaseModel):
    respuesta: str = Field(
        ...,
        description="Respuesta basada estrictamente en el contexto recuperado. Si la información no está presente, indica explícitamente que no se dispone de dicha información."
    )
    fuentes: List[str] = Field(
        default_factory=list,
        description="Lista de nombres de documentos o fuentes utilizadas para responder."
    )