from ingest import run_ingestion
from rag_system import RAGSystem
from evaluate import run_evaluation


def main():
    print("============================================================")
    print("🚀 PRE-ENTREGA 4: SISTEMA RAG ESCALABLE EN LA NUBE CON PINECONE")
    print("============================================================\n")

    # 1. Ejecutar Ingesta
    print("1️⃣ Verificando e indexando documentos en Pinecone Serverless...")
    run_ingestion()

    # 2. Prueba de Consulta Híbrida
    print("\n2️⃣ Ejecutando consulta de prueba con Búsqueda Híbrida (BM25 + Pinecone)...")
    rag = RAGSystem()
    pregunta = "¿Qué ventaja ofrece Pinecone Serverless en producción?"
    resultado = rag.query(pregunta)

    print(f"\n❓ Pregunta: {pregunta}")
    print(f"💬 Respuesta: {resultado['respuesta']}")
    print(f"📚 Fuentes recuperadas: {resultado['fuentes']}")

    # 3. Evaluación de Métricas
    print("\n3️⃣ Evaluando métricas del sistema con Golden Set...")
    run_evaluation()


if __name__ == "__main__":
    main()