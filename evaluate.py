import json
import logging
from rag_system import RAGSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")


def run_evaluation(dataset_path: str = "eval_dataset.json", top_k: int = 5):
    logger.info("🧪 Iniciando evaluación del Recuperador Híbrido...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    rag = RAGSystem(top_k=top_k)

    total_preguntas = len(benchmark)
    recalls = []
    precisions = []

    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE EVALUACIÓN INDIVIDUAL")
    print("=" * 60)

    for item in benchmark:
        pregunta = item["pregunta"]
        esperado = item["documento_esperado"]

        docs = rag.get_relevant_documents(pregunta)
        recuperados_sources = [d.metadata.get("source") for d in docs]

        # Recall@k: ¿El documento esperado apareció entre los K recuperados? (1.0 si sí, 0.0 si no)
        es_recuperado = 1.0 if esperado in recuperados_sources else 0.0
        recalls.append(es_recuperado)

        # Precision@k: Proporción de documentos recuperados que son relevantes
        relevantes_count = recuperados_sources.count(esperado)
        precision = relevantes_count / len(recuperados_sources) if recuperados_sources else 0.0
        precisions.append(precision)

        print(f"\n❓ Pregunta: {pregunta}")
        print(f"🎯 Fuente esperada: {esperado}")
        print(f"📂 Fuentes recuperadas en Top-{top_k}: {recuperados_sources}")
        print(f"📈 Recall@{top_k}: {es_recuperado:.2f} | Precision@{top_k}: {precision:.2f}")

    mean_recall = sum(recalls) / total_preguntas
    mean_precision = sum(precisions) / total_preguntas

    print("\n" + "=" * 60)
    print("🏆 RESUMEN GENERAL DE MÉTRICAS")
    print("=" * 60)
    print(f"Total de preguntas evaluadas: {total_preguntas}")
    print(f"Recall@{top_k} Promedio    : {mean_recall * 100:.2f}%")
    print(f"Precision@{top_k} Promedio : {mean_precision * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
