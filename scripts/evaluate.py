import json
from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model
from ragflow.retrieval.vector_store import ChromaVectorStore

def main() -> None:
    config = get_config()

    evaluation_dir = resolve_path(config["paths"]["evaluation_data"])

    evaluation_path = evaluation_dir / "sample_questions.json"

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: "
            f"{evaluation_path}"
        )

    

    with evaluation_path.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list):
        raise ValueError(
            "Evaluation data must be a list."
        )

    embedding_model = create_embedding_model(config["embedding"])
    
    vector_store = ChromaVectorStore(
        persist_directory=resolve_path(config["paths"]["vector_store"]),
        collection_name=config["vector_store"]["collection_name"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    passed = 0
    reciprocal_rank_total = 0.0
    modality_stats = {
        "text": {
            "total": 0,
            "passed": 0,
            "reciprocal_rank": 0.0,
        },
        "image": {
            "total": 0,
            "passed": 0,
            "reciprocal_rank": 0.0,
        }
    }

    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]
        expected_page = test_case["expected_page"]
        expected_content_type = test_case["expected_content_type"]
        query_embedding = embedding_model.embed_query(question)
        results = vector_store.search(query_embedding=query_embedding, top_k=5)
        matched_rank = None
        retrieved_sources = []

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            chunk_id = result.get(
                "id",
                metadata.get(
                    "chunk_id",
                    "unknown",
                ),
            )
            actual_page = metadata["page"]
            actual_content_type = metadata["content_type"]
            similarity = result["similarity"]

            retrieved_sources.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "page": actual_page,
                    "content_type": actual_content_type,
                    "similarity": similarity,
                }
            )

            page_matches = actual_page == expected_page
            type_matches = actual_content_type == expected_content_type
            if (
                page_matches
                and type_matches
                and matched_rank is None
            ):
                matched_rank = rank

        modality_stats[expected_content_type]["total"] += 1

        if matched_rank is not None:
            status = "PASS"
            passed += 1

            reciprocal_rank = 1 / matched_rank
            reciprocal_rank_total += reciprocal_rank

            modality_stats[expected_content_type]["passed"] += 1
            modality_stats[expected_content_type]["reciprocal_rank"] += reciprocal_rank

        else:
            status = "FAIL"

        print()
        print(f"Test ID: {test_id}")
        print(f"Question: {question}")

        print(
            f"Expected: "
            f"page={expected_page}, "
            f"type={expected_content_type}"
        )

        print(
            f"Matched rank: {matched_rank}"
        )

        print(f"Result: {status}")
        print("Retrieved sources:")

        for source in retrieved_sources:
            print(
                f"  Rank {source['rank']}: "
                f"{source['chunk_id']} | "
                f"type={source['content_type']} | "
                f"page={source['page']} | "
                f"similarity="
                f"{source['similarity']:.4f}"
            )

    total = len(test_cases)

    hit_rate = (
        passed / total
        if total
        else 0.0
    )

    mean_reciprocal_rank = (
        reciprocal_rank_total / total
        if total
        else 0.0
    )

    print()
    print("=" * 50)
    print("Overall evaluation")
    print("=" * 50)
    print(f"Questions: {total}")
    print(f"Passed: {passed}")
    print(f"Hit@5: {hit_rate:.2%}")
    print(
        f"MRR: "
        f"{mean_reciprocal_rank:.4f}"
    )

    for content_type, stats in (
        modality_stats.items()
    ):
        modality_total = stats["total"]
        modality_passed = stats["passed"]

        modality_hit_rate = (
            modality_passed
            / modality_total
            if modality_total
            else 0.0
        )

        modality_mrr = (
            stats["reciprocal_rank"]
            / modality_total
            if modality_total
            else 0.0
        )

        print()
        print(
            f"{content_type.capitalize()} evaluation"
        )
        print(
            f"Questions: {modality_total}"
        )
        print(
            f"Passed: {modality_passed}"
        )
        print(
            f"Hit@5: "
            f"{modality_hit_rate:.2%}"
        )
        print(
            f"MRR: {modality_mrr:.4f}"
        )


if __name__ == "__main__":
    main()