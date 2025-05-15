import os
from models.llm_evaluator import LLMEvaluator
from models.rag_implementation import init_vector_store, init_embedding_model


def test_rag_integration():
    """Test if the RAG system is properly integrated with the LLM evaluator."""
    print("Testing RAG integration with LLM evaluator...")

    # Initialize components
    vector_store = init_vector_store()
    if not vector_store:
        print("Error: Vector store could not be loaded.")
        return False

    embedding_model = init_embedding_model()

    # Initialize LLM evaluator with RAG enabled
    evaluator = LLMEvaluator(use_rag=True, max_chunks=5)
    evaluator.vector_store = vector_store
    evaluator.embedding_model = embedding_model
    evaluator.min_score = 0.0  # Accept all matches for testing
    evaluator.timeout = 120  # Longer timeout for testing

    # Test statements that should trigger specific RAG content
    test_statements = [
        # Statement that should trigger holistic analysis guidance
        "The factory reported record profits in Q1 while simultaneously laying off 30% of its workforce due to financial difficulties.",

        # Statement with potential outpoints
        "The event happened on Tuesday and Wednesday at the same time.",

        # Statement with obvious outpoints
        "Everyone knows the CEO is embezzling funds!!",

        # Statement with potential pluspoints
        "According to the verified financial records, the company increased revenue by 15% in Q2."
    ]

    success_count = 0
    outpoint_detection_count = 0

    for i, statement in enumerate(test_statements):
        print(f"\nTest {i+1}: Evaluating statement: '{statement[:50]}...'")

        # Evaluate statement
        result = evaluator.evaluate_statement_holistically(statement)

        # Check if RAG sources were used
        if "rag_sources" in result and result["rag_sources"]:
            print(f"  - Success: RAG sources used: {result['rag_sources']}")
            success_count += 1
        else:
            print("  - Failure: No RAG sources were used in evaluation")

        # Check if outpoints were detected
        if result["outpoints"]:
            print(f"  - Success: Outpoints detected: {result['outpoints']}")
            outpoint_detection_count += 1
        else:
            print("  - Note: No outpoints detected")

        print(f"  - Outpoints: {result['outpoints']}")
        print(f"  - Pluspoints: {result['pluspoints']}")
        print(f"  - Confidence: {result['confidence']}")

    # Report results
    rag_success_rate = (success_count / len(test_statements)) * 100
    outpoint_success_rate = (
        outpoint_detection_count / len(test_statements)) * 100

    print(f"\nRAG integration test complete:")
    print(
        f"- RAG usage: {rag_success_rate:.1f}% success rate ({success_count}/{len(test_statements)})")
    print(
        f"- Outpoint detection: {outpoint_success_rate:.1f}% success rate ({outpoint_detection_count}/{len(test_statements)})")

    return success_count > 0 and outpoint_detection_count > 0


if __name__ == "__main__":
    test_rag_integration()
