import os
from models.rag_implementation import extract_and_segment_documents, create_embeddings, VectorStore


def build_vector_store():
    """Build and save the vector store from documents."""
    # Define paths to your documents
    document_paths = []

    # Check if files exist before adding them to the list
    potential_documents = [
        "Investigations.txt",  # Main document on investigation methodology
        "reference_guide.md",  # Condensed guide of outpoints and pluspoints
        "Admin Dictionary.txt",  # Administrative dictionary
        "Tech Dictionary.txt",  # Technical dictionary
        "NGVManagementSeriesV1.txt",  # Management series document
        "Algorithm.txt",  # Conceptual framework for the Truth Algorithm
        "rules_analysis_guide.md",  # New document with holistic analysis guidance
        # Add other potential files here
    ]

    for doc_path in potential_documents:
        if os.path.exists(doc_path):
            document_paths.append(doc_path)
            print(f"Found document: {doc_path}")
        else:
            print(f"Warning: Document not found: {doc_path}")

    if not document_paths:
        print(
            "No documents found. Please ensure your documents are in the correct location.")
        return

    print(f"Processing {len(document_paths)} documents...")

    # Extract and segment documents
    chunks = extract_and_segment_documents(document_paths)

    print(f"Created {len(chunks)} chunks from documents")

    # Create embeddings
    chunks_with_embeddings = create_embeddings(chunks)

    # Create and save vector store
    vector_store = VectorStore()
    vector_store.add_chunks(chunks_with_embeddings)
    vector_store.save("truth_algorithm_vectorstore")

    print(f"Vector store built and saved with {len(chunks)} chunks")
    print(f"Vector store saved to: truth_algorithm_vectorstore")


if __name__ == "__main__":
    build_vector_store()
