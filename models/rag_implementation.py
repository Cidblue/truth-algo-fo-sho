import os
import re
import numpy as np
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def extract_and_segment_documents(file_paths, chunk_size=500, chunk_overlap=50):
    """Extract content from files and split into chunks."""
    documents = []

    # Read all documents
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append({"text": content, "source": file_path})

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split documents into chunks
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc["text"])
        for chunk in doc_chunks:
            chunks.append({
                "text": chunk,
                "source": doc["source"]
            })

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Create embeddings for text chunks using sentence-transformers."""
    # Load the embedding model
    model = SentenceTransformer(model_name)

    # Create embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    print(f"Created embeddings with dimension {len(embeddings[0])}")
    return chunks


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None

    def add_chunks(self, chunks_with_embeddings):
        """Add chunks with embeddings to the vector store."""
        self.chunks = chunks_with_embeddings

        # Get embedding dimension
        self.dimension = len(chunks_with_embeddings[0]["embedding"])

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Add embeddings to index
        embeddings = np.array([chunk["embedding"]
                              for chunk in chunks_with_embeddings]).astype('float32')
        self.index.add(embeddings)

        print(f"Added {len(chunks_with_embeddings)} chunks to vector store")

    def search(self, query_embedding, k=5):
        """Search for similar chunks."""
        if self.index is None:
            raise ValueError("Vector store is empty. Add chunks first.")

        # Convert query to numpy array with proper shape
        if isinstance(query_embedding, np.ndarray):
            # Ensure it's 2D: reshape if it's 1D
            if query_embedding.ndim == 1:
                query_np = query_embedding.reshape(1, -1).astype('float32')
            else:
                query_np = query_embedding.astype('float32')
        else:
            # If it's not a numpy array, convert it
            query_np = np.array([query_embedding]).astype('float32')

        # Search
        distances, indices = self.index.search(query_np, k)

        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:  # Ensure valid index
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": distances[0][i]
                })

        return results

    def save(self, path):
        """Save vector store to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "dimension": self.dimension
            }, f)

        # Save FAISS index separately
        faiss.write_index(self.index, f"{path}.index")

    @classmethod
    def load(cls, path):
        """Load vector store from disk."""
        store = cls()

        with open(path, 'rb') as f:
            data = pickle.load(f)
            store.chunks = data["chunks"]
            store.dimension = data["dimension"]

        # Load FAISS index
        store.index = faiss.read_index(f"{path}.index")

        return store


def retrieve_context(query, vector_store, embedding_model, max_chunks=3, min_score=0.0):
    """Retrieve relevant context from the vector store."""
    # Enhance query with keywords to improve retrieval
    enhanced_query = enrich_query(query)

    # Generate embedding for the query
    query_embedding = embedding_model.encode([enhanced_query])[0]

    # Search the vector store
    # Get more results than needed to filter
    results = vector_store.search(
        query_embedding, k=max_chunks*5)  # Increased from 3 to 5

    # Log retrieval details
    print(f"RAG Query: '{enhanced_query}' (original: '{query}')")

    # Filter results by similarity score
    filtered_results = []
    for result in results:
        similarity = 1.0 - result["distance"]
        if similarity >= min_score:
            filtered_results.append((result, similarity))

    # Sort by similarity score
    filtered_results.sort(key=lambda x: x[1], reverse=True)

    # Limit to max_chunks
    filtered_results = filtered_results[:max_chunks]

    print(f"Retrieved {len(filtered_results)} chunks from vector store")

    # Format context
    context = ""
    for i, (result, similarity) in enumerate(filtered_results):
        chunk = result["chunk"]
        source = os.path.basename(chunk['source'])
        print(
            f"  Chunk {i+1}: from '{source}', similarity score: {similarity:.4f}")

        context += f"--- Excerpt {i+1} (from {source}) ---\n"
        context += chunk["text"] + "\n\n"

    return context


def enrich_query(query):
    """Enhance query with relevant keywords to improve retrieval."""
    # Extract statement from query if it exists
    statement = query
    if "analyze statement:" in query:
        statement = query.split("analyze statement:")[-1].strip()

    # Check for potential outpoints in the statement
    outpoint_keywords = []
    if "!!" in statement or "everyone knows" in statement.lower():
        outpoint_keywords.append("wrong target generality")
    if "same time" in statement.lower() and ("and" in statement.lower()):
        outpoint_keywords.append("conflicting data")
    if "profit" in statement.lower() and "financial" in statement.lower():
        outpoint_keywords.append("financial data")

    # Create enhanced query
    if outpoint_keywords:
        return f"{query} {' '.join(outpoint_keywords)}"
    return query


def init_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Initialize and return the embedding model."""
    return SentenceTransformer(model_name)


def init_vector_store(path="truth_algorithm_vectorstore"):
    """Initialize and return the vector store from saved file."""
    try:
        return VectorStore.load(path)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None
