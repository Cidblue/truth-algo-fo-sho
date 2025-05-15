from typing import List, Dict, Tuple, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from .rag_implementation import init_embedding_model

class StatementComparator:
    """Compares statements to identify relationships, contradictions, and supporting evidence."""
    
    def __init__(self, embedding_model=None, similarity_threshold=0.65):
        """Initialize with optional embedding model."""
        self.embedding_model = embedding_model or init_embedding_model()
        self.similarity_threshold = similarity_threshold
        # Load spaCy for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        
    def find_related_statements(self, target_stmt: str, all_statements: List[str]) -> List[Tuple[str, float]]:
        """Find statements related to the target statement based on semantic similarity."""
        # Encode all statements
        target_embedding = self.embedding_model.encode([target_stmt])[0]
        all_embeddings = self.embedding_model.encode(all_statements)
        
        # Calculate similarities
        similarities = np.dot(all_embeddings, target_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(target_embedding)
        )
        
        # Return statements with similarity above threshold
        related = [(stmt, sim) for stmt, sim in zip(all_statements, similarities) 
                  if sim > self.similarity_threshold and stmt != target_stmt]
        
        # Sort by similarity (highest first)
        related.sort(key=lambda x: x[1], reverse=True)
        
        return related
    
    def extract_key_entities(self, statement: str) -> Set[str]:
        """Extract key entities from a statement for comparison."""
        doc = self.nlp(statement)
        # Extract named entities, nouns, and numbers
        entities = {ent.text.lower() for ent in doc.ents}
        nouns = {token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")}
        numbers = {token.text for token in doc if token.like_num}
        
        return entities.union(nouns).union(numbers)
    
    def detect_potential_contradictions(self, stmt1: str, stmt2: str) -> Dict:
        """Detect if two statements potentially contradict each other."""
        # Extract entities from both statements
        entities1 = self.extract_key_entities(stmt1)
        entities2 = self.extract_key_entities(stmt2)
        
        # If they share entities, they might be related
        common_entities = entities1.intersection(entities2)
        
        if not common_entities:
            return {"contradiction": False, "confidence": 0.0, "reason": "No common entities"}
        
        # Look for opposing sentiment or contradictory claims
        # This is a simplified approach - in practice, you'd want more sophisticated NLP
        contradictory_pairs = [
            ("increase", "decrease"), ("rise", "fall"), ("more", "less"),
            ("higher", "lower"), ("positive", "negative"), ("true", "false"),
            ("confirm", "deny"), ("support", "oppose"), ("agree", "disagree")
        ]
        
        doc1 = self.nlp(stmt1.lower())
        doc2 = self.nlp(stmt2.lower())
        
        # Extract lemmatized tokens
        tokens1 = {token.lemma_ for token in doc1}
        tokens2 = {token.lemma_ for token in doc2}
        
        # Check for contradictory pairs
        for word1, word2 in contradictory_pairs:
            if (word1 in tokens1 and word2 in tokens2) or (word1 in tokens2 and word2 in tokens1):
                return {
                    "contradiction": True, 
                    "confidence": 0.7,  # Medium confidence for this simple approach
                    "reason": f"Found contradictory terms: {word1} vs {word2}",
                    "common_entities": list(common_entities)
                }
        
        # Return low confidence result if no clear contradiction found
        return {
            "contradiction": False,
            "confidence": 0.3,
            "reason": "Common entities but no clear contradiction",
            "common_entities": list(common_entities)
        }