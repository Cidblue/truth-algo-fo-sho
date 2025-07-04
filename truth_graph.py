from typing import List, Dict, Set
import datetime
from models.statement_comparator import StatementComparator


class Statement:
    """Represents a single statement with its analysis results."""

    def __init__(self, id: str, text: str, source: str = None, timestamp: str = None, metadata: Dict = None):
        """Initialize a statement with its text and optional metadata."""
        self.id = id
        self.text = text
        self.source = source
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.outpoints = []  # List of identified outpoints
        self.pluspoints = []  # List of identified pluspoints
        self.truth_score = 0.0  # Overall truth score
        self.related_statements = set()  # IDs of related statements
        self.contradictions = set()  # IDs of contradicting statements
        self.supporting = set()  # IDs of supporting statements

    def to_dict(self):
        """Convert Statement object to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime.datetime) else self.timestamp,
            "outpoints": self.outpoints,
            "pluspoints": self.pluspoints,
            "truth_score": self.truth_score
        }

    def calculate_truth_score(self):
        """Calculate the truth score based on outpoints and pluspoints."""
        # Start with a neutral score
        base_score = 0.5

        # Subtract for each outpoint (more weight for critical outpoints)
        outpoint_penalty = sum(0.2 if op in ["falsehood", "altered_sequence", "wrong_target"]
                               else 0.1 for op in self.outpoints)

        # Add for each pluspoint
        pluspoint_bonus = sum(0.1 for _ in self.pluspoints)

        # Calculate final score and clamp between 0 and 1
        self.truth_score = max(
            0.0, min(1.0, base_score - outpoint_penalty + pluspoint_bonus))
        return self.truth_score


class TruthGraph:
    """Represents a graph of statements and their relationships."""

    def __init__(self):
        """Initialize an empty truth graph."""
        self.stmts = []  # List of Statement objects
        self.stmt_map = {}  # Map of statement ID to index in stmts list
        self.comparator = StatementComparator()

    def add_statement(self, stmt):
        """Add a statement to the graph."""
        self.stmts.append(stmt)
        self.stmt_map[stmt.id] = len(self.stmts) - 1

    def get_statement(self, stmt_id):
        """Get a statement by ID."""
        if stmt_id in self.stmt_map:
            return self.stmts[self.stmt_map[stmt_id]]
        return None

    def add_relation(self, stmt_id1, stmt_id2, relation_type):
        """Add a relation between two statements."""
        stmt1 = self.get_statement(stmt_id1)
        stmt2 = self.get_statement(stmt_id2)

        if not stmt1 or not stmt2:
            return False

        if relation_type == "related":
            stmt1.related_statements.add(stmt_id2)
            stmt2.related_statements.add(stmt_id1)
        elif relation_type == "contradiction":
            stmt1.contradictions.add(stmt_id2)
            stmt2.contradictions.add(stmt_id1)
        elif relation_type == "supporting":
            stmt1.supporting.add(stmt_id2)
            stmt2.related_statements.add(stmt_id1)

        return True

    def find_related_statements(self, stmt_id, max_depth=1):
        """Find all statements related to the given statement up to max_depth."""
        if max_depth <= 0:
            return set()

        stmt = self.get_statement(stmt_id)
        if not stmt:
            return set()

        related = set(stmt.related_statements)

        if max_depth > 1:
            for rel_id in stmt.related_statements:
                related.update(self.find_related_statements(
                    rel_id, max_depth - 1))

        return related

    def to_dict(self):
        """Convert the graph to a dictionary for serialization."""
        return {
            "statements": [stmt.to_dict() for stmt in self.stmts]
        }

    def get_related_statements(self, statement_id):
        """Get statements related to the given statement."""
        stmt = self.get_statement(statement_id)
        if not stmt:
            return []

        # Get all other statements
        other_statements = [(s.id, s.text)
                            for s in self.stmts if s.id != statement_id]
        if not other_statements:
            return []

        # Find related statements using semantic similarity
        other_ids, other_texts = zip(*other_statements)
        related = self.comparator.find_related_statements(
            stmt.text, list(other_texts))

        # Map back to statement objects
        result = []
        for text, similarity in related:
            idx = other_texts.index(text)
            related_id = other_ids[idx]
            related_stmt = self.get_statement(related_id)
            if related_stmt:
                result.append((related_stmt, similarity))

        # Sort by similarity
        result.sort(key=lambda x: x[1], reverse=True)

        return [stmt for stmt, _ in result]

    def find_contradictions(self):
        """Find contradictory statements in the graph."""
        # Get all statement texts and IDs
        all_statements = [(stmt.id, stmt.text) for stmt in self.stmts]

        # Compare each pair of statements
        for i, (id1, text1) in enumerate(all_statements):
            for id2, text2 in all_statements[i+1:]:
                # Skip if already marked as contradictions
                stmt1 = self.get_statement(id1)
                stmt2 = self.get_statement(id2)
                if id2 in stmt1.contradictions:
                    continue

                # Check for potential contradiction
                result = self.comparator.detect_potential_contradictions(
                    text1, text2)

                if result["contradiction"] and result["confidence"] > 0.5:
                    # Add contradiction relationship
                    self.add_relation(id1, id2, "contradiction")

                    # Add contrary_facts outpoint to both statements
                    if "contrary_facts" not in stmt1.outpoints:
                        stmt1.outpoints.append("contrary_facts")
                    if "contrary_facts" not in stmt2.outpoints:
                        stmt2.outpoints.append("contrary_facts")

                    print(
                        f"Found contradiction between {id1} and {id2}: {result['reason']}")

    def find_clusters(self):
        """Find clusters of related statements."""
        # Implementation depends on your clustering logic
        # This is a placeholder
        pass

    def calculate_truth_score(self, statement_id):
        """Calculate truth score for a specific statement."""
        if statement_id not in self.statements:
            return 0.5  # Default neutral score

        statement = self.statements[statement_id]

        # Start with a neutral score
        base_score = 0.5

        # Subtract for each outpoint (more weight for critical outpoints)
        outpoint_penalty = sum(0.2 if op in ["falsehood", "altered_sequence", "wrong_target"]
                               else 0.1 for op in statement.outpoints)

        # Add for each pluspoint
        pluspoint_bonus = sum(0.1 for _ in statement.pluspoints)

        # Calculate final score and clamp between 0 and 1
        truth_score = max(
            0.0, min(1.0, base_score - outpoint_penalty + pluspoint_bonus))

        return truth_score


def noun_overlap(text1, text2):
    """Check if two texts share significant noun phrases."""
    # Simple implementation - check for word overlap
    # In a real implementation, you would use NLP to extract and compare noun phrases
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "but",
                  "in", "on", "at", "to", "for", "with", "by"}
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    # Calculate overlap
    overlap = words1.intersection(words2)

    # Return True if there's significant overlap
    return len(overlap) >= 3
