# truth_algorithm.py
"""Truth Algorithm — COMPLETE rule‑set (14 × outpoints, 14 × pluspoints)
=======================================================================
Implements every outpoint and pluspoint exactly as listed in L. Ron Hubbard’s
*Investigations* booklet (pp. 10‑27).  Heavy semantic checks leverage a public,
unauthenticated MNLI model; structural checks rely on simple heuristics.

Quick‑start
-----------
```bash
pip install networkx transformers torch spacy python-dateutil
python -m spacy download en_core_web_sm
python truth_algorithm.py sample.json -v
```

* sample.json must be a JSON array of statements:
```json
[{"id":"s1","text":"The factory reported record profits in Q1.","time":"2025-01-15","source":"Finance"}, … ]
```
"""
from __future__ import annotations
from rules import Rule, OutpointRule, PluspointRule, RULES_OUT, RULES_PLUS
from truth_graph import TruthGraph, Statement
# Import the LLMEvaluator from the models package instead of directly
from models.llm_evaluator import LLMEvaluator
from pipeline.classifier import ClassificationPipeline

import argparse
import json
import pathlib
import re
import sys
import itertools
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from datetime import datetime as dt
import os
import requests
import pickle
import concurrent.futures

import networkx as nx
from transformers import pipeline
import spacy
from dateutil import parser as dateparse

# ---------------------------------------------------------------------------
#  Config & helpers
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/bart-large-mnli"   # public, ~120 MB
ENTAIL_THR = 0.80
CONTR_THR = 0.80
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
NLP.add_pipe('sentencizer')  # Add sentencizer for sentence segmentation


@lru_cache(maxsize=1)
def _mnli():
    print(f"Loading MNLI ➜ {MODEL_NAME}", file=sys.stderr)
    return pipeline("text-classification", model=MODEL_NAME,
                    tokenizer=MODEL_NAME, device=-1)


def nli(a: str, b: str):
    """Return `(label, score)` where label∈{entailment, contradiction, neutral}."""
    res = _mnli()(f"{a} </s></s> {b}", truncation=True, max_length=512)[0]
    return res["label"].lower(), float(res["score"])


def noun_overlap(a: str, b: str, thresh: int = 2) -> bool:
    doc_a, doc_b = NLP(a), NLP(b)
    nouns_a = {t.text.lower() for t in doc_a if t.pos_ in {"NOUN", "PROPN"}}
    nouns_b = {t.text.lower() for t in doc_b if t.pos_ in {"NOUN", "PROPN"}}
    return len(nouns_a & nouns_b) >= thresh

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Rule base & registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  LLM Integration
# ---------------------------------------------------------------------------

# Add outpoint and pluspoint descriptions at the top of your file
outpoint_descriptions = {
    "falsehood": "A statement that contains information that is factually incorrect or demonstrably untrue.",
    "omitted_data": "A statement that leaves out critical information that would change how the statement is interpreted.",
    "wrong_target": "A statement that incorrectly assigns blame, responsibility, or causation to the wrong person or thing."
}

pluspoint_descriptions = {
    "data_proven_factual": "A statement that contains information that has been verified as factually correct.",
    "related_facts_known": "A statement that demonstrates awareness of relevant contextual information."
}

# Add this class to implement the complete Truth Algorithm


class TruthAlgorithm:
    """Implements the complete Truth Algorithm as described in the conceptual framework."""

    def __init__(self, llm_evaluator=None, confidence_threshold=0.05, use_llm=True, use_deberta=True):
        """Initialize the Truth Algorithm with the classification pipeline."""
        self.confidence_threshold = confidence_threshold
        self.classifier = ClassificationPipeline(
            deberta_model_dir="models/round2-simple",
            deberta_threshold=confidence_threshold,
            use_llm_fallback=use_llm,
            llm_model_name="truth-evaluator"
        )
        self.truth_graph = TruthGraph()
        self.use_llm = use_llm
        self.use_deberta = use_deberta
        self.llm_evaluator = llm_evaluator

    def process_statements(self, statements):
        """Process a list of statements through the complete Truth Algorithm pipeline."""
        # 1. Statement Collection & Preprocessing
        processed_statements = self._preprocess_statements(statements)

        # 2. Individual Statement Analysis
        for stmt in processed_statements:
            self._analyze_statement(stmt)

        # 3. Relational Analysis
        self._perform_relational_analysis()

        # 4. Classification
        classified_statements = self._classify_statements()

        # 5. Why Finding
        whys = self._find_whys(classified_statements)

        # 6. Reporting
        report = self._generate_report(classified_statements, whys)

        # Create a serializable version of the results
        serializable_results = {
            "statements": [stmt.to_dict() for stmt in processed_statements],
            "classified_statements": classified_statements,
            "whys": whys,
            "report": report
        }

        return serializable_results

    def _preprocess_statements(self, statements):
        """Normalize and preprocess statements."""
        processed = []
        for i, stmt_data in enumerate(statements):
            # Create Statement object
            if isinstance(stmt_data, str):
                # If it's just a string, create a simple Statement object
                stmt = Statement(f"stmt_{i}", stmt_data)
            else:
                # If it's a dictionary, extract the fields
                stmt = Statement(
                    stmt_data.get("id", f"stmt_{i}"),
                    stmt_data.get("text", ""),
                    stmt_data.get("source"),
                    stmt_data.get("timestamp"),
                    stmt_data.get("metadata", {})
                )

            # Add to truth graph
            self.truth_graph.add_statement(stmt)
            processed.append(stmt)

        return processed

    def _analyze_statement(self, stmt):
        """Apply outpoint and pluspoint rules to a single statement."""
        # Get related statements for context
        related_statements = self.truth_graph.get_related_statements(stmt.id)
        related_texts = [s.text for s in related_statements]

        # Use the rule-based analysis first (always run this)
        self._apply_rule_analysis(stmt)

        # Only use DeBERTa if enabled
        if self.use_deberta:
            # Use the pipeline to classify with DeBERTa
            deberta_result = self.classifier.classify_with_deberta(
                stmt.text, related_texts)

            if deberta_result["label"]:
                if "_OUT" in deberta_result["label"]:
                    outpoint = deberta_result["label"].replace(
                        "_OUT", "").lower()
                    if outpoint not in stmt.outpoints:
                        stmt.outpoints.append(outpoint)
                        if "reasoning" in deberta_result:
                            stmt.outpoint_reasoning[outpoint] = deberta_result["reasoning"]
                elif "_PLUS" in deberta_result["label"]:
                    pluspoint = deberta_result["label"].replace(
                        "_PLUS", "").lower()
                    if pluspoint not in stmt.pluspoints:
                        stmt.pluspoints.append(pluspoint)
                        if "reasoning" in deberta_result:
                            stmt.pluspoint_reasoning[pluspoint] = deberta_result["reasoning"]

        # Only use LLM if enabled and we have an evaluator
        if self.use_llm and self.llm_evaluator:
            self._apply_llm_analysis(stmt)

    def _apply_rule_analysis(self, stmt):
        """Apply traditional rule-based analysis to a statement."""
        # Use the regex-based rule engine
        from rules.rule_engine import rule_classify

        rule_results = rule_classify(stmt.text)

        for label, confidence in rule_results:
            if "_OUT" in label:
                outpoint = label.replace("_OUT", "").lower()
                if outpoint not in stmt.outpoints:
                    stmt.outpoints.append(outpoint)
            elif "_PLUS" in label:
                pluspoint = label.replace("_PLUS", "").lower()
                if pluspoint not in stmt.pluspoints:
                    stmt.pluspoints.append(pluspoint)

    def _apply_llm_analysis(self, stmt):
        """Apply LLM-based analysis to a statement."""
        # Define the key outpoints and pluspoints to evaluate with LLM
        outpoints_to_evaluate = ["falsehood", "omitted_data", "wrong_target"]
        pluspoints_to_evaluate = ["data_proven_factual", "related_facts_known"]

        # Get related statements for context
        related_statements = self.truth_graph.get_related_statements(stmt.id)
        context = {"related_statements": [s.text for s in related_statements]}

        # Evaluate outpoints
        for rule_name in outpoints_to_evaluate:
            has_outpoint, confidence = self.llm_evaluator.evaluate_outpoint(
                rule_name, stmt.text, context)

            if has_outpoint and confidence >= self.confidence_threshold:
                if rule_name not in stmt.outpoints:
                    stmt.outpoints.append(rule_name)

        # Evaluate pluspoints
        for rule_name in pluspoints_to_evaluate:
            has_pluspoint, confidence = self.llm_evaluator.evaluate_pluspoint(
                rule_name, stmt.text, context)

            if has_pluspoint and confidence >= self.confidence_threshold:
                if rule_name not in stmt.pluspoints:
                    stmt.pluspoints.append(rule_name)

    def _perform_relational_analysis(self):
        """Analyze relationships between statements."""
        # Find contradictions
        self.truth_graph.find_contradictions()

        # Find clusters of related statements
        self.truth_graph.find_clusters()

    def _classify_statements(self):
        """Classify statements based on their truth scores."""
        classified = []

        # Iterate through all statements in the truth graph
        for stmt in self.truth_graph.stmts:
            # Calculate final truth score
            stmt.calculate_truth_score()

            # Determine classification
            if stmt.truth_score >= 0.8:
                classification = "Likely True"
            elif stmt.truth_score >= 0.5:
                classification = "Possibly True"
            elif stmt.truth_score >= 0.3:
                classification = "Possibly False"
            else:
                classification = "Likely False"

            # Add to classified list
            classified.append({
                "id": stmt.id,
                "text": stmt.text,
                "source": stmt.source,
                "timestamp": stmt.timestamp,
                "outpoints": stmt.outpoints,
                "pluspoints": stmt.pluspoints,
                "truth_score": stmt.truth_score,
                "classification": classification
            })

        return classified

    def _find_whys(self, classified_statements):
        """Find potential reasons for outpoints."""
        whys = []

        # Group red statements by common outpoints
        red_statements = [
            s for s in classified_statements if s["classification"] == "Likely False"]

        # Simple implementation - group by most common outpoint
        if red_statements:
            all_outpoints = []
            for stmt in red_statements:
                all_outpoints.extend(stmt["outpoints"])

            if all_outpoints:
                from collections import Counter
                most_common = Counter(all_outpoints).most_common(1)[0][0]

                whys.append({
                    "pattern": most_common,
                    "statements": [s["id"] for s in red_statements if most_common in s["outpoints"]],
                    "explanation": f"Multiple statements exhibit the '{most_common}' outpoint."
                })

        return whys

    def _generate_report(self, classified_statements, whys):
        """Generate a summary report of the analysis."""
        # Count statements by classification
        red_count = sum(
            1 for s in classified_statements if s["classification"] == "Likely False")
        yellow_count = sum(1 for s in classified_statements if s["classification"] in [
                           "Possibly False", "Possibly True"])
        green_count = sum(
            1 for s in classified_statements if s["classification"] == "Likely True")

        # Generate report
        report = {
            "summary": {
                "total_statements": len(classified_statements),
                "red_statements": red_count,
                "yellow_statements": yellow_count,
                "green_statements": green_count
            },
            "whys": whys,
            "recommendations": []
        }

        return report


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truth Algorithm")
    parser.add_argument(
        "input_file", help="JSON file containing statements to analyze")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM-based evaluation")
    parser.add_argument("--no-deberta", action="store_true",
                        help="Disable DeBERTa-based evaluation")
    parser.add_argument("--rules-only", action="store_true",
                        help="Use only rule-based evaluation (equivalent to --no-llm --no-deberta)")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG system")
    parser.add_argument("--max-chunks", type=int, default=3,
                        help="Maximum number of chunks to retrieve from RAG (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum similarity score for RAG chunks (default: 0.0)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for LLM queries (default: 60)")
    args = parser.parse_args()

    # If rules-only is specified, disable both LLM and DeBERTa
    if args.rules_only:
        args.no_llm = True
        args.no_deberta = True

    # Load statements from file
    try:
        with open(args.input_file, 'r') as f:
            statements = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # Initialize LLM evaluator only if needed
    evaluator = None
    if not args.no_llm:
        evaluator = LLMEvaluator(
            model_name="truth-evaluator",
            api_url="http://localhost:11434/api/generate",
            cache_file="llm_cache.pkl",
            use_rag=not args.no_rag
        )

        # Set max_chunks if provided
        if hasattr(evaluator, 'max_chunks'):
            evaluator.max_chunks = args.max_chunks

        # Set min_score for RAG
        evaluator.min_score = args.min_score

        # Set timeout for LLM queries
        evaluator.timeout = args.timeout

    # Initialize and run the algorithm
    algorithm = TruthAlgorithm(
        llm_evaluator=evaluator,
        use_llm=not args.no_llm,
        use_deberta=not args.no_deberta
    )

    # Print configuration
    print(f"Configuration:")
    print(f"  Rule-based evaluation: Enabled")
    print(
        f"  DeBERTa evaluation: {'Disabled' if args.no_deberta else 'Enabled'}")
    print(f"  LLM evaluation: {'Disabled' if args.no_llm else 'Enabled'}")
    if not args.no_llm:
        print(f"  RAG system: {'Disabled' if args.no_rag else 'Enabled'}")
        if not args.no_rag:
            print(f"  Max RAG chunks: {args.max_chunks}")
            print(f"  Min similarity score: {args.min_score}")
        print(f"  LLM timeout: {args.timeout} seconds")

    results = algorithm.process_statements(statements)

    # Print results
    if args.verbose:
        print("\nResults:")
        for stmt in results["classified_statements"]:
            print(f"\n{stmt['id']}: {stmt['text'][:50]}...")
            print(
                f"  Classification: {stmt['classification']} (score: {stmt['truth_score']:.2f})")
            print(
                f"  Outpoints: {', '.join(stmt['outpoints']) if stmt['outpoints'] else 'None'}")
            print(
                f"  Pluspoints: {', '.join(stmt['pluspoints']) if stmt['pluspoints'] else 'None'}")

    # Save results to file
    output_file = pathlib.Path(args.input_file).stem + "_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete. Results saved to {output_file}")
