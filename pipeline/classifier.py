from typing import Dict, List, Optional, Tuple, Any
from rules.rule_engine import rule_classify
from models.deberta_classifier import DeBERTaClassifier
from models.llm_evaluator import LLMEvaluator


class ClassificationPipeline:
    """
    Multi-stage classification pipeline that combines:
    1. Rule-based classification (high precision)
    2. DeBERTa model (efficient ML classification)
    3. LLM with RAG (fallback for uncertain cases)
    """

    def __init__(
        self,
        deberta_model_dir: str = "models/round2-simple",
        deberta_threshold: float = 0.75,
        use_llm_fallback: bool = True,
        llm_model_name: str = "truth-evaluator",
        llm_api_url: str = "http://localhost:11434/api/generate",
        llm_cache_file: str = "llm_cache.pkl"
    ):
        """Initialize the classification pipeline with all components."""
        # Initialize DeBERTa classifier
        self.deberta = DeBERTaClassifier(
            model_dir=deberta_model_dir,
            threshold=deberta_threshold
        )

        # Initialize LLM evaluator (if using fallback)
        self.use_llm_fallback = use_llm_fallback
        if use_llm_fallback:
            self.llm = LLMEvaluator(
                model_name=llm_model_name,
                api_url=llm_api_url,
                cache_file=llm_cache_file,
                use_rag=True
            )

        self.stats = {
            "rule_hits": 0,
            "deberta_hits": 0,
            "llm_hits": 0,
            "total": 0
        }

    def classify(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify text using the multi-stage pipeline.

        Args:
            text: The text to classify
            context: Optional context for LLM evaluation

        Returns:
            Dict with classification results
        """
        self.stats["total"] += 1

        # Stage 1: Rule-based classification
        rule_results = rule_classify(text)
        if rule_results:
            # Take highest confidence match
            label, confidence = rule_results[0]
            self.stats["rule_hits"] += 1
            return {
                "label": label,
                "confidence": confidence,
                "method": "rules",
                "all_matches": rule_results
            }

        # Stage 2: DeBERTa classification
        label, confidence = self.deberta.classify(text)
        if label:
            self.stats["deberta_hits"] += 1
            return {
                "label": label,
                "confidence": confidence,
                "method": "deberta"
            }

        # Stage 3: LLM fallback (if enabled)
        if self.use_llm_fallback and label is not None:  # Check if DeBERTa returned a label
            # Use your existing LLM evaluation logic
            if "_OUT" in label:
                rule_name = label.replace("_OUT", "").lower()
                has_outpoint, llm_confidence = self.llm.evaluate_outpoint(
                    rule_name, text, context
                )
                if has_outpoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm"
                    }
            elif "_PLUS" in label:
                rule_name = label.replace("_PLUS", "").lower()
                has_pluspoint, llm_confidence = self.llm.evaluate_pluspoint(
                    rule_name, text, context
                )
                if has_pluspoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm"
                    }

        # No confident classification
        return {
            "label": None,
            "confidence": confidence if label is not None else 0.0,
            "method": "uncertain"
        }

    def classify_with_deberta(self, text: str, related_texts: List[str] = None) -> Dict[str, Any]:
        """
        Classify text using DeBERTa with related context.

        Args:
            text: The text to classify
            related_texts: Optional list of related statements for context

        Returns:
            Dict with classification results
        """
        # For now, just use the regular DeBERTa classify method
        # In the future, this could incorporate the related_texts for better context
        label, confidence = self.deberta.classify(text)

        if label and confidence > self.deberta.threshold:
            return {
                "label": label,
                "confidence": confidence,
                "method": "deberta",
                "reasoning": f"DeBERTa classification with {confidence:.2f} confidence"
            }

        return {
            "label": None,
            "confidence": confidence,
            "method": "deberta_below_threshold"
        }

    def classify_with_comparisons(self, text: str, comparison_texts: List[str] = None) -> Dict[str, Any]:
        """
        Classify text using the multi-stage pipeline with comparison to other statements.

        Args:
            text: The text to classify
            comparison_texts: Optional list of related statements for comparison

        Returns:
            Dict with classification results
        """
        self.stats["total"] += 1

        # Stage 1: Rule-based classification
        rule_results = rule_classify(text)
        if rule_results:
            # Take highest confidence match
            label, confidence = rule_results[0]
            self.stats["rule_hits"] += 1
            return {
                "label": label,
                "confidence": confidence,
                "method": "rules",
                "all_matches": rule_results
            }

        # Stage 2: DeBERTa classification
        label, confidence = self.deberta.classify(text)
        if label and confidence > self.deberta.threshold:
            self.stats["deberta_hits"] += 1
            return {
                "label": label,
                "confidence": confidence,
                "method": "deberta"
            }

        # Stage 3: LLM with comparisons (if enabled and comparisons provided)
        if self.use_llm_fallback and comparison_texts:
            if "_OUT" in label:
                rule_name = label.replace("_OUT", "").lower()
                has_outpoint, llm_confidence, reasoning = self.llm.evaluate_statement_with_comparisons(
                    rule_name, text, comparison_texts
                )
                if has_outpoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm_with_comparisons",
                        "reasoning": reasoning
                    }
            elif "_PLUS" in label:
                rule_name = label.replace("_PLUS", "").lower()
                has_pluspoint, llm_confidence, reasoning = self.llm.evaluate_statement_with_comparisons(
                    rule_name, text, comparison_texts
                )
                if has_pluspoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm_with_comparisons",
                        "reasoning": reasoning
                    }

        # Fall back to regular LLM evaluation if comparisons didn't yield a result
        if self.use_llm_fallback:
            # Use existing LLM evaluation logic
            if "_OUT" in label:
                rule_name = label.replace("_OUT", "").lower()
                has_outpoint, llm_confidence = self.llm.evaluate_outpoint(
                    rule_name, text
                )
                if has_outpoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm"
                    }
            elif "_PLUS" in label:
                rule_name = label.replace("_PLUS", "").lower()
                has_pluspoint, llm_confidence = self.llm.evaluate_pluspoint(
                    rule_name, text
                )
                if has_pluspoint:
                    self.stats["llm_hits"] += 1
                    return {
                        "label": label,
                        "confidence": llm_confidence,
                        "method": "llm"
                    }

        # If no classification was made, return the DeBERTa result with low confidence
        return {
            "label": label,
            "confidence": confidence,
            "method": "deberta_low_confidence"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about pipeline usage."""
        stats = self.stats.copy()
        if stats["total"] > 0:
            stats["rule_pct"] = stats["rule_hits"] / stats["total"] * 100
            stats["deberta_pct"] = stats["deberta_hits"] / stats["total"] * 100
            stats["llm_pct"] = stats["llm_hits"] / stats["total"] * 100
        return stats

    def classify_statement(statement_text, context=None, use_llm=False, use_deberta=False):
        """
        Classify a statement using only the specified layers.

        Args:
            statement_text (str): The text of the statement to classify
            context (dict, optional): Additional context for the statement
            use_llm (bool): Whether to use the LLM layer
            use_deberta (bool): Whether to use the DeBERTa layer

        Returns:
            dict: Classification results with outpoints, pluspoints and confidence
        """
        from rules.rule_engine import classify_with_rules

        # Start with rule-based classification (always run this layer)
        rule_results = classify_with_rules(statement_text)

        # If we're only using the rule layer, return those results directly
        if not use_llm and not use_deberta:
            return {
                "outpoints": rule_results["outpoints"],
                "pluspoints": rule_results["pluspoints"],
                "confidence": rule_results["confidence"],
                "method": "rules_only"
            }

        # Add other layers as needed (code omitted for brevity)
        # ...

        return rule_results
