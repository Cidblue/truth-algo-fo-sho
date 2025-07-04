from typing import Dict, List, Tuple, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import os


class DeBERTaClassifier:
    """
    DeBERTa classifier for outpoint/pluspoint detection.
    Uses a fine-tuned DeBERTa model to classify statements.
    """

    def __init__(self, model_dir: str, threshold: float = 0.75):
        """
        Initialize the DeBERTa classifier.

        Args:
            model_dir: Path to the fine-tuned model directory
            threshold: Confidence threshold for accepting predictions
        """
        self.threshold = threshold
        self.model_dir = model_dir
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer if the model directory exists
        if os.path.exists(model_dir):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)

                # Create pipeline
                self.pipeline = TextClassificationPipeline(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device.index if self.device.type == "cuda" else -1,
                    top_k=None,
                    function_to_apply="softmax"
                )

                # Get label mapping from model config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                self.loaded = True
                print(
                    f"DeBERTa model loaded from {model_dir} with {len(self.id2label)} labels")
            except Exception as e:
                print(f"Error loading DeBERTa model: {e}")
                self.loaded = False
        else:
            print(
                f"Model directory {model_dir} not found. Using placeholder implementation.")
            self.loaded = False

    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """
        Classify text using the DeBERTa model.

        Returns:
            Tuple of (label, confidence) or (None, confidence) if below threshold
        """
        if not self.loaded:
            return None, 0.0

        # Run the model
        try:
            results = self.pipeline(text)[0]  # Get top prediction
            best_result = max(results, key=lambda x: x["score"])
            label, confidence = best_result["label"], best_result["score"]

            # Return None if below threshold
            if confidence < self.threshold:
                return None, confidence

            return label, confidence
        except Exception as e:
            print(f"Error during DeBERTa classification: {e}")
            return None, 0.0
