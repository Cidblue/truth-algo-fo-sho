"""
Script to test the DeBERTa classifier on a set of statements.
"""
import argparse
import pandas as pd
import sys
import os

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deberta_classifier import DeBERTaClassifier

def test_classifier(model_dir: str, test_file: str, threshold: float = 0.75):
    """Test the DeBERTa classifier on a set of statements."""
    # Initialize classifier
    classifier = DeBERTaClassifier(model_dir=model_dir, threshold=threshold)
    
    # Load test data
    if test_file.endswith('.csv'):
        df = pd.read_csv(test_file)
        statements = df['text'].tolist()
        if 'label' in df.columns:
            true_labels = df['label'].tolist()
        else:
            true_labels = None
    else:
        # Assume it's a text file with one statement per line
        with open(test_file, 'r', encoding='utf-8') as f:
            statements = [line.strip() for line in f if line.strip()]
        true_labels = None
    
    # Classify statements
    results = []
    for i, statement in enumerate(statements):
        label, confidence = classifier.classify(statement)
        result = {
            'statement': statement,
            'predicted_label': label,
            'confidence': confidence
        }
        if true_labels:
            result['true_label'] = true_labels[i]
            result['correct'] = label == true_labels[i]
        
        results.append(result)
    
    # Calculate accuracy if true labels are available
    if true_labels:
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nStatement {i+1}: {result['statement']}")
        print(f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
        if true_labels:
            print(f"True label: {result['true_label']}")
            print(f"Correct: {result['correct']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DeBERTa classifier")
    parser.add_argument("--model_dir", type=str, default="models/deberta-lora", help="Path to model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test file (CSV or TXT)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    
    args = parser.parse_args()
    test_classifier(args.model_dir, args.test_file, args.threshold)