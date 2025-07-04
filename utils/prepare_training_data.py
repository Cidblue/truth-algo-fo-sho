"""
Script to prepare training data for DeBERTa fine-tuning.
Converts raw statements to a labeled CSV file.
"""
import pandas as pd
import re
import argparse
import os
from typing import List, Dict, Any

def extract_statements(file_path: str) -> List[str]:
    """Extract statements from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple statement extraction - split by newlines and filter empty lines
    statements = [line.strip() for line in text.split('\n') if line.strip()]
    return statements

def label_statements_manually(statements: List[str], output_path: str):
    """
    Create a CSV file with statements for manual labeling.
    """
    df = pd.DataFrame({
        'text': statements,
        'label': [''] * len(statements)  # Empty labels for manual filling
    })
    
    df.to_csv(output_path, index=False)
    print(f"Created CSV file with {len(statements)} statements at {output_path}")
    print("Please manually add labels to the 'label' column.")

def main(args):
    # Extract statements
    statements = extract_statements(args.input_file)
    
    # Create CSV for manual labeling
    label_statements_manually(statements, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for DeBERTa")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw statements file")
    parser.add_argument("--output_file", type=str, default="data/statements_to_label.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    main(args)