"""
Clean and prepare scraped BBC data for DeBERTa training.
Removes duplicates, filters quality statements, and prepares for labeling.
"""
import pandas as pd
import re
import argparse
from typing import List, Set
import os

def clean_statement(text: str) -> str:
    """Clean a single statement."""
    # Remove timestamp and source prefix
    # Pattern: (timestamp, source)statement
    cleaned = re.sub(r'^\([^)]+\)\s*', '', text)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Remove very short statements (likely navigation elements)
    if len(cleaned.split()) < 3:
        return None
        
    # Remove common navigation/UI elements
    navigation_patterns = [
        r'^(Follow Live now|See how it played out|Recently Live)$',
        r'^(Live|Breaking|Latest|More)$',
        r'^[A-Z]{2,}\s*$',  # All caps short words
    ]
    
    for pattern in navigation_patterns:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return None
    
    return cleaned

def filter_quality_statements(statements: List[str]) -> List[str]:
    """Filter statements for quality and relevance."""
    quality_statements = []
    seen = set()
    
    for stmt in statements:
        cleaned = clean_statement(stmt)
        if not cleaned:
            continue
            
        # Remove duplicates (case-insensitive)
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        
        # Filter by length (too short or too long)
        word_count = len(cleaned.split())
        if word_count < 5 or word_count > 50:
            continue
            
        # Must contain some substantive content
        if any(word in cleaned.lower() for word in ['said', 'says', 'reported', 'according', 'claims', 'warns', 'announced']):
            quality_statements.append(cleaned)
        elif any(char in cleaned for char in ['.', '!', '?']) and word_count >= 8:
            quality_statements.append(cleaned)
    
    return quality_statements

def create_balanced_sample(statements: List[str], max_per_category: int = 50) -> List[str]:
    """Create a balanced sample for labeling."""
    # For now, just take a random sample
    # In the future, we could try to pre-categorize by topic
    import random
    random.seed(42)
    
    if len(statements) <= max_per_category * 5:  # Rough estimate for 5 categories
        return statements
    
    return random.sample(statements, min(len(statements), max_per_category * 5))

def main():
    parser = argparse.ArgumentParser(description="Clean scraped data for DeBERTa training")
    parser.add_argument("--input", default="docs/Statementstoclass.txt", help="Input file path")
    parser.add_argument("--output", default="data/statements_to_label.csv", help="Output CSV path")
    parser.add_argument("--max_statements", type=int, default=250, help="Maximum statements to include")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read raw statements
    print(f"Reading statements from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_statements = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(raw_statements)} raw statements")
    
    # Clean and filter
    print("Cleaning and filtering statements...")
    quality_statements = filter_quality_statements(raw_statements)
    print(f"After cleaning: {len(quality_statements)} quality statements")
    
    # Create balanced sample
    if len(quality_statements) > args.max_statements:
        print(f"Sampling {args.max_statements} statements for labeling...")
        final_statements = create_balanced_sample(quality_statements, args.max_statements // 5)
    else:
        final_statements = quality_statements
    
    # Create DataFrame with empty labels
    df = pd.DataFrame({
        'text': final_statements,
        'label': [''] * len(final_statements),
        'notes': [''] * len(final_statements)  # For additional notes during labeling
    })
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\nCreated {args.output} with {len(final_statements)} statements")
    print("\nSample statements:")
    for i, stmt in enumerate(final_statements[:5]):
        print(f"{i+1}. {stmt}")
    
    print(f"\nNext steps:")
    print(f"1. Open {args.output} in Excel or a text editor")
    print(f"2. Add labels to the 'label' column using these categories:")
    print("   Outpoints: OMITTED_DATA_OUT, FALSEHOOD_OUT, WRONG_SOURCE_OUT, CONTRARY_FACTS_OUT, etc.")
    print("   Pluspoints: DATA_PROVEN_FACTUAL_PLUS, CORRECT_SOURCE_PLUS, TIME_NOTED_PLUS, etc.")
    print("   Neutral: NEUTRAL (for statements with no clear outpoints or pluspoints)")
    print(f"3. Run: python models/train_deberta.py --data_path {args.output}")

if __name__ == "__main__":
    main()
