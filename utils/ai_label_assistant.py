"""
AI-assisted labeling tool for DeBERTa training data.
Uses the existing LLM to suggest labels, which can then be reviewed and corrected.
"""
from models.llm_evaluator import LLMEvaluator
import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define the complete label categories - All 14 outpoints and 14 pluspoints
OUTPOINT_LABELS = [
    "OMITTED_DATA_OUT",
    "ALTERED_SEQUENCE_OUT",
    "DROPPED_TIME_OUT",
    "FALSEHOOD_OUT",
    "ALTERED_IMPORTANCE_OUT",
    "WRONG_TARGET_OUT",
    "WRONG_SOURCE_OUT",
    "CONTRARY_FACTS_OUT",
    "ADDED_TIME_OUT",
    "ADDED_INAPPLICABLE_DATA_OUT",
    "INCORRECTLY_INCLUDED_DATUM_OUT",
    "ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT",
    "ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT",
    "ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT"
]

PLUSPOINT_LABELS = [
    "RELATED_FACTS_KNOWN_PLUS",
    "EVENTS_IN_CORRECT_SEQUENCE_PLUS",
    "TIME_NOTED_PLUS",
    "DATA_PROVEN_FACTUAL_PLUS",
    "CORRECT_RELATIVE_IMPORTANCE_PLUS",
    "EXPECTED_TIME_PERIOD_PLUS",
    "ADEQUATE_DATA_PLUS",
    "APPLICABLE_DATA_PLUS",
    "CORRECT_SOURCE_PLUS",
    "CORRECT_TARGET_PLUS",
    "DATA_IN_SAME_CLASSIFICATION_PLUS",
    "IDENTITIES_ARE_IDENTICAL_PLUS",
    "SIMILARITIES_ARE_SIMILAR_PLUS",
    "DIFFERENCES_ARE_DIFFERENT_PLUS"
]

# Paired categories (outpoint/pluspoint pairs that are opposite ends of scales)
PAIRED_CATEGORIES = {
    "OMITTED_DATA_OUT": "ADEQUATE_DATA_PLUS",
    "ALTERED_SEQUENCE_OUT": "EVENTS_IN_CORRECT_SEQUENCE_PLUS",
    "DROPPED_TIME_OUT": "TIME_NOTED_PLUS",
    "FALSEHOOD_OUT": "DATA_PROVEN_FACTUAL_PLUS",
    "ALTERED_IMPORTANCE_OUT": "CORRECT_RELATIVE_IMPORTANCE_PLUS",
    "WRONG_TARGET_OUT": "CORRECT_TARGET_PLUS",
    "WRONG_SOURCE_OUT": "CORRECT_SOURCE_PLUS",
    "ADDED_TIME_OUT": "EXPECTED_TIME_PERIOD_PLUS",
    "ADDED_INAPPLICABLE_DATA_OUT": "APPLICABLE_DATA_PLUS",
    "INCORRECTLY_INCLUDED_DATUM_OUT": "DATA_IN_SAME_CLASSIFICATION_PLUS",
    "ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT": "IDENTITIES_ARE_IDENTICAL_PLUS",
    "ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT": "SIMILARITIES_ARE_SIMILAR_PLUS",
    "ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT": "DIFFERENCES_ARE_DIFFERENT_PLUS"
}


def get_ai_suggestion(statement: str, evaluator: LLMEvaluator) -> str:
    """Get AI suggestion for statement label."""
    try:
        # Use the holistic evaluation method
        result = evaluator.holistic_evaluate(statement)

        # Determine the most appropriate label
        outpoints = result.get('outpoints', [])
        pluspoints = result.get('pluspoints', [])

        if outpoints:
            # Convert first outpoint to label format
            outpoint = outpoints[0].upper().replace(' ', '_')
            return f"{outpoint}_OUT"
        elif pluspoints:
            # Convert first pluspoint to label format
            pluspoint = pluspoints[0].upper().replace(' ', '_')
            return f"{pluspoint}_PLUS"
        else:
            return "NEUTRAL"

    except Exception as e:
        print(f"Error getting AI suggestion: {e}")
        return "NEUTRAL"


def interactive_labeling(df: pd.DataFrame, evaluator: LLMEvaluator, start_idx: int = 0):
    """Interactive labeling with AI assistance."""
    print("\n=== AI-Assisted Labeling Tool ===")
    print("Commands: 'a' = accept AI suggestion, 'c' = custom label, 's' = skip, 'q' = quit")
    print("Available labels:")
    print("Outpoints:", ", ".join(OUTPOINT_LABELS[:7]) + "...")
    print("Pluspoints:", ", ".join(PLUSPOINT_LABELS[:7]) + "...")
    print("Neutral: NEUTRAL")
    print("=" * 60)

    for i in range(start_idx, len(df)):
        statement = df.iloc[i]['text']
        current_label = df.iloc[i]['label']

        print(f"\n[{i+1}/{len(df)}] Statement:")
        print(f"'{statement}'")

        if current_label:
            print(f"Current label: {current_label}")

        # Get AI suggestion
        print("\nGetting AI suggestion...")
        ai_suggestion = get_ai_suggestion(statement, evaluator)
        print(f"AI suggests: {ai_suggestion}")

        # Get user input
        while True:
            choice = input("\nAction [a/c/s/q]: ").lower().strip()

            if choice == 'a':
                df.iloc[i, df.columns.get_loc('label')] = ai_suggestion
                print(f"✓ Accepted: {ai_suggestion}")
                break
            elif choice == 'c':
                custom_label = input("Enter custom label: ").strip()
                if custom_label:
                    df.iloc[i, df.columns.get_loc('label')] = custom_label
                    print(f"✓ Set custom: {custom_label}")
                    break
            elif choice == 's':
                print("⏭ Skipped")
                break
            elif choice == 'q':
                print("Quitting...")
                return i
            else:
                print("Invalid choice. Use 'a', 'c', 's', or 'q'")

    return len(df)


def batch_label_with_ai(df: pd.DataFrame, evaluator: LLMEvaluator):
    """Automatically label all statements with AI suggestions."""
    print("Batch labeling with AI...")

    for i in range(len(df)):
        if df.iloc[i]['label']:  # Skip if already labeled
            continue

        statement = df.iloc[i]['text']
        print(f"[{i+1}/{len(df)}] Processing: {statement[:50]}...")

        suggestion = get_ai_suggestion(statement, evaluator)
        df.iloc[i, df.columns.get_loc('label')] = suggestion

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1} statements...")

    print("Batch labeling complete!")


def main():
    parser = argparse.ArgumentParser(
        description="AI-assisted labeling for DeBERTa training")
    parser.add_argument(
        "--data_path", default="data/statements_to_label.csv", help="Path to CSV file")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                        help="Labeling mode")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index for interactive mode")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM initialization (for testing)")

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.data_path):
        print(
            f"Error: {args.data_path} not found. Run clean_scraped_data.py first.")
        return

    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} statements from {args.data_path}")

    # Initialize LLM evaluator
    if not args.no_llm:
        try:
            print("Initializing LLM evaluator...")
            evaluator = LLMEvaluator(
                model_name="truth-evaluator",
                api_url="http://localhost:11434/api/generate",
                use_rag=True
            )
            print("✓ LLM evaluator ready")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("Make sure Ollama is running with the truth-evaluator model")
            return
    else:
        evaluator = None
        print("Skipping LLM initialization")

    # Run labeling
    if args.mode == "interactive" and evaluator:
        last_idx = interactive_labeling(df, evaluator, args.start_idx)
        print(f"Processed up to index {last_idx}")
    elif args.mode == "batch" and evaluator:
        batch_label_with_ai(df, evaluator)
    else:
        print("No labeling performed (LLM required for labeling)")

    # Save results
    df.to_csv(args.data_path, index=False)
    print(f"Saved results to {args.data_path}")

    # Show label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        if label:  # Skip empty labels
            print(f"  {label}: {count}")

    empty_labels = (df['label'] == '').sum()
    if empty_labels > 0:
        print(f"  (unlabeled): {empty_labels}")


if __name__ == "__main__":
    main()
