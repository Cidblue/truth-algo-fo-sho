"""
Simple manual labeling helper for DeBERTa training data.
Shows unlabeled statements one by one with quick labeling options.
"""
import pandas as pd
import argparse
import os

# Complete label shortcuts - All 14 outpoints and 14 pluspoints
LABEL_SHORTCUTS = {
    # 14 Outpoints (Logical Errors)
    '1': 'OMITTED_DATA_OUT',
    '2': 'ALTERED_SEQUENCE_OUT',
    '3': 'DROPPED_TIME_OUT',
    '4': 'FALSEHOOD_OUT',
    '5': 'ALTERED_IMPORTANCE_OUT',
    '6': 'WRONG_TARGET_OUT',
    '7': 'WRONG_SOURCE_OUT',
    '8': 'CONTRARY_FACTS_OUT',
    '9': 'ADDED_TIME_OUT',
    '10': 'ADDED_INAPPLICABLE_DATA_OUT',
    '11': 'INCORRECTLY_INCLUDED_DATUM_OUT',
    '12': 'ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT',
    '13': 'ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT',
    '14': 'ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT',

    # 14 Pluspoints (Logical Strengths)
    '15': 'RELATED_FACTS_KNOWN_PLUS',
    '16': 'EVENTS_IN_CORRECT_SEQUENCE_PLUS',
    '17': 'TIME_NOTED_PLUS',
    '18': 'DATA_PROVEN_FACTUAL_PLUS',
    '19': 'CORRECT_RELATIVE_IMPORTANCE_PLUS',
    '20': 'EXPECTED_TIME_PERIOD_PLUS',
    '21': 'ADEQUATE_DATA_PLUS',
    '22': 'APPLICABLE_DATA_PLUS',
    '23': 'CORRECT_SOURCE_PLUS',
    '24': 'CORRECT_TARGET_PLUS',
    '25': 'DATA_IN_SAME_CLASSIFICATION_PLUS',
    '26': 'IDENTITIES_ARE_IDENTICAL_PLUS',
    '27': 'SIMILARITIES_ARE_SIMILAR_PLUS',
    '28': 'DIFFERENCES_ARE_DIFFERENT_PLUS',

    # Neutral
    '0': 'NEUTRAL'
}


def show_menu():
    """Display the labeling menu."""
    print("\n" + "="*80)
    print("COMPLETE LABELING MENU - All 14 Outpoints & 14 Pluspoints")
    print("="*80)
    print("OUTPOINTS (Logical Errors):")
    print("  1 - OMITTED_DATA_OUT (missing info)")
    print("  2 - ALTERED_SEQUENCE_OUT (wrong order)")
    print("  3 - DROPPED_TIME_OUT (missing time)")
    print("  4 - FALSEHOOD_OUT (untrue/misleading)")
    print("  5 - ALTERED_IMPORTANCE_OUT (exaggerated/minimized)")
    print("  6 - WRONG_TARGET_OUT (misplaced blame)")
    print("  7 - WRONG_SOURCE_OUT (unreliable source)")
    print("  8 - CONTRARY_FACTS_OUT (contradictory)")
    print("  9 - ADDED_TIME_OUT (unnecessary time)")
    print("  10 - ADDED_INAPPLICABLE_DATA_OUT (irrelevant info)")
    print("  11 - INCORRECTLY_INCLUDED_DATUM_OUT (wrong classification)")
    print("  12 - ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT (false equivalence)")
    print("  13 - ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT (false analogy)")
    print("  14 - ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT (false distinction)")
    print()
    print("PLUSPOINTS (Logical Strengths):")
    print("  15 - RELATED_FACTS_KNOWN_PLUS (supporting evidence)")
    print("  16 - EVENTS_IN_CORRECT_SEQUENCE_PLUS (logical order)")
    print("  17 - TIME_NOTED_PLUS (proper timestamps)")
    print("  18 - DATA_PROVEN_FACTUAL_PLUS (verified facts)")
    print("  19 - CORRECT_RELATIVE_IMPORTANCE_PLUS (proper emphasis)")
    print("  20 - EXPECTED_TIME_PERIOD_PLUS (appropriate timeframe)")
    print("  21 - ADEQUATE_DATA_PLUS (sufficient detail)")
    print("  22 - APPLICABLE_DATA_PLUS (relevant info)")
    print("  23 - CORRECT_SOURCE_PLUS (reliable source)")
    print("  24 - CORRECT_TARGET_PLUS (proper attribution)")
    print("  25 - DATA_IN_SAME_CLASSIFICATION_PLUS (consistent categorization)")
    print("  26 - IDENTITIES_ARE_IDENTICAL_PLUS (proper equivalence)")
    print("  27 - SIMILARITIES_ARE_SIMILAR_PLUS (valid comparison)")
    print("  28 - DIFFERENCES_ARE_DIFFERENT_PLUS (valid distinction)")
    print()
    print("OTHER:")
    print("  0 - NEUTRAL (no clear outpoints/pluspoints)")
    print("  c - Custom label")
    print("  s - Skip")
    print("  q - Quit")
    print("  h - Show this help menu")
    print("="*80)


def manual_labeling(df: pd.DataFrame, start_idx: int = 0):
    """Manual labeling with shortcuts."""
    show_menu()

    unlabeled_indices = df[(df['label'] == '') | (
        df['label'].isna())].index.tolist()
    if start_idx > 0:
        unlabeled_indices = [i for i in unlabeled_indices if i >= start_idx]

    if not unlabeled_indices:
        print("No unlabeled statements found!")
        return len(df)

    print(f"\nFound {len(unlabeled_indices)} unlabeled statements")

    for i, idx in enumerate(unlabeled_indices):
        statement = df.iloc[idx]['text']

        print(f"\n[{i+1}/{len(unlabeled_indices)}] Statement {idx+1}:")
        print("-" * 60)
        print(f"'{statement}'")
        print("-" * 60)

        while True:
            choice = input("Label choice: ").strip()

            if choice in LABEL_SHORTCUTS:
                label = LABEL_SHORTCUTS[choice]
                df.iloc[idx, df.columns.get_loc('label')] = label
                print(f"✓ Labeled as: {label}")
                break
            elif choice.lower() == 'c':
                custom_label = input("Enter custom label: ").strip()
                if custom_label:
                    df.iloc[idx, df.columns.get_loc('label')] = custom_label
                    print(f"✓ Custom label: {custom_label}")
                    break
            elif choice.lower() == 's':
                print("⏭ Skipped")
                break
            elif choice.lower() == 'q':
                print("Quitting...")
                return idx
            elif choice.lower() == 'h':
                show_menu()
            else:
                print(
                    "Invalid choice. Enter number (0-11), 'c', 's', 'q', or 'h' for help")

    return len(df)


def show_progress(df: pd.DataFrame):
    """Show labeling progress."""
    total = len(df)
    labeled = ((df['label'] != '') & (df['label'].notna())).sum()
    unlabeled = total - labeled

    print(f"\nPROGRESS:")
    print(f"  Total statements: {total}")
    print(f"  Labeled: {labeled} ({labeled/total*100:.1f}%)")
    print(f"  Unlabeled: {unlabeled}")

    if labeled > 0:
        print(f"\nLABEL DISTRIBUTION:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            if label:  # Skip empty labels
                print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Manual labeling helper")
    parser.add_argument(
        "--data_path", default="data/statements_to_label.csv", help="Path to CSV file")
    parser.add_argument("--start_idx", type=int,
                        default=0, help="Starting index")
    parser.add_argument("--progress_only",
                        action="store_true", help="Just show progress")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found")
        return

    df = pd.read_csv(args.data_path)

    if args.progress_only:
        show_progress(df)
        return

    print(f"Manual Labeling Helper")
    print(f"Data: {args.data_path}")
    show_progress(df)

    # Start labeling
    last_idx = manual_labeling(df, args.start_idx)

    # Save results
    df.to_csv(args.data_path, index=False)
    print(f"\nSaved results to {args.data_path}")

    # Show final progress
    show_progress(df)

    # Check if ready for training
    labeled_count = (df['label'] != '').sum()
    if labeled_count >= 50:
        print(f"\n🎉 You have {labeled_count} labeled statements!")
        print("You can now train the DeBERTa model:")
        print(f"python models/train_deberta.py --data_path {args.data_path}")
    else:
        print(
            f"\nNeed at least 50 labeled statements for training (currently: {labeled_count})")


if __name__ == "__main__":
    main()
