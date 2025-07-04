"""
Quick utility to update session status and check current progress.
Helps maintain continuity between sessions.
"""
import os
import sys
import pandas as pd
from datetime import datetime


def get_labeling_progress():
    """Get current labeling progress."""
    csv_path = "data/statements_to_label.csv"
    if not os.path.exists(csv_path):
        return None, "No training data file found"

    df = pd.read_csv(csv_path)
    total = len(df)
    labeled = ((df['label'] != '') & (df['label'].notna())).sum()
    unlabeled = total - labeled

    progress = {
        'total': total,
        'labeled': labeled,
        'unlabeled': unlabeled,
        'percentage': (labeled/total*100) if total > 0 else 0
    }

    # Get label distribution
    if labeled > 0:
        label_counts = df['label'].value_counts()
        progress['distribution'] = dict(label_counts)
    else:
        progress['distribution'] = {}

    return progress, None


def check_system_status():
    """Check if key system components are working."""
    status = {}

    # Check if main files exist
    key_files = [
        "truth_algorithm.py",
        "pipeline/classifier.py",
        "models/deberta_classifier.py",
        "models/train_deberta.py",
        "rules/patterns.yml"
    ]

    for file in key_files:
        status[file] = "✅" if os.path.exists(file) else "❌"

    return status


def generate_status_summary():
    """Generate a quick status summary."""
    print("=" * 60)
    print("TRUTH ALGORITHM - CURRENT STATUS")
    print("=" * 60)

    # Labeling progress
    progress, error = get_labeling_progress()
    if error:
        print(f"❌ Training Data: {error}")
    else:
        print(f"📊 Training Data Progress:")
        print(f"   Total statements: {progress['total']}")
        print(
            f"   Labeled: {progress['labeled']} ({progress['percentage']:.1f}%)")
        print(f"   Unlabeled: {progress['unlabeled']}")

        if progress['labeled'] >= 50:
            print("   ✅ Ready for DeBERTa training!")
        elif progress['labeled'] >= 20:
            print("   ⚠️  Getting close - aim for 50+ labels")
        else:
            print("   🎯 Need more labels for training")

    print()

    # System status
    print("🔧 System Components:")
    file_status = check_system_status()
    for file, status in file_status.items():
        print(f"   {status} {file}")

    print()

    # Next steps
    if progress and progress['labeled'] < 50:
        print("🎯 NEXT STEPS:")
        print("   1. Continue labeling data:")
        print("      python utils/statement_categorizer.py  (GUI - recommended)")
        print("      python utils/manual_labeling_helper.py  (command line)")
        print("   2. Check progress: python utils/update_session_status.py")
    elif progress and progress['labeled'] >= 50:
        print("🎯 NEXT STEPS:")
        print("   1. Train DeBERTa model:")
        print("      pip install datasets transformers torch scikit-learn")
        print(
            "      python models/train_deberta.py --data_path data/statements_to_label.csv")
        print("   2. Test integration:")
        print("      python truth_algorithm.py sample.json -v")

    print("=" * 60)


def update_session_file(notes="", ai_update=False):
    """Update the SESSION_STATUS.md file with current progress."""
    progress, error = get_labeling_progress()
    if error:
        print(f"Cannot update session file: {error}")
        return

    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Read current session file
    session_file = "SESSION_STATUS.md"
    if not os.path.exists(session_file):
        print(f"Session file {session_file} not found")
        return

    with open(session_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update the timestamp
    content = content.replace(
        "**Last Updated**: January 3, 2025",
        f"**Last Updated**: {timestamp}"
    )

    # Update progress numbers
    old_progress = "**Status**: 9/100 statements labeled (9% complete)"
    new_progress = f"**Status**: {progress['labeled']}/{progress['total']} statements labeled ({progress['percentage']:.0f}% complete)"
    content = content.replace(old_progress, new_progress)

    # Add notes if provided
    if notes:
        if ai_update:
            notes_section = f"\n### **AI Session Update** ({timestamp}):\n{notes}\n"
        else:
            notes_section = f"\n### **Session Notes** ({timestamp}):\n{notes}\n"
        # Insert after the "What We Just Accomplished" section
        insert_point = content.find("### **Current System Status**:")
        if insert_point != -1:
            content = content[:insert_point] + \
                notes_section + "\n" + content[insert_point:]

    # Write back
    with open(session_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Updated {session_file}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--update":
            notes = input("Enter session notes (optional): ").strip()
            update_session_file(notes)
        elif sys.argv[1] == "--ai-update":
            if len(sys.argv) > 2:
                notes = " ".join(sys.argv[2:])
                update_session_file(notes, ai_update=True)
                print("✅ AI session update completed")
            else:
                print(
                    "Usage: python utils/update_session_status.py --ai-update 'Your update notes here'")
        elif sys.argv[1] == "--help":
            print("Usage:")
            print(
                "  python utils/update_session_status.py           # Show current status")
            print(
                "  python utils/update_session_status.py --update  # Update session file (interactive)")
            print(
                "  python utils/update_session_status.py --ai-update 'notes'  # AI update with notes")
            print("  python utils/update_session_status.py --help    # Show this help")
        else:
            print("Unknown option. Use --help for usage.")
    else:
        generate_status_summary()


if __name__ == "__main__":
    main()
