#!/usr/bin/env python3
"""
Fix the critical issue: We need exactly 28 categories (14 outpoints + 14 pluspoints)
Currently we only have 17 categories in training data.
"""
import pandas as pd
import json
from pathlib import Path

# OFFICIAL 28 CATEGORIES from L. Ron Hubbard's Investigations
OFFICIAL_28_CATEGORIES = [
    # 14 OUTPOINTS
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
    "ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT",
    
    # 14 PLUSPOINTS
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

# Mapping from current inconsistent names to official names
CATEGORY_MAPPING = {
    # Current names → Official names
    "ADEQUATE_DATA_PLUS": "ADEQUATE_DATA_PLUS",  # Already correct
    "CORRECT_DATA_PLUS": "DATA_PROVEN_FACTUAL_PLUS",  # Fix
    "OMITTED_DATA_OUT": "OMITTED_DATA_OUT",  # Already correct
    "TIME_NOTED_PLUS": "TIME_NOTED_PLUS",  # Already correct
    "DATA_PROVEN_FACTUAL_PLUS": "DATA_PROVEN_FACTUAL_PLUS",  # Already correct
    "CORRECT_SOURCE_PLUS": "CORRECT_SOURCE_PLUS",  # Already correct
    "CONTRARY_FACTS_OUT": "CONTRARY_FACTS_OUT",  # Already correct
    "FALSE_DATA_OUT": "FALSEHOOD_OUT",  # Fix
    "ALTERED_IMPORTANCE_OUT": "ALTERED_IMPORTANCE_OUT",  # Already correct
    "CORRECT_SEQUENCE_PLUS": "EVENTS_IN_CORRECT_SEQUENCE_PLUS",  # Fix
    "NEUTRAL": "NEUTRAL",  # Keep neutral
    "CORRECT_IMPORTANCE_PLUS": "CORRECT_RELATIVE_IMPORTANCE_PLUS",  # Fix
    "ASSUMED_IDENTITIES_OUT": "ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT",  # Fix
    "ALTERED_SEQUENCE_OUT": "ALTERED_SEQUENCE_OUT",  # Already correct
    "WRONG_TARGET_OUT": "WRONG_TARGET_OUT",  # Already correct
    "ADDED_DATA_OUT": "ADDED_INAPPLICABLE_DATA_OUT",  # Fix (assuming this was meant to be inapplicable)
    "ADDED_INAPPLICABLES_OUT": "ADDED_INAPPLICABLE_DATA_OUT",  # Fix
}

def analyze_current_categories():
    """Analyze what categories we currently have"""
    print("🔍 ANALYZING CURRENT CATEGORY ISSUES")
    print("="*50)
    
    # Load current data
    df = pd.read_csv("data/enhanced_training_data.csv")
    labeled_df = df[df['label'].notna() & (df['label'] != '')]
    
    # Get all current categories
    current_categories = set()
    for _, row in labeled_df.iterrows():
        labels = [label.strip() for label in row['label'].split(',')]
        current_categories.update(labels)
    
    print(f"📊 Current categories: {len(current_categories)}")
    print(f"📊 Required categories: {len(OFFICIAL_28_CATEGORIES)} + NEUTRAL = 29")
    
    print(f"\n📋 CURRENT CATEGORIES:")
    for i, cat in enumerate(sorted(current_categories), 1):
        mapped = CATEGORY_MAPPING.get(cat, "❌ UNMAPPED")
        print(f"  {i:2d}. {cat:<30} → {mapped}")
    
    print(f"\n❌ MISSING CATEGORIES:")
    missing = set(OFFICIAL_28_CATEGORIES) - set(CATEGORY_MAPPING.values())
    for i, cat in enumerate(sorted(missing), 1):
        print(f"  {i:2d}. {cat}")
    
    return current_categories, missing

def create_complete_28_category_dataset():
    """Create a complete dataset with all 28 categories"""
    print(f"\n🔧 CREATING COMPLETE 28-CATEGORY DATASET")
    print("="*50)
    
    # Load current data
    df = pd.read_csv("data/enhanced_training_data.csv")
    
    # Map existing categories to official names
    mapped_rows = []
    for _, row in df.iterrows():
        if pd.isna(row['label']) or row['label'] == '':
            mapped_rows.append(row.to_dict())
            continue
            
        labels = [label.strip() for label in row['label'].split(',')]
        mapped_labels = []
        
        for label in labels:
            if label in CATEGORY_MAPPING:
                mapped_labels.append(CATEGORY_MAPPING[label])
            else:
                print(f"⚠️ Unmapped category: {label}")
                mapped_labels.append(label)  # Keep as-is for now
        
        new_row = row.to_dict()
        new_row['label'] = ', '.join(mapped_labels)
        mapped_rows.append(new_row)
    
    # Create synthetic examples for missing categories
    missing_categories = set(OFFICIAL_28_CATEGORIES) - set(CATEGORY_MAPPING.values())
    
    synthetic_examples = {
        # Missing outpoints
        "DROPPED_TIME_OUT": [
            "The report mentions the incident but doesn't specify when it occurred.",
            "The meeting was held but no date or time was recorded in the minutes.",
            "The contract was signed without any timestamp or date notation."
        ],
        
        "WRONG_SOURCE_OUT": [
            "The financial report cited Wikipedia as the source for market data.",
            "The medical study referenced a blog post instead of peer-reviewed research.",
            "The legal brief quoted social media comments as authoritative law."
        ],
        
        "ADDED_TIME_OUT": [
            "The 5-minute task somehow took 3 hours to complete.",
            "The instant download required 2 days of processing time.",
            "The brief phone call lasted longer than a full work day."
        ],
        
        "INCORRECTLY_INCLUDED_DATUM_OUT": [
            "The car manual included instructions for operating a boat engine.",
            "The cooking recipe contained steps for assembling furniture.",
            "The software documentation included medical treatment procedures."
        ],
        
        "ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT": [
            "The report treated cats and cars as similar because both start with 'ca'.",
            "The analysis compared apples and mathematics because both can be counted.",
            "The study grouped swimming and flying as identical activities."
        ],
        
        "ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT": [
            "The report treated identical twins as completely different people.",
            "The analysis separated 'automobile' and 'car' as different categories.",
            "The study classified 'happy' and 'joyful' as opposite emotions."
        ],
        
        # Missing pluspoints
        "RELATED_FACTS_KNOWN_PLUS": [
            "The earthquake report included relevant geological data, population density, and building codes.",
            "The market analysis referenced current economic indicators, historical trends, and regulatory changes.",
            "The medical diagnosis considered symptoms, patient history, and recent test results."
        ],
        
        "EVENTS_IN_CORRECT_SEQUENCE_PLUS": [
            "First the alarm sounded, then emergency responders arrived, followed by evacuation procedures.",
            "The company filed paperwork, received approval, then began construction as required.",
            "Students enrolled, attended classes, took exams, and then received grades in proper order."
        ],
        
        "EXPECTED_TIME_PERIOD_PLUS": [
            "The pizza delivery arrived within the promised 30-minute window.",
            "The construction project was completed on schedule after 6 months.",
            "The software update installed in the expected 10-minute timeframe."
        ],
        
        "APPLICABLE_DATA_PLUS": [
            "The weather report included only relevant meteorological information for the local area.",
            "The financial statement contained pertinent data about company performance and assets.",
            "The medical chart recorded only relevant symptoms and treatment information."
        ],
        
        "CORRECT_TARGET_PLUS": [
            "The investigation correctly focused on the actual source of the problem.",
            "The marketing campaign targeted the appropriate demographic for the product.",
            "The repair efforts addressed the root cause rather than symptoms."
        ],
        
        "DATA_IN_SAME_CLASSIFICATION_PLUS": [
            "All financial data was properly categorized as revenue, expenses, or assets.",
            "The inventory system correctly grouped similar items in the same categories.",
            "The research data was appropriately classified by methodology and subject."
        ],
        
        "IDENTITIES_ARE_IDENTICAL_PLUS": [
            "The system correctly recognized that 'John Smith' and 'J. Smith' refer to the same person.",
            "The database properly identified duplicate entries as the same record.",
            "The analysis correctly treated 'USA' and 'United States' as identical entities."
        ],
        
        "SIMILARITIES_ARE_SIMILAR_PLUS": [
            "The comparison correctly identified that both cars and trucks are motor vehicles.",
            "The analysis properly grouped cats and dogs as similar domestic animals.",
            "The study appropriately classified running and jogging as similar activities."
        ],
        
        "DIFFERENCES_ARE_DIFFERENT_PLUS": [
            "The report correctly distinguished between revenue and profit as different concepts.",
            "The analysis properly separated cats from dogs as different species.",
            "The study appropriately differentiated between correlation and causation."
        ]
    }
    
    # Add synthetic examples for missing categories
    for category, examples in synthetic_examples.items():
        if category in missing_categories:
            print(f"📈 Adding {len(examples)} examples for {category}")
            for i, example in enumerate(examples, 1):
                new_row = {
                    'text': example,
                    'label': category,
                    'notes': f'Synthetic example {i} for complete 28-category coverage'
                }
                mapped_rows.append(new_row)
    
    # Create complete dataframe
    complete_df = pd.DataFrame(mapped_rows)
    
    # Save complete dataset
    output_file = "data/complete_28_category_dataset.csv"
    complete_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Complete dataset saved to: {output_file}")
    print(f"📊 Total statements: {len(complete_df)}")
    
    return complete_df

def create_official_label_mapping():
    """Create the official 28-category label mapping"""
    print(f"\n📋 CREATING OFFICIAL 28-CATEGORY MAPPING")
    print("="*40)
    
    # Create mapping with NEUTRAL
    all_categories = OFFICIAL_28_CATEGORIES + ["NEUTRAL"]
    label_to_id = {label: i for i, label in enumerate(sorted(all_categories))}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    print(f"✅ Created mapping for {len(label_to_id)} categories")
    
    # Save mapping
    with open('official_28_category_mapping.json', 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'total_categories': len(label_to_id),
            'outpoints': [cat for cat in OFFICIAL_28_CATEGORIES if cat.endswith('_OUT')],
            'pluspoints': [cat for cat in OFFICIAL_28_CATEGORIES if cat.endswith('_PLUS')]
        }, f, indent=2)
    
    print(f"💾 Official mapping saved to: official_28_category_mapping.json")
    
    return label_to_id, id_to_label

def validate_28_categories(complete_df):
    """Validate that we now have all 28 categories"""
    print(f"\n✅ VALIDATING 28-CATEGORY COVERAGE")
    print("="*40)
    
    # Get all categories in dataset
    labeled_df = complete_df[complete_df['label'].notna() & (complete_df['label'] != '')]
    
    found_categories = set()
    for _, row in labeled_df.iterrows():
        labels = [label.strip() for label in row['label'].split(',')]
        found_categories.update(labels)
    
    # Check coverage
    missing = set(OFFICIAL_28_CATEGORIES) - found_categories
    extra = found_categories - set(OFFICIAL_28_CATEGORIES + ["NEUTRAL"])
    
    print(f"📊 Found categories: {len(found_categories)}")
    print(f"📊 Required: 28 + NEUTRAL = 29")
    
    if missing:
        print(f"❌ Still missing: {missing}")
    else:
        print(f"✅ All 28 official categories present!")
    
    if extra:
        print(f"⚠️ Extra categories: {extra}")
    
    # Count examples per category
    from collections import Counter
    category_counts = Counter()
    for _, row in labeled_df.iterrows():
        labels = [label.strip() for label in row['label'].split(',')]
        for label in labels:
            category_counts[label] += 1
    
    print(f"\n📈 CATEGORY DISTRIBUTION:")
    print(f"{'Category':<40} {'Count'}")
    print("-" * 50)
    
    for category in sorted(OFFICIAL_28_CATEGORIES + ["NEUTRAL"]):
        count = category_counts.get(category, 0)
        status = "✅" if count > 0 else "❌"
        print(f"{category:<40} {count:>5} {status}")
    
    return len(missing) == 0

def main():
    try:
        print("🚨 FIXING CRITICAL 28-CATEGORY ISSUE")
        print("="*50)
        
        # Analyze current issues
        current_categories, missing = analyze_current_categories()
        
        # Create complete dataset
        complete_df = create_complete_28_category_dataset()
        
        # Create official mapping
        label_to_id, id_to_label = create_official_label_mapping()
        
        # Validate coverage
        success = validate_28_categories(complete_df)
        
        if success:
            print(f"\n🎉 SUCCESS: All 28 categories now covered!")
            print(f"📊 Dataset ready for proper 28-category training")
            print(f"🚀 Next: Train DeBERTa model with complete 28 categories")
        else:
            print(f"\n❌ FAILURE: Still missing categories")
            print(f"🔧 Manual intervention required")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error fixing categories: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
