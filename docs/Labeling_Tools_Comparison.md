# Labeling Tools Comparison

You have **three excellent options** for labeling your DeBERTa training data. Each has different strengths:

## 🖥️ **Option 1: GUI Statement Categorizer (Recommended for Quality)**
**File**: `utils/statement_categorizer.py`

### ✅ **Advantages**:
- **Complete coverage**: All 14 outpoints + 14 pluspoints
- **Rich descriptions**: Built-in help buttons with full descriptions from Investigations.txt
- **Visual interface**: Easy to see all options at once
- **Multiple selections**: Can assign multiple categories to one statement
- **Professional workflow**: Save/load progress, proper CSV export
- **Built-in validation**: Ensures exactly 14 outpoints and 14 pluspoints

### ⚠️ **Requirements**:
- Requires GUI (tkinter)
- Slightly slower for bulk labeling

### 🚀 **Usage**:
```bash
# Load your cleaned data into the GUI
python utils/statement_categorizer.py docs/Statementstoclass.txt data/labeled_statements.csv

# Or start GUI and load data manually
python utils/statement_categorizer.py
```

## ⌨️ **Option 2: Manual Labeling Helper (Fast & Complete)**
**File**: `utils/manual_labeling_helper.py` (Updated)

### ✅ **Advantages**:
- **All 28 categories**: Now includes complete set of outpoints/pluspoints
- **Keyboard shortcuts**: Numbers 1-28 for quick selection
- **Command line**: Works in any terminal
- **Progress tracking**: Shows completion status
- **Resume capability**: Can start from any index

### ⚠️ **Limitations**:
- One category per statement (no multiple selections)
- Need to remember category numbers

### 🚀 **Usage**:
```bash
# Quick labeling with shortcuts
python utils/manual_labeling_helper.py

# Check progress
python utils/manual_labeling_helper.py --progress_only

# Resume from specific statement
python utils/manual_labeling_helper.py --start_idx 50
```

## 🤖 **Option 3: AI-Assisted Labeling (Fastest)**
**File**: `utils/ai_label_assistant.py` (Updated)

### ✅ **Advantages**:
- **AI suggestions**: Uses your existing LLM to suggest labels
- **Batch mode**: Can auto-label entire dataset
- **Interactive review**: Review and correct AI suggestions
- **All 28 categories**: Complete coverage with paired relationships

### ⚠️ **Requirements**:
- Requires Ollama running with truth-evaluator model
- AI suggestions need human review for quality

### 🚀 **Usage**:
```bash
# Auto-label everything (requires review)
python utils/ai_label_assistant.py --mode batch

# Interactive review of AI suggestions
python utils/ai_label_assistant.py --mode interactive
```

## 📊 **Category Pairs (Opposite Ends of Scales)**

You're absolutely right about the paired nature! Here are the main pairs:

### **Information Completeness Scale**:
- `OMITTED_DATA_OUT` ↔ `ADEQUATE_DATA_PLUS`
- `ADDED_INAPPLICABLE_DATA_OUT` ↔ `APPLICABLE_DATA_PLUS`

### **Time Accuracy Scale**:
- `DROPPED_TIME_OUT` ↔ `TIME_NOTED_PLUS`
- `ADDED_TIME_OUT` ↔ `EXPECTED_TIME_PERIOD_PLUS`

### **Sequence Correctness Scale**:
- `ALTERED_SEQUENCE_OUT` ↔ `EVENTS_IN_CORRECT_SEQUENCE_PLUS`

### **Truth/Factuality Scale**:
- `FALSEHOOD_OUT` ↔ `DATA_PROVEN_FACTUAL_PLUS`

### **Source Reliability Scale**:
- `WRONG_SOURCE_OUT` ↔ `CORRECT_SOURCE_PLUS`

### **Target Accuracy Scale**:
- `WRONG_TARGET_OUT` ↔ `CORRECT_TARGET_PLUS`

### **Importance Assessment Scale**:
- `ALTERED_IMPORTANCE_OUT` ↔ `CORRECT_RELATIVE_IMPORTANCE_PLUS`

### **Classification Accuracy Scale**:
- `INCORRECTLY_INCLUDED_DATUM_OUT` ↔ `DATA_IN_SAME_CLASSIFICATION_PLUS`

### **Identity Recognition Scale**:
- `ASSUMED_IDENTITIES_NOT_IDENTICAL_OUT` ↔ `IDENTITIES_ARE_IDENTICAL_PLUS`

### **Similarity Recognition Scale**:
- `ASSUMED_SIMILARITIES_NOT_SIMILAR_OUT` ↔ `SIMILARITIES_ARE_SIMILAR_PLUS`

### **Difference Recognition Scale**:
- `ASSUMED_DIFFERENCES_NOT_DIFFERENT_OUT` ↔ `DIFFERENCES_ARE_DIFFERENT_PLUS`

## 🎯 **Recommendation**

### **For Best Quality**: Use the **GUI Statement Categorizer**
- Most comprehensive and user-friendly
- Built-in descriptions help ensure consistent labeling
- Can handle complex cases with multiple categories

### **For Speed**: Use the **Manual Labeling Helper**
- Updated with all 28 categories
- Keyboard shortcuts make it very fast
- Good for straightforward cases

### **For Initial Pass**: Use **AI-Assisted Labeling**
- Get AI suggestions for all statements
- Then review and correct with one of the other tools

## 🔄 **Hybrid Approach (Recommended)**

1. **Start with AI batch labeling** to get initial suggestions
2. **Review with GUI tool** for complex/uncertain cases
3. **Use manual helper** for quick corrections and final cleanup

This gives you the best of all worlds: speed, accuracy, and comprehensive coverage!

## 📈 **Data Quality Tips**

- **Aim for balance**: Try to get examples of all 28 categories
- **Focus on clear cases**: Start with obvious examples
- **Use NEUTRAL sparingly**: Only for truly borderline cases
- **Consider the scale**: A statement might be slightly wrong (outpoint) or clearly excellent (pluspoint)
- **Document difficult decisions**: Use the notes field for borderline cases
