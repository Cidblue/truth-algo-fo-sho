# TruthAlgorithm Model Documentation

## 📋 **MODEL OVERVIEW**

The TruthAlgorithm uses a **3-layer architecture** implementing L. Ron Hubbard's Investigations methodology:

1. **Regex Layer** - Pattern-based detection
2. **DeBERTa/BERT Layer** - Transformer-based classification
3. **LLM Layer** - Large Language Model with RAG system

## 🤖 **CURRENT MODELS IN USE**

### **1. DeBERTa/BERT Classifier Models**

#### **Baseline Model (Round 1)**

- **Location**: `models/bert-test/`
- **Base Model**: `bert-base-uncased`
- **Categories**: 17 (INCOMPLETE - missing 11 official categories)
- **Performance**: 51.7% accuracy, 0.172 average confidence
- **Threshold**: 0.05 (optimized from 0.3)
- **Status**: ⚠️ **DEPRECATED** - Use only for comparison
- **Issue**: Missing 11 L. Ron Hubbard categories

#### **Round 2 Model (COMPLETED)**

- **Location**: `models/round2-simple/` ✅ **TRAINED**
- **Base Model**: `bert-base-uncased`
- **Categories**: 29 (28 official + NEUTRAL)
- **Training Data**: `data/complete_28_category_dataset.csv` (186 examples)
- **Actual Performance**: 3.4% accuracy, training duration 12:06 minutes
- **Status**: ✅ **COMPLETED** - Needs optimization
- **Training Results**: `round2_simple_results.json`

#### **Round 2 Primary Model**

- **Location**: `models/round2-28categories/` (alternative approach)
- **Base Model**: `bert-base-uncased`
- **Categories**: 29 (28 official + NEUTRAL)
- **Training Strategy**: Complete feature set
- **Status**: 🔄 **AVAILABLE FOR TRAINING**

### **2. Local LLM (Ollama)**

#### **Truth-Evaluator Model**

- **Model Name**: `truth-evaluator`
- **Platform**: Ollama (local deployment)
- **Location**: `E:\Ollama\models\`
- **Startup Script**: `E:\startlocalLLMServer.bat`
- **Performance**: 24 seconds to 8+ minutes per statement
- **Status**: ✅ **ACTIVE**
- **Usage**: LLM layer fallback and RAG system

### **3. Model Selection History**

#### **DeBERTa v3 Issues**

- **Attempted**: `microsoft/deberta-v3-base`
- **Issue**: Tokenizer conversion errors
- **Resolution**: Switched to `bert-base-uncased` for reliability

#### **BERT Selection Rationale**

- **Chosen**: `bert-base-uncased`
- **Reasons**:
  - Reliable tokenizer
  - Good performance on classification tasks
  - Wide compatibility
  - Proven track record

## 📊 **MODEL PERFORMANCE TRACKING**

### **Baseline Performance (Round 1)**

```
Model: models/bert-test/
Accuracy: 51.7% (30/58 correct)
Average Confidence: 0.172
Coverage: 100% (after threshold optimization)
Categories: 17/28 (INCOMPLETE)
Threshold: 0.05
Training Data: 100 statements
```

### **Round 2 Performance (ACTUAL)**

```
Model: models/round2-simple/
Actual Accuracy: 3.4% (NEEDS IMPROVEMENT)
Training Duration: 12:06 minutes
Coverage: 100% (assumed)
Categories: 29/29 (COMPLETE - 28 + NEUTRAL)
Threshold: 0.05
Training Data: 186 statements (115 train, 29 validation)
Training Examples: 115
Validation Examples: 29
```

### **Target Performance (Future Rounds)**

```
Target Accuracy: >60% (vs current 3.4%)
Target Confidence: >0.3
Coverage: 100%
Categories: 29/29 (COMPLETE)
Threshold: 0.05
Optimization Needed: YES - significant improvement required
```

## 🗂️ **MODEL FILES STRUCTURE**

### **Complete Model Directory Layout**

```
models/
├── bert-test/                    # Round 1 (DEPRECATED)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_map.json           # 17 categories only
│
├── round2-28categories/         # Round 2 (PRIMARY)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_map.json           # 29 categories (28 + NEUTRAL)
│
└── round2-simple/               # Round 2 (BACKUP)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── label_map.json           # 29 categories (28 + NEUTRAL)
```

### **Label Mapping Files**

#### **Official 28-Category Mapping**

- **File**: `official_28_category_mapping.json`
- **Categories**: 29 total (28 official + NEUTRAL)
- **Format**:

```json
{
  "label_to_id": {"CATEGORY_NAME": id},
  "id_to_label": {"id": "CATEGORY_NAME"},
  "total_categories": 29,
  "outpoints": [...],
  "pluspoints": [...]
}
```

## 🔧 **MODEL USAGE INSTRUCTIONS**

### **Loading Models in Code**

#### **Current Production Model**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load Round 2 model (when available)
model_dir = "models/round2-28categories"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Load label mapping
with open(f"{model_dir}/label_map.json", "r") as f:
    label_mapping = json.load(f)
```

#### **Fallback to Baseline**

```python
# If Round 2 not available, use baseline (with warnings)
model_dir = "models/bert-test"
# Note: Only has 17 categories - missing 11 official categories
```

### **Integration with TruthAlgorithm**

#### **Current Configuration**

```python
# In truth_algorithm.py
classifier = ClassificationPipeline(
    deberta_model_dir="models/round2-28categories",  # Will update after training
    deberta_threshold=0.05,
    use_llm_fallback=True,
    llm_model_name="truth-evaluator"
)
```

## 📈 **MODEL TRAINING HISTORY**

### **Round 1 Training**

- **Date**: Previous session
- **Script**: `models/train_deberta.py`
- **Result**: Baseline model with 17 categories
- **Issues**: Missing 11 official L. Ron Hubbard categories

### **Round 2 Training (COMPLETED)**

- **Date**: Current session (2025-08-03)
- **Scripts**:
  - `simple_round2_training.py` ✅ **EXECUTED**
  - `train_round2_28categories.py` (alternative available)
- **Results**:
  - ✅ Complete 28-category coverage achieved
  - ⚠️ Low accuracy (3.4%) - needs optimization
  - ✅ Model saved to `models/round2-simple/`
  - ✅ Training duration: 12:06 minutes
  - ✅ Training examples: 115, Validation: 29

## ⚠️ **CRITICAL MODEL NOTES**

### **Always Use 28 Categories**

- **NEVER** train with incomplete category sets
- **ALWAYS** use `data/complete_28_category_dataset.csv`
- **ALWAYS** use `official_28_category_mapping.json`

### **Model Selection Priority**

1. **Current**: `models/round2-simple/` ✅ **TRAINED** (28 categories, needs optimization)
2. **Alternative**: `models/round2-28categories/` (available for training)
3. **Deprecated**: `models/bert-test/` (only for comparison - 17 categories)

### **Threshold Settings**

- **DeBERTa/BERT Threshold**: 0.05 (optimized for 100% coverage)
- **Never use**: 0.3 or higher (causes 0% predictions)

## 🚀 **FUTURE MODEL DEVELOPMENT**

### **Round 3+ Planning**

- **Target**: 98%+ accuracy across all 28 categories
- **Strategies**:
  - Advanced data augmentation
  - Ensemble methods
  - Fine-tuned hyperparameters
  - Larger training datasets

### **Model Versioning**

- **Convention**: `models/round{N}-{description}/`
- **Always preserve**: Previous models for comparison
- **Always document**: Performance metrics and changes

---

**Last Updated**: Current session - Round 2 training preparation complete
