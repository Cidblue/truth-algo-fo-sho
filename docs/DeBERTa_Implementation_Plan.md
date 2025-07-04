# DeBERTa Implementation Plan - Complete Guide

## 🎯 Current Status

- ✅ **Data Cleaned**: 100 quality statements extracted from BBC scrapes
- ✅ **Infrastructure Ready**: Training scripts, classifier class, and tools prepared
- ✅ **Sample Labels**: 9 statements manually labeled as examples
- ⏳ **Next Step**: Complete labeling and train model

## 📁 Files Created/Updated

### Data Preparation

- `utils/clean_scraped_data.py` - Cleans and filters scraped data
- `data/statements_to_label.csv` - 100 statements ready for labeling
- `docs/DeBERTa_Labeling_Guide.md` - Comprehensive labeling instructions

### Training Infrastructure

- `models/train_deberta.py` - DeBERTa fine-tuning script
- `models/deberta_classifier.py` - Updated classifier class (ready for trained model)
- `utils/prepare_training_data.py` - Data preparation utilities

### Labeling Tools

- `utils/manual_labeling_helper.py` - Quick manual labeling with shortcuts
- `utils/ai_label_assistant.py` - AI-assisted labeling (requires LLM)

## 🚀 Step-by-Step Implementation

### Step 1: Complete Data Labeling (CURRENT STEP)

**Goal**: Label all 100 statements (currently 9/100 done)

**Option A - GUI Statement Categorizer (RECOMMENDED - Best Quality)**:

```bash
# Professional GUI with all 28 categories and descriptions
python utils/statement_categorizer.py docs/Statementstoclass.txt data/labeled_statements.csv

# Or start GUI and load data manually
python utils/statement_categorizer.py
```

**Option B - Manual Labeling Helper (UPDATED - Fast & Complete)**:

```bash
# Now includes all 28 categories (1-28 shortcuts)
python utils/manual_labeling_helper.py

# Check progress anytime
python utils/manual_labeling_helper.py --progress_only
```

**Option C - AI-Assisted Labeling (Fastest but requires review)**:

```bash
# Requires Ollama running with truth-evaluator model
python utils/ai_label_assistant.py --mode batch
python utils/ai_label_assistant.py --mode interactive  # Review AI suggestions
```

**HYBRID APPROACH (Recommended)**:

1. Start with AI batch labeling for initial suggestions
2. Review complex cases with GUI tool
3. Use manual helper for quick final cleanup

**Label Categories**:

- **Outpoints**: OMITTED_DATA_OUT, FALSEHOOD_OUT, WRONG_SOURCE_OUT, etc.
- **Pluspoints**: DATA_PROVEN_FACTUAL_PLUS, CORRECT_SOURCE_PLUS, TIME_NOTED_PLUS, etc.
- **Neutral**: NEUTRAL (for statements without clear issues)

### Step 2: Install Training Dependencies

```bash
pip install datasets transformers torch scikit-learn
```

### Step 3: Train DeBERTa Model

```bash
# Basic training (adjust parameters as needed)
python models/train_deberta.py --data_path data/statements_to_label.csv

# Advanced training with custom parameters
python models/train_deberta.py \
    --data_path data/statements_to_label.csv \
    --model_name microsoft/deberta-v3-base \
    --output_dir models/deberta-lora \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --epochs 3
```

### Step 4: Test Trained Model

```bash
# Test the classifier
python utils/test_deberta.py

# Test integration with main pipeline
python truth_algorithm.py sample.json -v
```

### Step 5: Evaluate and Iterate

- Check model performance on validation set
- Add more training data if needed
- Adjust hyperparameters for better results

## 📊 Current Data Status

**Labeled Examples (9/100)**:

- OMITTED_DATA_OUT: 1 (missing context/information)
- FALSEHOOD_OUT: 0
- WRONG_SOURCE_OUT: 0
- ALTERED_IMPORTANCE_OUT: 1 (exaggerated claims)
- DATA_PROVEN_FACTUAL_PLUS: 2 (verified facts)
- CORRECT_SOURCE_PLUS: 1 (reliable sources)
- TIME_NOTED_PLUS: 2 (good time references)
- ADEQUATE_DATA_PLUS: 1 (sufficient detail)
- NEUTRAL: 1

**Recommended Target Distribution (for 100 statements)**:

- Outpoints: ~40 statements (various types)
- Pluspoints: ~40 statements (various types)
- Neutral: ~20 statements

## 🎯 Quality Guidelines

### Good Training Data Characteristics:

1. **Balanced**: Roughly equal outpoints, pluspoints, and neutral
2. **Diverse**: Multiple types of each category
3. **Clear**: Obvious examples of each category
4. **Consistent**: Same standards applied throughout

### Labeling Tips:

- **Start with obvious cases** (clear outpoints/pluspoints)
- **Use NEUTRAL for borderline cases** when unsure
- **Focus on logical structure**, not content agreement
- **Add notes for difficult decisions**

## 🔧 Troubleshooting

### Common Issues:

1. **Insufficient training data**: Need at least 50 labeled statements
2. **Imbalanced classes**: Ensure good distribution across categories
3. **Memory issues**: Reduce batch size or use smaller model
4. **Poor performance**: Add more diverse training examples

### Model Selection:

- **microsoft/deberta-v3-base**: Good balance of performance and size
- **microsoft/deberta-v3-small**: For memory-constrained environments
- **microsoft/deberta-v3-large**: For maximum performance (requires more resources)

## 📈 Expected Results

After training with 100 well-labeled statements:

- **Accuracy**: 70-85% on validation set
- **Integration**: Should work seamlessly with existing pipeline
- **Performance**: Faster than LLM for obvious cases
- **Fallback**: LLM still handles complex cases

## 🎉 Success Metrics

**Ready for Production When**:

- ✅ 80+ statements labeled with good distribution
- ✅ Model trains without errors
- ✅ Validation accuracy > 70%
- ✅ Integration tests pass
- ✅ Performance improvement over rules-only mode

## 📝 Next Immediate Actions

1. **Complete labeling**: Use `python utils/manual_labeling_helper.py`
2. **Train model**: Run training script once you have 50+ labels
3. **Test integration**: Verify it works with the main pipeline
4. **Document results**: Update project documentation with performance metrics

The foundation is solid - you just need to complete the labeling to get DeBERTa working!
