# LLM Optimization Plan for Perfect Outpoint/Pluspoint Detection

## Current Status
- ✅ dolphin3 model working with ~6.2 GB memory usage
- ✅ 2.6 GB available memory for optimization
- ✅ Clean prompts without massive RAG overhead
- ✅ No timeout constraints

## Optimization Strategy

### 1. PROMPT ENGINEERING OPTIMIZATION
**Goal**: Minimize prompt size while maximizing accuracy

#### Current Issues:
- Prompts may be too verbose
- Inconsistent formatting
- Unclear instructions

#### Solutions:
- **Streamlined prompts**: Remove unnecessary text
- **Consistent format**: Standardize all prompt templates
- **Clear examples**: Include 2-3 perfect examples per outpoint/pluspoint
- **Focused instructions**: One clear task per prompt

### 2. TRAINING DATA CURATION
**Goal**: Create perfect training examples for each outpoint/pluspoint

#### High-Quality Training Set:
- **28 categories**: All 14 outpoints + 14 pluspoints
- **50-100 examples each**: ~1,400-2,800 total examples
- **Manual verification**: Every example checked for accuracy
- **Clear labels**: Use actual names (not LABEL_0, LABEL_1)

#### Training Data Structure:
```json
{
  "statement": "The company reported record profits but couldn't pay suppliers",
  "outpoints": ["contrary_facts", "falsehood"],
  "pluspoints": [],
  "confidence": 95,
  "reasoning": "Two contradictory facts presented together"
}
```

### 3. MODEL FINE-TUNING APPROACH
**Goal**: Train dolphin3 specifically for outpoint/pluspoint detection

#### Option A: LoRA Fine-tuning (Recommended)
- **Memory efficient**: ~1-2 GB additional
- **Fast training**: Hours instead of days
- **Preserves base model**: Can revert if needed
- **Specific task focus**: Outpoint/pluspoint detection only

#### Option B: Full Fine-tuning
- **Higher memory**: ~4-6 GB additional
- **Longer training**: Days
- **Better results**: More thorough adaptation
- **Risk**: May lose general capabilities

### 4. PROMPT OPTIMIZATION EXPERIMENTS

#### Experiment 1: Minimal Prompts
```
Task: Identify outpoints in this statement.
Statement: [TEXT]
Outpoints found: [LIST]
Confidence: [0-100]
```

#### Experiment 2: Example-Based Prompts
```
Examples:
- "Record profits but can't pay bills" → contrary_facts
- "Everyone knows..." → wrong_target

Statement: [TEXT]
Analysis: [OUTPOINTS]
```

#### Experiment 3: Step-by-Step Prompts
```
1. Read statement: [TEXT]
2. Check for contradictions: [YES/NO]
3. Check for generalizations: [YES/NO]
4. Final outpoints: [LIST]
```

### 5. EVALUATION FRAMEWORK
**Goal**: Measure improvement objectively

#### Metrics:
- **Accuracy**: Correct outpoint/pluspoint identification
- **Precision**: No false positives
- **Recall**: No missed outpoints/pluspoints
- **Consistency**: Same input → same output
- **Speed**: Response time per evaluation

#### Test Sets:
- **Validation set**: 200 manually verified examples
- **Edge cases**: Ambiguous or complex statements
- **Baseline comparison**: Current dolphin3 performance

### 6. ITERATIVE IMPROVEMENT PROCESS

#### Phase 1: Prompt Optimization (Week 1)
1. Test 5-10 different prompt formats
2. Measure accuracy on validation set
3. Select best performing prompts
4. Document optimal prompt templates

#### Phase 2: Training Data Creation (Week 2)
1. Generate 100 examples per category
2. Manual review and correction
3. Create balanced training set
4. Validate data quality

#### Phase 3: Model Fine-tuning (Week 3)
1. LoRA fine-tuning on curated data
2. Evaluate on test set
3. Compare to baseline
4. Iterate if needed

#### Phase 4: Production Testing (Week 4)
1. Deploy optimized model
2. Test on real pipeline data
3. Monitor performance
4. Collect feedback for next iteration

### 7. TECHNICAL IMPLEMENTATION

#### Tools Needed:
- **Hugging Face Transformers**: For model fine-tuning
- **LoRA/PEFT**: For efficient fine-tuning
- **Custom evaluation scripts**: For measuring performance
- **Data validation tools**: For training data quality

#### Memory Management:
- **Current usage**: 13.2 GB
- **LoRA training**: +1-2 GB (total ~15 GB)
- **Safety buffer**: Keep under 15.5 GB
- **Monitoring**: Track memory usage during training

### 8. SUCCESS CRITERIA

#### Target Performance:
- **Accuracy**: >95% on validation set
- **Response time**: <60 seconds per statement
- **Consistency**: >98% same results on repeated tests
- **Memory usage**: <15.5 GB total
- **Prompt size**: <2K tokens (vs current unknown size)

#### Deliverables:
1. Optimized prompt templates
2. High-quality training dataset
3. Fine-tuned model weights
4. Performance evaluation report
5. Production deployment guide

## Next Steps

1. **Immediate**: Test current prompt sizes and response accuracy
2. **This week**: Experiment with prompt optimization
3. **Next week**: Begin training data curation
4. **Following weeks**: Model fine-tuning and evaluation

This approach will give us a highly optimized, memory-efficient LLM that excels at outpoint/pluspoint detection without the overhead of DeBERTa or RAG systems.
