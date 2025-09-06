# NEW AGENT QUICK START - LLM OPTIMIZATION PHASE

## 🚀 **IMMEDIATE CONTEXT - SEPTEMBER 5, 2025**

**PREVIOUS PHASE COMPLETED**: RAG optimization and memory management successfully finished.
**CURRENT PHASE**: LLM prompt optimization and model fine-tuning for perfect outpoint detection.

## ✅ **WHAT'S ALREADY DONE (DO NOT REPEAT)**

- ✅ RAG disabled (eliminated 1.1M token prompts)
- ✅ Switched to dolphin3 model (no built-in RAG)
- ✅ Memory optimized (11 GB used, 4.6 GB available)
- ✅ DeBERTa disabled (memory constraints)
- ✅ File structure cleaned up
- ✅ Complete documentation created

## 🎯 **YOUR IMMEDIATE GOALS**

**Phase 1: Prompt Optimization (Week 1)**

```bash
# Step 1: Implement minimal prompts (identified 73% reduction potential)
# Current: 109 tokens → Target: 29 tokens
python implement_minimal_prompts.py

# Step 2: Test accuracy with optimized prompts
python test_prompt_optimization.py

# Step 3: Measure performance improvements
python measure_prompt_performance.py
```

**Phase 2: Training Data Creation (Week 2)**

```bash
# Step 1: Generate high-quality training examples
python create_training_data.py --examples 100 --per-category

# Step 2: Manual verification and correction
python verify_training_data.py

# Step 3: Create balanced training set
python prepare_final_dataset.py
```

**Phase 3: LoRA Fine-tuning (Week 3)**

```bash
# Step 1: Setup LoRA fine-tuning (memory efficient)
python setup_lora_training.py

# Step 2: Fine-tune dolphin3 for outpoint detection
python train_lora_model.py

# Step 3: Evaluate fine-tuned model
python evaluate_finetuned_model.py
```

## 🎯 **SUCCESS CRITERIA**

- **Prompt Size**: Reduce from 109 tokens to 29 tokens (73% reduction)
- **Accuracy**: Achieve >95% on validation set
- **Response Time**: Keep <60 seconds per evaluation
- **Memory Usage**: Stay <15 GB total
- **Consistency**: >98% same results on repeated tests

## 📁 **CRITICAL FILES TO READ FIRST**

1. **`SESSION_STATUS_2025_09_05.md`** - **START HERE** - Complete session summary
2. **`llm_optimization_plan.md`** - **YOUR ROADMAP** - Detailed strategy
3. **`simple_prompt_test.py`** - Prompt analysis (73% reduction identified)
4. **`docs/FILE_REGISTRY.md`** - Complete file documentation (updated today)

## 🔧 **CURRENT SYSTEM STATUS**

### **What's Working Now**

- **Ollama Server**: Running on localhost:11434
- **Model**: dolphin3 (Dolphin 3.0 Llama 3.1 8B) loaded and ready
- **Memory**: 11 GB used, 4.6 GB available (stable)
- **Pipeline**: LLM-only mode (DeBERTa disabled for memory)

### **Key Tools Available**

- `quick_memory_check.py` - Monitor memory usage
- `test_dolphin3.py` - Test current model functionality
- `pipeline_readiness_test.py` - Verify system status
- `simple_prompt_test.py` - Analyze prompt optimization potential

## 🎯 **OPTIMIZATION STRATEGY**

### **Prompt Optimization Approach**

- **Current**: 109 tokens per prompt (verbose instructions)
- **Target**: 29 tokens per prompt (minimal format)
- **Method**: "Contrary facts in: [statement] YES/NO + confidence 0-100:"

### **Training Data Strategy**

- **Quality over quantity**: 50-100 perfect examples per category
- **Manual verification**: Every example checked against L. Ron Hubbard definitions
- **Clear labeling**: Use actual outpoint/pluspoint names (not LABEL_X)

### **Fine-tuning Strategy**

- **LoRA approach**: Memory-efficient (1-2 GB additional)
- **Preserve base model**: Keep general capabilities
- **Task-specific**: Focus only on outpoint/pluspoint detection

## 🎯 **IMMEDIATE NEXT STEPS**

1. **Read the session summary**: `SESSION_STATUS_2025_09_05.md`
2. **Review the optimization plan**: `llm_optimization_plan.md`
3. **Test current system**: Run `python test_dolphin3.py`
4. **Start prompt optimization**: Implement minimal prompt formats

## ⚠️ **TROUBLESHOOTING**

### **If Ollama Server Not Running**

```bash
# Start Ollama server
cmd /c "set OLLAMA_MODELS=E:\Ollama\models && set OLLAMA_HOST=0.0.0.0:11434 && E:\Ollama\ollama.exe serve"
```

### **If Memory Issues Occur**

- **Check memory**: Run `python quick_memory_check.py`
- **Free memory**: Close unnecessary applications
- **Monitor usage**: Keep total <15 GB

### **If Model Not Responding**

- **Test connection**: Run `python test_dolphin3.py`
- **Check server logs**: View `ollama_server.log`
- **Restart if needed**: Kill ollama.exe and restart

### **If Scripts Need Creation**

- **Follow the plan**: Use `llm_optimization_plan.md` as guide
- **Start simple**: Begin with prompt optimization
- **Build incrementally**: Test each change

## 🎉 **CURRENT PROJECT STATUS**

- ✅ **RAG Optimization**: COMPLETED - 1.1M token prompts eliminated
- ✅ **Memory Management**: COMPLETED - Stable 11 GB usage
- ✅ **File Cleanup**: COMPLETED - 15 obsolete files removed
- ✅ **Documentation**: COMPLETED - All changes documented
- ✅ **System Ready**: dolphin3 loaded, 4.6 GB available for optimization

## � **READY FOR OPTIMIZATION PHASE**

The previous agent completed the foundation work. Your mission is to optimize the LLM for perfect outpoint detection through prompt optimization and fine-tuning.

**Start with reading `SESSION_STATUS_2025_09_05.md` for complete context!**

---

**Created**: Current session  
**Purpose**: Quick start guide for new agent
**Status**: Ready for immediate execution
