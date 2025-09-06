# NEXT AGENT INSTRUCTIONS - LLM OPTIMIZATION PHASE

## 🎯 **MISSION: OPTIMIZE LLM FOR PERFECT OUTPOINT DETECTION**

You are taking over the TruthAlgorithm project in the **LLM Optimization Phase**. The previous agent completed RAG optimization and memory management. Your mission is to achieve **>95% accuracy** in outpoint/pluspoint detection through prompt optimization and model fine-tuning.

## 📊 **CURRENT STATUS (September 5, 2025)**

### **✅ FOUNDATION COMPLETED (DO NOT REPEAT)**
- **RAG Disabled**: Eliminated 1.1M token prompts causing timeouts
- **Model Switch**: Changed from truth-evaluator to dolphin3 (no built-in RAG)
- **Memory Optimized**: Stable 11 GB usage, 4.6 GB available
- **File Cleanup**: 15 obsolete files removed, project structure cleaned
- **Documentation**: Complete session records and optimization plan created

### **🎯 YOUR OPTIMIZATION TARGETS**
- **Prompt Size**: Reduce from 109 tokens to 29 tokens (73% reduction)
- **Accuracy**: Achieve >95% on validation set
- **Response Time**: Keep <60 seconds per evaluation
- **Memory Usage**: Stay <15 GB total
- **Consistency**: >98% same results on repeated tests

## 🚀 **IMMEDIATE EXECUTION PLAN**

### **Phase 1: Prompt Optimization (Week 1)**

**Step 1: Implement Minimal Prompts**
```python
# Current format (109 tokens):
"""You are analyzing statements for outpoints according to L. Ron Hubbard's methodology.
OUTPOINT: contrary_facts
DEFINITION: Two or more facts that contradict each other
STATEMENT TO ANALYZE: [statement]
Please analyze if this statement contains the outpoint 'contrary_facts'.
Provide your response in this format:
RESULT: YES or NO
CONFIDENCE: 0-100
REASONING: Brief explanation"""

# Target format (29 tokens):
"""Contrary facts in: "[statement]"
YES/NO + confidence 0-100:"""
```

**Step 2: Test Accuracy**
- Compare optimized vs current prompts
- Measure accuracy on validation set
- Document performance improvements

**Step 3: Implement Best Prompts**
- Update `models/llm_evaluator.py` with optimized prompts
- Test full pipeline functionality
- Verify no accuracy loss

### **Phase 2: Training Data Creation (Week 2)**

**Step 1: Generate High-Quality Examples**
- Create 50-100 examples per outpoint/pluspoint
- Focus on clear, unambiguous cases
- Use actual L. Ron Hubbard definitions

**Step 2: Manual Verification**
- Review every example for accuracy
- Apply strict L. Ron Hubbard criteria
- Remove ambiguous or incorrect examples

**Step 3: Create Balanced Dataset**
- Ensure equal representation of all 28 categories
- Create train/validation/test splits
- Document data quality metrics

### **Phase 3: LoRA Fine-tuning (Week 3)**

**Step 1: Setup LoRA Training**
- Install PEFT library for LoRA
- Configure memory-efficient training
- Prepare training infrastructure

**Step 2: Fine-tune dolphin3**
- Train on curated dataset
- Monitor memory usage (<15 GB total)
- Preserve general model capabilities

**Step 3: Evaluate Results**
- Test on validation set
- Compare to baseline performance
- Document accuracy improvements

## 📁 **CRITICAL FILES TO READ FIRST**

1. **`SESSION_STATUS_2025_09_05.md`** - **START HERE** - Complete session summary
2. **`llm_optimization_plan.md`** - **YOUR DETAILED ROADMAP** - Complete strategy
3. **`simple_prompt_test.py`** - Prompt analysis results (73% reduction identified)
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

## ⚠️ **IMPORTANT: WHAT NOT TO DO**

### **❌ DO NOT REPEAT COMPLETED WORK**
- Do NOT try to re-enable RAG (already disabled)
- Do NOT switch back to truth-evaluator model
- Do NOT try to enable DeBERTa (memory constraints)
- Do NOT redo file cleanup (already completed)

### **❌ DO NOT IGNORE MEMORY CONSTRAINTS**
- Total system has 15.8 GB RAM
- Current usage: 11 GB (stable)
- Available: 4.6 GB for optimization
- Stay under 15 GB total to avoid OOM crashes

## 🎯 **SUCCESS CRITERIA**

### **Phase 1 Success (Prompt Optimization)**
- [ ] Prompts reduced from 109 to 29 tokens
- [ ] No accuracy loss with minimal prompts
- [ ] 3x faster response times achieved
- [ ] Memory usage remains stable

### **Phase 2 Success (Training Data)**
- [ ] 50-100 high-quality examples per category created
- [ ] Manual verification completed for all examples
- [ ] Balanced dataset prepared for training
- [ ] Data quality metrics documented

### **Phase 3 Success (Fine-tuning)**
- [ ] LoRA fine-tuning completed successfully
- [ ] >95% accuracy achieved on validation set
- [ ] Memory usage stays <15 GB during training
- [ ] Model performance documented and verified

## 🚀 **GETTING STARTED**

1. **Read the session summary**: `SESSION_STATUS_2025_09_05.md`
2. **Review the optimization plan**: `llm_optimization_plan.md`
3. **Test current system**: `python test_dolphin3.py`
4. **Check memory status**: `python quick_memory_check.py`
5. **Start prompt optimization**: Begin implementing minimal prompts

## 📞 **TROUBLESHOOTING**

### **If Ollama Server Not Running**
```bash
cmd /c "set OLLAMA_MODELS=E:\Ollama\models && set OLLAMA_HOST=0.0.0.0:11434 && E:\Ollama\ollama.exe serve"
```

### **If Memory Issues**
- Run `python quick_memory_check.py`
- Close unnecessary applications
- Monitor total usage <15 GB

### **If Model Not Responding**
- Test with `python test_dolphin3.py`
- Check `ollama_server.log` for errors
- Restart server if needed

---

**Created**: September 5, 2025
**Purpose**: Complete handoff instructions for LLM optimization phase
**Status**: Ready for immediate execution
**Previous Phase**: RAG optimization and memory management (COMPLETED)
