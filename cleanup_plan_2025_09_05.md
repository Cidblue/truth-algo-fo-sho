# File Cleanup Plan - 2025-09-05

## OBSOLETE FILES TO REMOVE

### **Test Files (Superseded)**
- `prompt_analyzer.py` - Caused OOM crash, superseded by simple_prompt_test.py
- `quick_model_test.py` - Redundant with test_dolphin3.py
- `quick_ollama_test.py` - Redundant with pipeline_readiness_test.py
- `test_ollama_connection.py` - Basic functionality covered by other tests
- `test_ollama_startup.py` - Redundant with startup verification

### **Old Training/Testing Files**
- `simple_round3_training.py` - Only copied model, didn't train (superseded by minimal_round3_training.py)
- `quick_round2_test.py` - Round 2 specific, now using Round 3
- `quick_training_test.py` - Generic test, superseded by specific tests
- `simple_validation_test.py` - Superseded by comprehensive tests

### **Redundant Analysis Files**
- `analyze_data_quality.py` - Functionality integrated into other tools
- `analyze_training_data.py` - Superseded by manual review tools
- `simple_error_analysis.py` - Basic functionality, superseded by detailed analysis

### **Old Configuration/Batch Files**
- `temp_server_wrapper.bat` - Temporary file, no longer needed
- `capture_server_output.py` - Functionality integrated into server management
- `read_server_logs.py` - Basic functionality, superseded by direct log viewing

### **Redundant Data Files**
- `sample.json` - Generic sample, superseded by specific test data
- `simple_test.json` - Basic test, superseded by pipeline_test_data.json
- `test_simple_statement.json` - Superseded by comprehensive test data

### **Old Result Files (Keep for Reference)**
- `sample_results.json` - Old results, keep in archive
- `simple_test_results.json` - Old results, keep in archive
- `test_simple_statement_results.json` - Old results, keep in archive

## FILES TO ARCHIVE (NOT DELETE)

### **Historical Training Results**
- `round2_simple_results.json` - Historical data
- `round2_validation_results.json` - Historical data
- `deberta_comparison_results.json` - Historical comparison
- `deberta_threshold_test_results.json` - Historical threshold data

### **Old Training Scripts (Keep for Reference)**
- `train_round2_28categories.py` - Historical training approach
- `train_round2_model.py` - Historical training approach
- `simple_round2_training.py` - Historical training approach

## FILES TO KEEP (ACTIVE)

### **Current Core Files**
- `truth_algorithm.py` - Main algorithm (updated for dolphin3)
- `models/llm_evaluator.py` - LLM integration (RAG disabled)
- `pipeline/classifier.py` - Classification pipeline
- `models/deberta_classifier.py` - DeBERTa wrapper

### **Current Test Files**
- `test_dolphin3.py` - Current model testing
- `pipeline_readiness_test.py` - System verification
- `simple_prompt_test.py` - Prompt optimization
- `test_deberta_comparison.py` - Model comparison
- `test_fixed_labels.py` - Label verification

### **Current Data Files**
- `pipeline_test_data.json` - Current test data
- `data/round3_sample_dataset.csv` - Current training data
- `llm_cache_dolphin3.pkl` - Current model cache

### **Current Optimization Files**
- `llm_optimization_plan.md` - Current strategy
- `quick_memory_check.py` - Memory management
- `minimal_round3_training.py` - Current training approach

### **Documentation**
- `docs/FILE_REGISTRY.md` - Updated registry
- `ROUND3_CATEGORY_FIX.md` - Important fix documentation
- `SESSION_STATUS.md` - Current status

## CLEANUP ACTIONS

1. **Move to Archive**: Historical results and old training scripts
2. **Delete**: Redundant test files and temporary files
3. **Update Registry**: Document cleanup actions
4. **Verify**: Ensure no dependencies broken

## MEMORY IMPACT

Removing obsolete files will:
- Free up disk space
- Reduce confusion about which files are current
- Improve project navigation
- Maintain clean development environment

## SAFETY

- All deletions will be documented
- Historical data moved to archive, not deleted
- Core functionality preserved
- Registry updated to reflect changes
