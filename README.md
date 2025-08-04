# Truth Algorithm

A comprehensive implementation of L. Ron Hubbard's Investigations methodology for logical analysis, evaluating statements based on 14 outpoints (logical errors) and 14 pluspoints (logical strengths).

## 🎯 Overview

The Truth Algorithm uses a multi-layer approach to analyze statements and detect logical inconsistencies:

1. **Regex Layer**: High-precision pattern matching for obvious cases
2. **DeBERTa Layer**: Semantic understanding via machine learning ✅ **TRAINED**
3. **LLM Layer**: Complex analysis with RAG (Retrieval-Augmented Generation)

## ✅ Current Status

**WORKING FEATURES:**

- ✅ Multi-statement contradiction detection
- ✅ Single-statement outpoint/pluspoint detection
- ✅ Truth score calculation and traffic light classification
- ✅ LLM integration with custom "truth-evaluator" model
- ✅ RAG system with knowledge base retrieval
- ✅ Comprehensive regex patterns for all 14 outpoints/pluspoints
- ✅ Knowledge graph for relational analysis
- ✅ **DeBERTa classifier trained** with complete 28-category coverage
- ✅ **Round 2 model available** at `models/round2-simple/`

## 🚀 Quick Start

### Prerequisites

```bash
pip install networkx transformers torch spacy python-dateutil
python -m spacy download en_core_web_sm
```

### Basic Usage

```bash
# Rules-only analysis (fastest)
python truth_algorithm.py sample.json --rules-only -v

# Full analysis with LLM (requires Ollama)
python truth_algorithm.py sample.json -v

# Disable specific layers
python truth_algorithm.py sample.json --no-llm -v
python truth_algorithm.py sample.json --no-deberta -v
```

### Input Format

Create a JSON file with statements:

```json
[
  {
    "id": "s1",
    "text": "The factory reported record profits in Q1.",
    "time": "2025-01-15",
    "source": "Finance-Dept"
  },
  {
    "id": "s2",
    "text": "We were unable to pay suppliers in Q1.",
    "time": "2025-01-20",
    "source": "Accounts-Payable"
  }
]
```

## 📊 Example Output

```
Found contradiction between s1 and s2: Found contradictory terms: record vs unable

Results:
s1: The factory reported record profits in Q1....
  Classification: Possibly False (score: 0.40)
  Outpoints: contrary_facts
  Pluspoints: None

s2: We were unable to pay suppliers in Q1....
  Classification: Possibly False (score: 0.40)
  Outpoints: contrary_facts
  Pluspoints: None
```

## 🏗️ Architecture

### Project Structure

```
truth-pipeline/
├─ rules/                    # Regex-based pattern matching
│  ├─ patterns.yml          # Comprehensive regex patterns
│  └─ rule_engine.py        # Pattern matching engine
├─ models/                   # ML and LLM components
│  ├─ llm_evaluator.py      # LLM integration with RAG
│  ├─ rag_implementation.py # Knowledge retrieval system
│  └─ deberta_classifier.py # ML classifier (placeholder)
├─ pipeline/                 # Multi-layer orchestration
│  └─ classifier.py         # Pipeline coordination
├─ utils/                    # Supporting utilities
│  └─ build_vector_store.py # Knowledge base creation
├─ truth_graph.py           # Relational analysis
└─ truth_algorithm.py       # Main entry point
```

### The 14 Outpoints (Logical Errors)

1. **Omitted Data** - Missing crucial information
2. **Altered Sequence** - Events out of logical order
3. **Dropped Time** - Missing timestamps
4. **Falsehood** - Demonstrably untrue statements
5. **Altered Importance** - Exaggeration or minimization
6. **Wrong Target** - Misplaced blame or focus
7. **Wrong Source** - Unreliable information origin
8. **Contrary Facts** - Direct contradictions
9. **Added Time** - Unnecessary time data
10. **Added Inapplicable Data** - Irrelevant information
11. **Incorrectly Included Datum** - Data that doesn't belong
12. **Assumed Identities Not Identical** - False equivalences
13. **Assumed Similarities Not Similar** - False analogies
14. **Assumed Differences Not Different** - False distinctions

### The 14 Pluspoints (Logical Strengths)

1. **Related Facts Known** - Supporting evidence exists
2. **Events in Correct Sequence** - Logical order
3. **Time Noted** - Proper timestamps
4. **Data Proven Factual** - Verified information
5. **Correct Relative Importance** - Proper emphasis
6. **Expected Time Period** - Appropriate timeframes
7. **Adequate Data** - Sufficient information
8. **Applicable Data** - Relevant information
9. **Correct Source** - Reliable origin
10. **Correct Target** - Proper attribution
11. **Data in Same Classification** - Consistent categorization
12. **Identities Are Identical** - Proper equivalences
13. **Similarities Are Similar** - Valid comparisons
14. **Differences Are Different** - Valid distinctions

## 🔧 Configuration

### Command Line Options

- `--rules-only`: Use only regex patterns (fastest)
- `--no-llm`: Disable LLM evaluation
- `--no-deberta`: Disable DeBERTa evaluation
- `--no-rag`: Disable RAG system
- `--max-chunks N`: Limit RAG context chunks
- `--min-score X`: Minimum RAG relevance score
- `--timeout N`: LLM query timeout seconds
- `-v, --verbose`: Detailed output

## 📈 Recent Improvements

**Latest Session (Aug 3, 2025) - Round 2 Training Complete:**

- ✅ **CRITICAL FIX**: Resolved 17→28 category issue (missing L. Ron Hubbard categories)
- ✅ **Round 2 Model Trained**: Complete 28-category DeBERTa classifier
- ✅ **Comprehensive Documentation**: Created MODEL_DOCUMENTATION.md
- ✅ **Training Infrastructure**: Multiple training scripts and validation tools
- ✅ **Data Enhancement**: 186 training examples with balanced category coverage
- ✅ **Threshold Optimization**: Improved from 0% to 100% prediction coverage

**Previous Session (Jan 3, 2025):**

- ✅ Fixed pipeline integration bugs
- ✅ Enhanced contradiction detection
- ✅ Improved pattern matching for unsubstantiated claims
- ✅ Added automatic outpoint assignment for contradictions
- ✅ Fixed regex syntax errors

## 🚧 Known Limitations

- **DeBERTa Performance**: Round 2 model has 3.4% accuracy (needs optimization)
- **Performance**: Not optimized for large datasets
- **Domain Specificity**: Patterns are general-purpose, not domain-specific
- **Round 3 Needed**: Hyperparameter tuning required for better accuracy

## 📚 Documentation

- **`SESSION_STATUS.md`** - **Current session status and next steps** (START HERE)
- **`docs/MODEL_DOCUMENTATION.md`** - **Complete model guide and usage** (NEW)
- **`docs/Round2_Progress_Documentation.md`** - **Round 2 training details** (NEW)
- **`docs/Session_Continuity_Guide.md`** - **Complete guide to session management**
- `docs/Plan.txt` - Development roadmap and priorities
- `docs/Algorithm.txt` - Conceptual framework
- `docs/RecentFixes_2025-01-03.md` - Latest improvements
- `CurrentChecklistofState.txt` - Detailed implementation status

## 🔄 Session Continuity

**Between sessions, check**: `SESSION_STATUS.md` for current progress and next steps

**Quick status check**: `python utils/update_session_status.py`

**Update session**: `python utils/update_session_status.py --update`

**Full guide**: See `docs/Session_Continuity_Guide.md` for detailed command reference

## 🤝 Contributing

The system is designed for extensibility. Key areas for contribution:

- DeBERTa model training and integration
- Domain-specific pattern development
- Performance optimization
- Evaluation dashboard creation

## 📄 License

Based on L. Ron Hubbard's Investigations methodology from Scientology materials.
