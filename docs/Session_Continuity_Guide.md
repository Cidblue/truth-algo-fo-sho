# Session Continuity Guide

This guide explains how to maintain perfect continuity between work sessions using the Truth Algorithm's session management system.

## 🎯 **Core Concept**

The **`SESSION_STATUS.md`** file is your **single source of truth** for:

- Where you left off
- What you accomplished last session
- What to do next
- Current progress and status

## 📋 **Key Files**

1. **`SESSION_STATUS.md`** - Main session status file (read this first each session)
2. **`utils/update_session_status.py`** - Utility to check progress and update status
3. **`README.md`** - Quick reference to session continuity system

## 🚀 **Session Workflow**

### **Starting a New Session**

**Step 1: Check where you left off**

```bash
# Option A: Read the full status file
cat SESSION_STATUS.md

# Option B: Get a quick summary
python utils/update_session_status.py
```

**Step 2: Follow the "IMMEDIATE NEXT STEPS" section**
The SESSION_STATUS.md file always has a clear "IMMEDIATE NEXT STEPS" section telling you exactly what to do.

### **During a Session**

**Check progress anytime:**

```bash
# Quick status summary
python utils/update_session_status.py

# Check specific labeling progress
python utils/manual_labeling_helper.py --progress_only

# Test current system
python truth_algorithm.py sample.json --rules-only -v
```

### **Ending a Session**

**Two ways to update the session file:**

**Option 1: Manual Update (You type notes)**

```bash
python utils/update_session_status.py --update
```

- Script prompts for notes
- You type what you accomplished
- Press Enter to save

**Option 2: AI Update (AI updates with context)**

- Simply ask the AI: "Please update the session status with our progress"
- AI will directly edit SESSION_STATUS.md with comprehensive notes
- Better for complex sessions where AI has full context

**What gets updated automatically:**

- Current timestamp
- Updated progress numbers (e.g., labeling progress)
- Session notes (either yours or AI-generated)

## 📖 **Detailed Command Reference**

### **`python utils/update_session_status.py`** (no arguments)

**Purpose**: Show current status summary
**Output**:

- Training data progress (X/100 statements labeled)
- System component status (✅/❌ for key files)
- Next steps based on current progress

**Example Output:**

```
============================================================
TRUTH ALGORITHM - CURRENT STATUS
============================================================
📊 Training Data Progress:
   Total statements: 100
   Labeled: 9 (9.0%)
   Unlabeled: 91
   🎯 Need more labels for training

🔧 System Components:
   ✅ truth_algorithm.py
   ✅ pipeline/classifier.py
   ✅ models/deberta_classifier.py

🎯 NEXT STEPS:
   1. Continue labeling data:
      python utils/statement_categorizer.py  (GUI - recommended)
      python utils/manual_labeling_helper.py  (command line)
============================================================
```

### **`python utils/update_session_status.py --update`**

**Purpose**: Update SESSION_STATUS.md with current progress and notes
**Interactive Process**:

1. Script calculates current progress automatically
2. Prompts for optional session notes
3. Updates SESSION_STATUS.md with new timestamp, progress, and notes

**Example Session:**

```bash
$ python utils/update_session_status.py --update
Enter session notes (optional): Labeled 15 more statements using GUI tool. Found some tricky cases with ASSUMED_SIMILARITIES. Ready to continue tomorrow.
✅ Updated SESSION_STATUS.md
```

**What gets updated in SESSION_STATUS.md:**

- **Last Updated** timestamp
- **Progress numbers** (e.g., "9/100" becomes "24/100")
- **Session Notes** section with your notes and timestamp

### **`python utils/update_session_status.py --help`**

**Purpose**: Show usage help
**Output**: Command options and brief descriptions

## 🔧 **Other Key Commands for Session Management**

### **Labeling Progress Commands**

```bash
# Detailed labeling progress with label distribution
python utils/manual_labeling_helper.py --progress_only

# Start/resume manual labeling
python utils/manual_labeling_helper.py

# Start GUI labeling tool (recommended)
python utils/statement_categorizer.py

# Check if training data file exists and is readable
ls -la data/statements_to_label.csv
```

### **System Testing Commands**

```bash
# Test current system (rules + LLM only)
python truth_algorithm.py sample.json --rules-only -v

# Test full system (after DeBERTa training)
python truth_algorithm.py sample.json -v

# Quick import test
python -c "from truth_algorithm import TruthAlgorithm; print('System imports OK')"
```

### **Training Commands** (when ready)

```bash
# Check if ready for training (need 50+ labeled statements)
python utils/update_session_status.py

# Install training dependencies
pip install datasets transformers torch scikit-learn

# Train DeBERTa model
python models/train_deberta.py --data_path data/statements_to_label.csv
```

## 📝 **Session Notes Best Practices**

### **Good Session Notes Examples:**

- `"Labeled 20 statements using GUI tool. Focused on OMITTED_DATA and WRONG_SOURCE cases."`
- `"Fixed bug in contradiction detection. System now properly assigns contrary_facts outpoint."`
- `"Completed DeBERTa training. Model accuracy: 78%. Ready for integration testing."`
- `"Reviewed difficult cases: statements with multiple outpoints. Need to decide on labeling strategy."`

### **What to Include in Notes:**

- **Progress made** (how many statements labeled, what fixed, etc.)
- **Issues encountered** (bugs found, difficult decisions, etc.)
- **Next priorities** (what to focus on next session)
- **Important decisions** (labeling strategies, parameter choices, etc.)

### **When to Update:**

- **End of each work session** (always)
- **After major milestones** (completing training, fixing bugs, etc.)
- **Before taking a break** (if you might forget context)

## 🎯 **Quick Reference Card**

**Starting a session:**

```bash
cat SESSION_STATUS.md                    # Read current status
python utils/update_session_status.py    # Quick summary
```

**During work:**

```bash
python utils/update_session_status.py    # Check progress
```

**Ending a session:**

```bash
python utils/update_session_status.py --update    # Update with notes
```

**The notes prompt is interactive** - you type your notes when prompted, then press Enter. No separate notes file needed!

## 🔄 **Example Complete Session**

```bash
# 1. Start session - check status
$ cat SESSION_STATUS.md
# (Read current objective and next steps)

# 2. Work on current objective
$ python utils/statement_categorizer.py
# (Label some statements)

# 3. Check progress during work
$ python utils/update_session_status.py
# (See updated progress)

# 4. End session - update status
$ python utils/update_session_status.py --update
Enter session notes (optional): Labeled 25 statements today. Focused on time-related outpoints. System working well.
✅ Updated SESSION_STATUS.md
```

This system ensures you **never lose context** between sessions and always know exactly what to do next!
