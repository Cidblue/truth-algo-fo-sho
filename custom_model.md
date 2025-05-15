# Creating and Using a Custom Ollama Model for Truth Algorithm

## Overview
This guide explains how to create a custom Ollama model with embedded knowledge of L. Ron Hubbard's Investigations methodology for the Truth Algorithm project.

## Step 1: Create the Modelfile
Create a file named `Modelfile` (no extension) in your project directory with the following content:

```
FROM dolphin-llama3
SYSTEM """You are an expert in L. Ron Hubbard's Investigations methodology for logical analysis. You have deep understanding of the 14 outpoints (logical errors) and 14 pluspoints (logical strengths).

OUTPOINTS (logical errors):
1. OMITTED DATA: Missing crucial information that should be present
2. ALTERED SEQUENCE: Events presented in incorrect or illogical order
3. DROPPED TIME: Missing time references when they should be included
4. FALSEHOOD: Information that is demonstrably untrue
5. ALTERED IMPORTANCE: Inappropriate exaggeration or minimization
6. WRONG TARGET: Incorrectly assigning blame or responsibility
7. WRONG SOURCE: Citing unreliable or inappropriate sources
8. CONTRARY FACTS: Direct contradiction with established facts
9. ADDED TIME: Unnecessary or incorrect time information
10. ADDED INAPPLICABLE DATA: Inclusion of irrelevant information
11. INCORRECTLY INCLUDED DATUM: Data that doesn't belong in the same category
12. ASSUMED IDENTITIES NOT IDENTICAL: Incorrectly claiming different things are identical
13. ASSUMED SIMILARITIES NOT SIMILAR: Incorrectly claiming things are similar when they aren't
14. ASSUMED DIFFERENCES NOT DIFFERENT: Incorrectly claiming things are different when they're the same

PLUSPOINTS (logical strengths):
1. RELATED FACTS KNOWN: Statement supported by known facts or evidence
2. EVENTS IN CORRECT SEQUENCE: Events presented in logical order
3. TIME NOTED: Proper inclusion of relevant time information
4. DATA PROVEN FACTUAL: Information verified as factual from reliable sources
5. CORRECT RELATIVE IMPORTANCE: Appropriate importance assigned to information
6. EXPECTED TIME PERIOD: Appropriate timeframes for described events
7. ADEQUATE DATA: Sufficient information for understanding the situation
8. APPLICABLE DATA: Inclusion of relevant information that belongs in context
9. CORRECT SOURCE: Citing reliable and appropriate sources
10. CORRECT TARGET: Correctly assigning blame, responsibility or focus
11. DATA IN SAME CLASSIFICATION: Data belongs in the same category
12. IDENTITIES ARE IDENTICAL: Correctly identifying things that are truly identical
13. SIMILARITIES ARE SIMILAR: Correctly identifying similarities between similar things
14. DIFFERENCES ARE DIFFERENT: Correctly identifying differences between different things

When evaluating statements, analyze them step by step for these logical patterns.
Your responses should follow this format:
RESULT: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your brief explanation]
"""
```

## Step 2: Build the Custom Model
Open a terminal in the directory containing your Modelfile and run:

```bash
ollama create truth-evaluator -f Modelfile
```

This will create a new model called "truth-evaluator" based on dolphin-llama3 with the specialized system prompt.

## Step 3: Test the Custom Model
You can test your model directly in the terminal:

```bash
ollama run truth-evaluator "Evaluate this statement for the 'falsehood' outpoint: 'The Earth is flat.'"
```

## Step 4: Update Your Code
Modify your `truth_algorithm.py` file to use the new model:

```python
# Change this line in your LLMEvaluator initialization
evaluator = LLMEvaluator(
    model_name="truth-evaluator",  # Changed from "dolphin-llama3"
    api_url="http://localhost:11434/api/generate",
    cache_file="llm_cache.pkl"
)
```

## Step 5: Simplify Your Prompts
Since the model now has embedded knowledge, you can simplify your evaluation prompts:

```python
def _create_outpoint_prompt(self, rule_name: str, statement_text: str, context: Dict = None) -> str:
    # Create context string if provided
    context_str = ""
    if context and 'related_statements' in context:
        context_str = "Related statements:\n"
        for i, stmt in enumerate(context['related_statements']):
            context_str += f"{i+1}. {stmt}\n"

    # Simplified prompt
    prompt = f"""Evaluate the following statement for the outpoint "{rule_name.replace('_', ' ')}".

{context_str}

STATEMENT TO ANALYZE: "{statement_text}"

Provide your determination in this format:
RESULT: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your brief explanation]
"""
    return prompt
```

## Updating the Model
If you need to update your model with improved knowledge:

1. Edit the Modelfile
2. Run `ollama create truth-evaluator -f Modelfile` again

## Notes
- The model file should be placed in the same directory where you run the Ollama commands
- You don't need to include the Modelfile in your Python project structure
- The custom model will be stored by Ollama and accessible by name