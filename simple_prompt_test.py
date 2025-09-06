#!/usr/bin/env python3
"""
Simple prompt size test - memory conscious
"""

def estimate_tokens(text):
    return len(text) // 4

# Test minimal prompt formats
test_statement = "The company reported record profits but was unable to pay suppliers."

print("🎯 SIMPLE PROMPT SIZE COMPARISON")
print("=" * 50)

# Current format (estimated)
current_prompt = f"""
You are analyzing statements for outpoints according to L. Ron Hubbard's methodology.

OUTPOINT: contrary_facts
DEFINITION: Two or more facts that contradict each other

STATEMENT TO ANALYZE: {test_statement}

Please analyze if this statement contains the outpoint 'contrary_facts'.
Provide your response in this format:
RESULT: YES or NO
CONFIDENCE: 0-100
REASONING: Brief explanation
"""

# Minimal format
minimal_prompt = f"""Contrary facts in: "{test_statement}"
YES/NO + confidence 0-100:"""

# Example-based format  
example_prompt = f"""Examples:
- "Record profits but can't pay bills" → YES
- "Sales increased 15%" → NO

Statement: {test_statement}
Contrary facts: YES/NO (0-100% confidence)"""

print(f"Current format: {len(current_prompt)} chars (~{estimate_tokens(current_prompt)} tokens)")
print(f"Minimal format: {len(minimal_prompt)} chars (~{estimate_tokens(minimal_prompt)} tokens)")
print(f"Example format: {len(example_prompt)} chars (~{estimate_tokens(example_prompt)} tokens)")

print(f"\nSavings with minimal: {len(current_prompt) - len(minimal_prompt)} chars")
print(f"Savings with examples: {len(current_prompt) - len(example_prompt)} chars")

print("\n📄 MINIMAL PROMPT:")
print(minimal_prompt)

print("\n📄 EXAMPLE PROMPT:")
print(example_prompt)

print("\n✅ Analysis complete - no memory intensive operations")
