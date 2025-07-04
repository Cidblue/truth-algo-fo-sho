# DeBERTa Training Data Labeling Guide

## Overview
This guide helps you label statements for training the DeBERTa classifier to detect outpoints (logical errors) and pluspoints (logical strengths) in text.

## Label Categories

### Outpoints (Logical Errors)
Use these labels when a statement exhibits logical problems:

1. **OMITTED_DATA_OUT** - Missing crucial information
   - Example: "The company reported profits" (missing: how much? when? compared to what?)
   - Look for: Incomplete information, missing context, vague statements

2. **FALSEHOOD_OUT** - Demonstrably untrue or misleading statements  
   - Example: "Everyone knows the CEO is corrupt" (unsubstantiated claim)
   - Look for: Unverified claims, obvious lies, misleading information

3. **WRONG_SOURCE_OUT** - Information from unreliable or inappropriate sources
   - Example: "According to rumors..." or "People say..."
   - Look for: Vague sources, unqualified sources, hearsay

4. **CONTRARY_FACTS_OUT** - Contradictory information within the statement
   - Example: "The profitable company filed for bankruptcy"
   - Look for: Internal contradictions, conflicting claims

5. **ALTERED_IMPORTANCE_OUT** - Exaggeration or minimization
   - Example: "The minor incident caused worldwide panic"
   - Look for: Overstatement, understatement, inappropriate emphasis

6. **WRONG_TARGET_OUT** - Misplaced blame or attribution
   - Example: "The weather caused the software bug"
   - Look for: Incorrect causation, misattributed responsibility

### Pluspoints (Logical Strengths)
Use these labels when a statement exhibits logical strengths:

1. **DATA_PROVEN_FACTUAL_PLUS** - Verified, factual information
   - Example: "According to the official government report released yesterday..."
   - Look for: Cited sources, verifiable facts, official data

2. **CORRECT_SOURCE_PLUS** - Information from reliable, appropriate sources
   - Example: "The medical journal published findings..."
   - Look for: Expert sources, official sources, credible authorities

3. **TIME_NOTED_PLUS** - Proper time references
   - Example: "On January 15, 2025, the company announced..."
   - Look for: Specific dates, clear timeframes, temporal context

4. **ADEQUATE_DATA_PLUS** - Sufficient information provided
   - Example: "Sales increased 15% from $2M to $2.3M in Q4 2024"
   - Look for: Complete information, sufficient detail, clear metrics

5. **APPLICABLE_DATA_PLUS** - Relevant information for the context
   - Example: "The automotive safety report shows crash test results"
   - Look for: Relevant information, appropriate context, on-topic data

### Neutral
Use **NEUTRAL** for statements that don't clearly exhibit outpoints or pluspoints:
- Simple factual statements without obvious logical issues
- Neutral reporting without clear strengths or weaknesses
- Statements that are neither particularly good nor bad logically

## Labeling Examples

### Good Examples:

**Statement**: "Daniel Graham, 39, and Adam Carruthers, 32, both from Cumbria, deny the charges."
**Label**: TIME_NOTED_PLUS or ADEQUATE_DATA_PLUS
**Reason**: Provides specific details (ages, location, clear action)

**Statement**: "She said she herself had no intention of giving up the lucrative trade but felt others should be aware of what could lie ahead of them."
**Label**: OMITTED_DATA_OUT  
**Reason**: Missing context - what trade? what lies ahead? who is "she"?

**Statement**: "Joe Biden has told the BBC that pressure from the Trump administration on Ukraine to give up territory to Russia is 'modern-day appeasement' in an exclusive interview, his first since leaving the White House."
**Label**: CORRECT_SOURCE_PLUS
**Reason**: Clear source (BBC interview), specific attribution, verifiable

### Borderline Cases:
- When in doubt, choose NEUTRAL
- If a statement has both outpoints and pluspoints, choose the more prominent one
- Consider the overall logical quality of the statement

## Labeling Process

1. **Read the statement carefully**
2. **Identify the main claim or information**
3. **Look for logical strengths or weaknesses**
4. **Choose the most appropriate label**
5. **Add notes if the decision was difficult**

## Quality Guidelines

- **Consistency**: Apply the same standards throughout
- **Context**: Consider what information a reader would need
- **Objectivity**: Focus on logical structure, not content agreement
- **Clarity**: When unsure, lean toward NEUTRAL

## Common Mistakes to Avoid

- Don't label based on political agreement/disagreement
- Don't assume missing information is always OMITTED_DATA_OUT (some statements are naturally brief)
- Don't label every quote as CORRECT_SOURCE_PLUS (consider the source quality)
- Don't over-analyze simple statements

## Next Steps

After labeling:
1. Review your labels for consistency
2. Ensure you have a balanced distribution across categories
3. Run the training script: `python models/train_deberta.py --data_path data/statements_to_label.csv`
