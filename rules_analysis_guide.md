# Holistic Analysis Guide for Truth Algorithm

## Overview
This guide provides instructions for analyzing statements holistically, considering all outpoints and pluspoints simultaneously rather than evaluating them individually.

## Holistic Analysis Approach

When analyzing a statement, consider:
1. The statement in its entirety
2. The context in which it was made
3. Related statements that provide additional context
4. All possible logical issues (outpoints) and strengths (pluspoints) at once

## Outpoint and Pluspoint Relationships

Certain outpoints and pluspoints often appear together or have relationships:

- **Falsehood** often appears with **Dropped Time** or **Wrong Target**
- **Omitted Data** frequently relates to **Altered Sequence**
- **Contrary Facts** may indicate **Wrong Source** or **Wrong Target**
- **Data Proven Factual** is incompatible with **Falsehood**
- **Adequate Data** contradicts **Omitted Data**

## Holistic Evaluation Process

1. Read the statement carefully
2. Consider all available context
3. Identify any immediate logical issues or strengths
4. Check for relationships between identified outpoints/pluspoints
5. Evaluate the overall logical consistency and factual accuracy
6. Assign confidence level to your evaluation

## Example Holistic Analysis

**Statement**: "The factory reported record profits in Q1 2025, despite having been closed for renovations during that entire period."

**Holistic Analysis**:
- **Outpoints**: contrary_facts, falsehood
- **Pluspoints**: none
- **Confidence**: 90
- **Reasoning**: The statement contains a logical contradiction (contrary_facts) - a factory cannot generate record profits while closed. This strongly suggests the statement contains false information (falsehood).

## Common Patterns to Watch For

1. **Temporal inconsistencies**: Watch for statements that contain impossible time sequences
2. **Logical contradictions**: Identify statements that contradict themselves internally
3. **Factual impossibilities**: Note claims that violate known facts or physical laws
4. **Source reliability patterns**: Consider patterns of reliability/unreliability from sources
5. **Contextual inconsistencies**: Identify statements that don't align with established context

## Confidence Assessment

When determining confidence in your evaluation:
- High confidence (80-100): Clear, unambiguous evidence of outpoints/pluspoints
- Medium confidence (50-79): Some evidence, but with potential alternative interpretations
- Low confidence (0-49): Limited evidence, highly context-dependent or ambiguous

## Integration with RAG System

When using the RAG system:
1. Formulate queries that capture the essence of the statement
2. Consider retrieving context about specific entities mentioned
3. Use retrieved information to enhance your evaluation
4. Balance retrieved information with the statement's internal logic