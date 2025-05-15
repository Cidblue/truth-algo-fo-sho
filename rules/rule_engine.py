import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_patterns() -> Dict:
    """Load regex patterns from patterns.yml file."""
    patterns_path = Path(__file__).parent / "patterns.yml"
    if patterns_path.exists():
        with open(patterns_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Fallback to hardcoded patterns if file doesn't exist
        return {
            "TIME_NOTED": {
                "plus": [r"\b(today|yesterday|last week|this month)\b"],
                "out": [r"\b(no date|undated|unknown time)\b"]
            },
            "RELATED_FACTS_KNOWN": {
                "plus": [r"\b(according to|as reported by|per the)\b.*\b(report|study|dataset)\b"],
                "out": [r"\b(unverified|no source|unknown origin)\b"]
            }
        }


def rule_classify(text: str) -> List[Tuple[str, float]]:
    """
    Apply regex rules to classify text.
    Returns list of (label, confidence) pairs.
    Confidence near 0.0 = strong PLUS, near 1.0 = strong OUT
    """
    patterns = load_patterns()

    results = []
    for axis, sub in patterns.items():
        fired_plus = any(re.search(rx, text, re.I)
                         for rx in sub.get("plus", []))
        fired_out = any(re.search(rx, text, re.I) for rx in sub.get("out", []))

        # Determine result based on which patterns matched
        if fired_plus and not fired_out:
            results.append((f"{axis}_PLUS", 0.05))  # High confidence pluspoint
        elif fired_out and not fired_plus:
            results.append((f"{axis}_OUT", 0.95))   # High confidence outpoint

    return results
