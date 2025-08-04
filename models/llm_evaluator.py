import hashlib
import json
import pickle
import re
import requests
import time
from typing import Dict, List, Tuple, Optional, Any

from .rag_implementation import retrieve_context


class LLMEvaluator:
    """Class to handle LLM-based evaluation of statements for outpoints and pluspoints."""

    def __init__(
        self,
        model_name: str = "truth-evaluator",
        api_url: str = "http://localhost:11434/api/generate",
        cache_file: str = "llm_cache.pkl",
        batch_size: int = 1,
        use_rag: bool = True,
        max_chunks: int = 3,
        timeout: int = 600
    ):
        """Initialize the LLM evaluator with model configuration."""
        self.model_name = model_name
        self.api_url = api_url
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.use_rag = use_rag
        self.max_chunks = max_chunks
        self.timeout = timeout

        # Load cache if it exists
        try:
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Loaded {len(self.cache)} cached evaluations")
        except (FileNotFoundError, EOFError):
            self.cache = {}
            print("Created new evaluation cache")

        # Initialize vector store and embedding model if RAG is enabled
        if use_rag:
            try:
                from .rag_implementation import init_vector_store, init_embedding_model
                self.vector_store = init_vector_store()
                self.embedding_model = init_embedding_model()
            except ImportError:
                print("Warning: RAG implementation not found, disabling RAG")
                self.use_rag = False

    def evaluate_outpoint(self, rule_name: str, statement_text: str, context: Dict = None) -> Tuple[bool, float]:
        """Evaluate if a statement exhibits a specific outpoint."""
        print(
            f"\nEvaluating outpoint '{rule_name}' for: '{statement_text[:40]}...'")

        # Create the prompt for this evaluation
        prompt = self._create_outpoint_prompt(
            rule_name, statement_text, context)

        # Query the LLM
        result = self._query_llm(prompt)

        # Parse the result
        decision, confidence = self._parse_llm_result(result)

        # Print the result
        print(
            f"  - {rule_name}: {'YES' if decision else 'NO'} (confidence: {confidence:.2f})")

        return decision, confidence

    def evaluate_pluspoint(self, rule_name: str, statement_text: str, context: Dict = None) -> Tuple[bool, float]:
        """Evaluate if a statement exhibits a specific pluspoint."""
        # Create cache key
        cache_key = f"plus_{rule_name}_{hashlib.md5(statement_text.encode()).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Create prompt based on rule type
        prompt = self._create_pluspoint_prompt(
            rule_name, statement_text, context)

        # Get LLM response
        result = self._query_llm(prompt)

        # Parse result to get boolean decision and confidence
        has_pluspoint, confidence = self._parse_llm_result(result)

        # Cache the result
        self.cache[cache_key] = (has_pluspoint, confidence)

        return has_pluspoint, confidence

    def batch_evaluate(self, statements: List[str], rule_name: str, rule_type: str = "outpoint") -> List[Tuple[bool, float]]:
        """Evaluate multiple statements against a rule in a single batch."""
        if self.batch_size <= 1:
            # Fall back to individual evaluation if batch processing is disabled
            return [self.evaluate_outpoint(rule_name, stmt) if rule_type == "outpoint"
                    else self.evaluate_pluspoint(rule_name, stmt) for stmt in statements]

        # Prepare batch prompt
        batch_prompt = f"""You are an expert in logical analysis and critical thinking.
...

Here are the statements to analyze:
"""

        for i, stmt in enumerate(statements):
            batch_prompt += f"\nSTATEMENT #{i+1}: \"{stmt}\"\n"

        # Query LLM with batch prompt
        result = self._query_llm(batch_prompt)

        # Parse results
        results = []
        for i in range(len(statements)):
            pattern = rf"STATEMENT #{i+1}.*?RESULT:\s*(\w+).*?CONFIDENCE:\s*(\d+)"
            match = re.search(pattern, result, re.DOTALL)
            if match:
                decision = match.group(1).upper() == "YES"
                confidence = float(match.group(2)) / 100.0
                results.append((decision, confidence))
            else:
                # Default if parsing fails
                results.append((False, 0.0))

        return results

    def _create_outpoint_prompt(self, rule_name: str, statement_text: str, context: Dict = None) -> str:
        """Create a prompt for evaluating an outpoint rule."""
        from rules import outpoint_descriptions

        # Create context string if provided
        context_str = ""
        if context and 'related_statements' in context:
            context_str = "CONTEXT:\n"
            for i, stmt in enumerate(context['related_statements']):
                context_str += f"{i+1}. {stmt}\n"

        # Add RAG context if enabled
        rag_context = ""
        if self.use_rag:
            query = f"{rule_name} {statement_text}"
            rag_context = retrieve_context(
                query, self.vector_store, self.embedding_model, self.max_chunks)
            if rag_context:
                rag_context = "Relevant information from knowledge base (CITE THESE IN YOUR REASONING):\n" + \
                    rag_context

        # Build the prompt
        prompt = f"""Evaluate the following statement for the outpoint "{rule_name.replace('_', ' ')}".

DEFINITION: {outpoint_descriptions.get(rule_name, f"Evaluate if this statement exhibits the '{rule_name}' logical error.")}

{context_str}

{rag_context}

STATEMENT TO ANALYZE: "{statement_text}"

Provide your determination in this format:
RESULT: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your brief explanation with citations to knowledge base if relevant]
KNOWLEDGE_USED: [YES/NO - indicate if you used the provided knowledge base]
"""
        return prompt

    def _create_pluspoint_prompt(self, rule_name: str, statement_text: str, context: Dict = None) -> str:
        """Create a prompt for evaluating a pluspoint rule."""
        from rules import pluspoint_descriptions

        # Create context string if provided
        context_str = ""
        if context and 'related_statements' in context:
            context_str = "Related statements:\n"
            for i, stmt in enumerate(context['related_statements']):
                context_str += f"{i+1}. {stmt}\n"

        # Add RAG context if enabled
        rag_context = ""
        if self.use_rag:
            query = f"{rule_name} {statement_text}"
            rag_context = retrieve_context(
                query, self.vector_store, self.embedding_model, self.max_chunks)
            if rag_context:
                rag_context = "Relevant information from knowledge base:\n" + rag_context

        # Build the prompt
        prompt = f"""You are an expert in logical analysis and critical thinking, evaluating statements for logical consistency and accuracy.

TASK: Analyze the following statement to determine if it exhibits the logical strength known as "{rule_name.replace('_', ' ')}".

DEFINITION: {pluspoint_descriptions.get(
    rule_name, f"Evaluate if this statement exhibits the '{rule_name}' logical strength.")}

{context_str}

{rag_context}

STATEMENT TO ANALYZE: "{statement_text}"

Analyze the statement step by step, then provide your final determination in this exact format:
RESULT: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Your brief explanation]
"""
        return prompt

    def _query_llm(self, prompt: str, max_retries: int = 3, timeout: int = 600) -> str:
        """Send a query to the local LLM API and get the response with retry logic."""
        # Check cache first to avoid duplicate evaluations
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.cache:
            print("  - Using cached response")
            return self.cache[cache_key]

        # Use the instance timeout if available, otherwise use the parameter
        actual_timeout = getattr(self, 'timeout', timeout)

        original_prompt = prompt
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 300
                }

                print(f"Querying LLM (attempt {attempt+1}/{max_retries})...")
                response = requests.post(
                    self.api_url, headers=headers, data=json.dumps(data), timeout=actual_timeout)

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")

                    # Store in cache
                    self.cache[cache_key] = response_text
                    self._save_cache()  # Save cache after each successful query
                    return response_text
                else:
                    print(f"  - Error: HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                print(
                    f"  - Error: Request timed out after {actual_timeout} seconds")
                # Reduce prompt size for next attempt
                if attempt < max_retries - 1:
                    print("  - Reducing prompt size for next attempt")
                    # Progressively reduce prompt by removing context sections
                    if "Relevant information from knowledge base" in prompt:
                        # Remove RAG context first
                        prompt = re.sub(r"Relevant information from knowledge base.*?(?=\n\nSTATEMENT TO ANALYZE)",
                                        "", prompt, flags=re.DOTALL)
                    elif "CONTEXT:" in prompt:
                        # Then remove context if present
                        prompt = re.sub(r"CONTEXT:.*?(?=\n\nSTATEMENT TO ANALYZE)",
                                        "", prompt, flags=re.DOTALL)
                    else:
                        # Last resort: truncate the prompt
                        prompt = prompt[:len(prompt)//2] + "\n\nSTATEMENT TO ANALYZE:" + \
                            original_prompt.split("STATEMENT TO ANALYZE:")[-1]
            except Exception as e:
                print(f"  - Error: {str(e)}")

            # Only sleep if we're going to retry
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff

        # If all retries fail, return a default response
        default_response = "RESULT: NO\nCONFIDENCE: 50\nREASONING: Unable to evaluate due to LLM timeout."
        print("  - All retries failed, using default response")
        self.cache[cache_key] = default_response
        self._save_cache()
        return default_response

    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def _parse_llm_result(self, result: str) -> Tuple[bool, float]:
        """Parse the LLM response to extract the decision and confidence score with improved robustness."""
        # Look for the RESULT and CONFIDENCE patterns in the response
        result_match = re.search(r'RESULT:\s*(\w+)', result, re.IGNORECASE)
        confidence_match = re.search(
            r'CONFIDENCE:\s*(\d+)', result, re.IGNORECASE)

        # If standard pattern fails, try alternative patterns
        if not result_match:
            # Look for YES/NO directly
            if re.search(r'\bYES\b', result, re.IGNORECASE):
                decision_value = 'YES'
                result_match = type('obj', (object,), {
                                    'group': lambda x: decision_value})
            elif re.search(r'\bNO\b', result, re.IGNORECASE):
                decision_value = 'NO'
                result_match = type('obj', (object,), {
                                    'group': lambda x: decision_value})

        if not confidence_match:
            # Look for percentage patterns like "90%" or "confidence is 90"
            alt_match = re.search(r'(\d{1,3})%', result) or re.search(
                r'confidence\D+(\d{1,3})', result, re.IGNORECASE)
            if alt_match:
                confidence_value = alt_match.group(1)
                confidence_match = type(
                    'obj', (object,), {'group': lambda x: confidence_value})

        if result_match and confidence_match:
            # Extract the decision (YES/NO) and convert to boolean
            decision = result_match.group(1).upper() == "YES"

            # Extract the confidence score and normalize to 0-1 range
            confidence = float(confidence_match.group(1)) / 100.0
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            return decision, confidence
        else:
            # If we can't parse the result properly, return a default
            print("  - Warning: Could not parse LLM result properly")
            # Try to make an educated guess based on the content
            likely_yes = any(term in result.lower() for term in [
                'yes', 'correct', 'true', 'accurate', 'valid'])
            likely_no = any(term in result.lower() for term in [
                            'no', 'incorrect', 'false', 'inaccurate', 'invalid'])

            if likely_yes and not likely_no:
                return True, 0.6
            elif likely_no and not likely_yes:
                return False, 0.6
            else:
                return False, 0.5

    def evaluate_statement_holistically(self, statement_text: str, context: Dict = None) -> Dict:
        """Evaluate a statement holistically for all outpoints and pluspoints."""
        # Create cache key
        cache_key = f"holistic_{hashlib.md5(statement_text.encode()).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Create context string if provided
        context_str = ""
        if context and context.get("related_statements"):
            context_str = "RELATED STATEMENTS:\n"
            for i, stmt in enumerate(context["related_statements"]):
                context_str += f"{i+1}. {stmt}\n"

        # Add RAG context if enabled
        rag_context = ""
        rag_sources = []
        if self.use_rag:
            query = f"analyze statement: {statement_text}"
            # Default to 0.0 if not set
            min_score = getattr(self, 'min_score', 0.0)
            rag_context = retrieve_context(
                query, self.vector_store, self.embedding_model,
                self.max_chunks, min_score)

            # Extract source information for verification
            if rag_context:
                source_matches = re.findall(r"from (.*?)\)", rag_context)
                rag_sources = source_matches
                print(f"  - RAG sources used: {', '.join(rag_sources)}")
                rag_context = "RELEVANT INFORMATION (MUST USE THIS IN YOUR ANALYSIS):\n" + \
                    rag_context

        # Detect obvious outpoints before LLM evaluation
        obvious_outpoints = []
        if "!!" in statement_text or "everyone knows" in statement_text.lower():
            obvious_outpoints.append("wrong_target")
            obvious_outpoints.append("generality")

        # Build the prompt with stronger instructions
        prompt = f"""You are an expert in logical analysis and critical thinking, evaluating statements for logical consistency and accuracy.

TASK: Analyze the following statement to identify any logical errors (outpoints) or strengths (pluspoints).

STATEMENT TO ANALYZE: "{statement_text}"

{context_str}

{rag_context}

IMPORTANT INSTRUCTIONS:
1. You MUST use the relevant information provided above in your analysis
2. Cite specific excerpts when making your determination
3. Be critical and thorough in your analysis
4. Look for both outpoints (logical errors) and pluspoints (logical strengths)

Analyze the statement step by step, then provide your final determination in this exact format:
OUTPOINTS: [List all outpoints found, separated by commas]
PLUSPOINTS: [List all pluspoints found, separated by commas]
CONFIDENCE: [0-100]
SOURCES_USED: [List any sources you referenced from the provided information]
REASONING: [Your brief explanation]
"""

        # Query LLM
        result = self._query_llm(prompt)

        # Parse result
        outpoints, pluspoints, confidence = self._parse_holistic_result(result)

        # Add any obvious outpoints detected programmatically
        for outpoint in obvious_outpoints:
            if outpoint not in outpoints:
                outpoints.append(outpoint)
                print(f"  - Added obvious outpoint: {outpoint}")

        # Create result dictionary
        evaluation_result = {
            "outpoints": outpoints,
            "pluspoints": pluspoints,
            "confidence": confidence,
            "rag_sources": rag_sources
        }

        # Cache the result
        self.cache[cache_key] = evaluation_result
        self._save_cache()

        return evaluation_result

    def _parse_holistic_result(self, result: str) -> Tuple[List[str], List[str], float]:
        """Parse the holistic evaluation result from LLM."""
        outpoints = []
        pluspoints = []
        confidence = 0.5  # Default confidence

        # Extract outpoints
        outpoints_match = re.search(
            r"OUTPOINTS:\s*(.*?)(?:\n|$)", result, re.IGNORECASE)
        if outpoints_match:
            outpoints_text = outpoints_match.group(1).strip().lower()
            if outpoints_text and "none" not in outpoints_text:
                outpoints = [op.strip() for op in outpoints_text.split(",")]

        # Extract pluspoints
        pluspoints_match = re.search(
            r"PLUSPOINTS:\s*(.*?)(?:\n|$)", result, re.IGNORECASE)
        if pluspoints_match:
            pluspoints_text = pluspoints_match.group(1).strip().lower()
            if pluspoints_text and "none" not in pluspoints_text:
                pluspoints = [pp.strip() for pp in pluspoints_text.split(",")]

        # Extract confidence
        confidence
