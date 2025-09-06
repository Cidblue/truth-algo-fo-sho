#!/usr/bin/env python3
"""
Test TruthAlgorithm with dolphin3 model to verify RAG is disabled
and no massive prompts are generated.
"""

from models.llm_evaluator import LLMEvaluator
import time

def test_dolphin3_simple():
    """Test simple LLM evaluation with dolphin3 model."""
    print("🧪 TESTING DOLPHIN3 MODEL")
    print("=" * 40)
    
    # Initialize evaluator with dolphin3 model
    print("Initializing LLM evaluator with dolphin3...")
    evaluator = LLMEvaluator(
        model_name="dolphin3",
        api_url="http://localhost:11434/api/generate",
        cache_file="llm_cache_dolphin3.pkl",
        use_rag=False,  # Explicitly disable RAG
        timeout=120  # 2 minute timeout
    )
    
    # Test simple outpoint evaluation
    print("\nTesting outpoint evaluation...")
    test_statement = "The company reported record profits but was unable to pay suppliers."
    
    start_time = time.time()
    try:
        has_outpoint, confidence = evaluator.evaluate_outpoint(
            "contrary_facts", test_statement, {}
        )
        end_time = time.time()
        
        print(f"✅ SUCCESS!")
        print(f"   Statement: {test_statement}")
        print(f"   Outpoint detected: {has_outpoint}")
        print(f"   Confidence: {confidence}")
        print(f"   Response time: {end_time - start_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ FAILED!")
        print(f"   Error: {str(e)}")
        print(f"   Time elapsed: {end_time - start_time:.1f} seconds")
        return False

def test_dolphin3_holistic():
    """Test holistic evaluation with dolphin3 model."""
    print("\n🔍 TESTING HOLISTIC EVALUATION")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = LLMEvaluator(
        model_name="dolphin3",
        use_rag=False,
        timeout=120
    )
    
    # Test holistic evaluation
    test_statement = "Everyone knows the CEO is embezzling funds."
    
    start_time = time.time()
    try:
        result = evaluator.evaluate_statement_holistically(test_statement, {})
        end_time = time.time()
        
        print(f"✅ SUCCESS!")
        print(f"   Statement: {test_statement}")
        print(f"   Outpoints: {result.get('outpoints', [])}")
        print(f"   Pluspoints: {result.get('pluspoints', [])}")
        print(f"   Confidence: {result.get('confidence', 0)}")
        print(f"   Response time: {end_time - start_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ FAILED!")
        print(f"   Error: {str(e)}")
        print(f"   Time elapsed: {end_time - start_time:.1f} seconds")
        return False

if __name__ == "__main__":
    print("🚀 DOLPHIN3 MODEL TEST SUITE")
    print("=" * 50)
    print("Testing TruthAlgorithm with dolphin3 model")
    print("Verifying no massive prompts are generated")
    print()
    
    # Run tests
    test1_passed = test_dolphin3_simple()
    test2_passed = test_dolphin3_holistic()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Simple evaluation: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Holistic evaluation: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Dolphin3 model is working correctly without massive prompts.")
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        print("Check the error messages above for details.")
