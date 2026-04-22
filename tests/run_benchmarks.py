import unittest
import sys
import os

# Ensure we can import rcwa and tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_benchmarks():
    """
    Discovers and runs the system benchmarks.
    If --update is in sys.argv, the GoldenManager will handle updating the files.
    """
    print("=== RCWA Benchmark Runner ===")
    if "--update" in sys.argv:
        print("UPDATE MODE ENABLED: Golden files will be overwritten with current results.")
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests/test_system', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll benchmarks passed (or updated).")
        sys.exit(0)
    else:
        print("\nBenchmarks FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_benchmarks()
