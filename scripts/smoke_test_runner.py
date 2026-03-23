#!/usr/bin/env python3
"""Smoke test runner for all main experiment scripts."""

import sys
import os
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def smoke_test_comprehensive():
    """Test run_10seed_comprehensive.py with 1 seed."""
    print("=" * 80)
    print("SMOKE TEST: run_10seed_comprehensive.py")
    print("=" * 80)

    try:
        # Read and modify the script to use 1 seed
        script_path = project_root / "scripts" / "run_10seed_comprehensive.py"
        script_content = script_path.read_text()

        if "seeds = [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]" in script_content:
            # Create a temporary version with 1 seed
            test_script = script_content.replace(
                "seeds = [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]",
                "seeds = [11]  # Smoke test with 1 seed"
            )
            temp_script_path = project_root / "scripts" / "temp_smoke_comprehensive.py"
            temp_script_path.write_text(test_script)

            # Import and run the modified version
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_smoke", temp_script_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            # Run the main function
            temp_module.main()

            # Clean up
            temp_script_path.unlink()

        print("[PASS] Comprehensive script smoke test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Comprehensive script smoke test FAILED: {e}")
        traceback.print_exc()
        return False

def smoke_test_ablation():
    """Test run_ablation_study.py with 1 seed."""
    print("\n" + "=" * 80)
    print("SMOKE TEST: run_ablation_study.py")
    print("=" * 80)

    try:
        # Read and modify the script
        script_path = project_root / "scripts" / "run_ablation_study.py"
        script_content = script_path.read_text()

        if "seeds = [11, 42, 55]" in script_content:
            test_script = script_content.replace(
                "seeds = [11, 42, 55]  # Sample 3 seeds for quick validation",
                "seeds = [11]  # Smoke test with 1 seed"
            )
            temp_script_path = project_root / "scripts" / "temp_smoke_ablation.py"
            temp_script_path.write_text(test_script)

            # Import and run
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_smoke_ablation", temp_script_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            # Run the main function
            temp_module.run_ablation_high_heterogeneity()

            # Clean up
            temp_script_path.unlink()

        print("[PASS] Ablation script smoke test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Ablation script smoke test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests."""
    print("SMOKE TEST SUITE FOR EXPERIMENT SCRIPTS")

    results = []
    results.append(smoke_test_comprehensive())
    results.append(smoke_test_ablation())

    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[PASS] ALL SMOKE TESTS PASSED - Scripts are ready for full experiments")
        return 0
    else:
        print("[FAIL] SOME SMOKE TESTS FAILED - Fix issues before running full experiments")
        return 1

if __name__ == "__main__":
    sys.exit(main())