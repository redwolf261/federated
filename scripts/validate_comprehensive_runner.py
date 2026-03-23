#!/usr/bin/env python3
"""Quick validation run of the standardized comprehensive runner with 1 seed per condition."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the comprehensive runner with modified seed list
import importlib.util

def run_validation():
    """Run validation with 1 seed per condition."""

    # Load the comprehensive runner
    runner_path = project_root / "scripts" / "standardized_comprehensive_runner.py"

    # Read the script and modify it to use just 1 seed
    script_content = runner_path.read_text()

    # Replace the seeds list with just one seed
    modified_script = script_content.replace(
        'seeds = [11, 22, 33, 42, 55, 66, 77, 88, 99, 100]',
        'seeds = [11]  # Validation run with 1 seed'
    )

    # Write temporary script
    temp_script_path = project_root / "scripts" / "temp_validation_runner.py"
    temp_script_path.write_text(modified_script)

    try:
        # Import and run
        spec = importlib.util.spec_from_file_location("temp_validation", temp_script_path)
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules["temp_validation"] = temp_module
        spec.loader.exec_module(temp_module)

        # Run the main function
        temp_module.run_comprehensive_experiments()

        print("\n[SUCCESS] Validation run completed successfully!")
        print("The standardized comprehensive runner is ready for full 10-seed experiments.")

        return True

    except Exception as e:
        print(f"\n[ERROR] Validation run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if temp_script_path.exists():
            temp_script_path.unlink()
        if "temp_validation" in sys.modules:
            del sys.modules["temp_validation"]

if __name__ == "__main__":
    print("VALIDATION RUN: Standardized Comprehensive Runner")
    print("=" * 60)
    print("Testing with 1 seed per condition (4 total runs)")
    print()

    success = run_validation()

    if success:
        print("\n[SUCCESS] Ready to proceed with full 10-seed experiments!")
        print("   - Use standardized_comprehensive_runner.py")
        print("   - Will generate bulletproof evidence for paper")
    else:
        print("\n[ERROR] Fix validation issues before proceeding to full runs")
        sys.exit(1)