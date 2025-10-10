#!/usr/bin/env python3
"""
Inference workflow script for Wav2TextGrid
Runs inference on all wav/lab pairs in examples directory and validates outputs
"""

import argparse
import glob
import os
import platform
import subprocess
import sys
from pathlib import Path

# Add the src directory to the path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import after path modification
try:
    import nltk
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Check if the data is already available
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        print("NLTK data already available")
    except LookupError:
        print("Downloading required NLTK data...")
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("punkt", quiet=True)  # Often needed too
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"WARNING: Failed to download NLTK data: {e}")
            print("You may need to download it manually with:")
            print(">>> import nltk; nltk.download('averaged_perceptron_tagger_eng')")


def discover_example_pairs(examples_dir: str) -> list[tuple[str, str]]:
    """
    Discover all wav/lab pairs in the examples directory
    Returns list of (wav_file, lab_file) tuples
    """
    pairs = []

    # Find all wav files
    if platform.system() == "Windows":
        wav_files = [str(p) for p in Path(examples_dir).rglob("*.wav")]
    else:
        wav_files = glob.glob(os.path.join(examples_dir, "**", "*.wav"), recursive=True)

    for wav_file in wav_files:
        # Generate corresponding lab file path
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        lab_file = os.path.join(examples_dir, f"{base_name}.lab")

        if os.path.exists(lab_file):
            pairs.append((wav_file, lab_file))
        else:
            print(f"Warning: No corresponding .lab file found for {wav_file}")

    return pairs


def test_cli_interface(examples_dir: str, output_dir: str) -> bool:
    """
    Test the actual CLI interface to ensure it works correctly
    Returns True if CLI test passes, False otherwise
    """
    cli_output_dir = f"{output_dir}_cli"

    print("\n==> Testing CLI interface...")

    try:
        # Create CLI output directory
        os.makedirs(cli_output_dir, exist_ok=True)

        # Test the CLI with the same examples
        cmd = [
            sys.executable,
            "-m",
            "Wav2TextGrid.wav2textgrid",
            examples_dir,
            examples_dir,
            cli_output_dir,
        ]

        print(f"Running CLI command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"CLI test FAILED with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        # Check if CLI generated TextGrid files
        cli_textgrids = list(Path(cli_output_dir).glob("*.TextGrid"))
        if not cli_textgrids:
            print("CLI test FAILED: No TextGrid files generated")
            return False

        print(f"CLI test PASSED: Generated {len(cli_textgrids)} TextGrid files")
        for tg in cli_textgrids:
            print(f"  * {tg.name}")

        return True

    except subprocess.TimeoutExpired:
        print("CLI test FAILED: Command timed out")
        return False
    except Exception as e:
        print(f"CLI test FAILED with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Wav2TextGrid inference workflow")
    parser.add_argument(
        "--examples-dir",
        default="examples",
        help="Directory containing wav/lab pairs (default: examples)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save TextGrid outputs (default: outputs)",
    )
    parser.add_argument(
        "--no-cli-test",
        action="store_true",
        help="Skip CLI interface testing (default: test CLI)",
    )
    parser.add_argument(
        "--cli-only",
        action="store_true",
        help="Only test CLI interface, skip function-based tests",
    )

    args = parser.parse_args()

    # Ensure examples directory exists
    if not os.path.exists(args.examples_dir):
        print(f"ERROR: Examples directory not found: {args.examples_dir}")
        sys.exit(1)

    # Test only CLI interface
    print("==> Testing CLI interface only...")
    cli_passed = test_cli_interface(args.examples_dir, args.output_dir)
    if cli_passed:
        print("\nSUCCESS: CLI interface test passed!")
    else:
        print("\nERROR: CLI interface test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
