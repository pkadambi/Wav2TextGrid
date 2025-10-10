#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local test script for Wav2TextGrid inference
Can be run independently to test the workflow locally
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run the inference workflow locally"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Change to project root
    os.chdir(project_root)
    
    print("==> Running local Wav2TextGrid inference test")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 50)
    
    # Check if examples directory exists
    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        print(f"ERROR: Examples directory not found: {examples_dir}")
        print("Please ensure the examples directory exists with wav/lab pairs")
        sys.exit(1)
    
    # List example files
    wav_files = list(examples_dir.glob("*.wav"))
    lab_files = list(examples_dir.glob("*.lab"))
    
    print(f"Found {len(wav_files)} wav files and {len(lab_files)} lab files")
    
    if not wav_files or not lab_files:
        print("ERROR: No example files found!")
        sys.exit(1)
    
    # Run the inference workflow script
    script_path = script_dir / "run_inference_workflow.py"
    
    try:
        print("\n==> Starting inference workflow...")
        subprocess.run([
            sys.executable, str(script_path),
            "--examples-dir", str(examples_dir),
            "--output-dir", "outputs"
        ], check=True, capture_output=False)
        
        print("\nSUCCESS: Inference workflow completed successfully!")
        
        # Check outputs
        output_dir = project_root / "outputs"
        if output_dir.exists():
            textgrid_files = list(output_dir.glob("*.TextGrid"))
            print(f"Generated {len(textgrid_files)} TextGrid files:")
            for tg_file in textgrid_files:
                print(f"  * {tg_file.name}")
        
        # Check validation report
        report_file = project_root / "validation_report.txt"
        if report_file.exists():
            print(f"\nValidation report available: {report_file}")
            print("First few lines:")
            with open(report_file, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"  {line.rstrip()}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Inference workflow failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWARNING: Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()