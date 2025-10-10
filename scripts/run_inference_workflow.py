#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference workflow script for Wav2TextGrid
Runs inference on all wav/lab pairs in examples directory and validates outputs
"""

import os
import sys
import glob
import platform
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Add the src directory to the path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import after path modification
try:
    from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner
    from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor
    import torch
    import nltk
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Check if the data is already available
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        print("NLTK data already available")
    except LookupError:
        print("Downloading required NLTK data...")
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('punkt', quiet=True)  # Often needed too
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"WARNING: Failed to download NLTK data: {e}")
            print("You may need to download it manually with:")
            print(">>> import nltk; nltk.download('averaged_perceptron_tagger_eng')")


def align_file_with_models(wavfilepath, transcriptfilepath, outfilepath, xvx, aligner, target_phns=None):
    """
    Modified version of align_file that takes models as parameters instead of using globals
    """
    xvector = xvx.extract_xvector(wavfilepath)
    xvector = xvector[0][0].view(1, -1)
    if torch.cuda.is_available():
        xvector = xvector.cuda()

    transcript = open(transcriptfilepath, "r").readlines()[0]
    transcript = transcript.replace("\n", "")
    aligner.serve(
        audio=wavfilepath,
        text=transcript,
        save_to=outfilepath,
        ixvector=xvector,
        target_phones=target_phns,
    )


def discover_example_pairs(examples_dir: str) -> List[Tuple[str, str]]:
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


def validate_textgrid(textgrid_path: str) -> Dict[str, any]:
    """
    Validate a TextGrid file to ensure it contains meaningful predictions
    Returns a dictionary with validation results
    """
    validation_result = {
        "file": textgrid_path,
        "exists": False,
        "has_content": False,
        "has_intervals": False,
        "num_tiers": 0,
        "num_intervals": 0,
        "total_duration": 0.0,
        "non_empty_intervals": 0,
        "errors": []
    }
    
    try:
        import praatio
        
        if not os.path.exists(textgrid_path):
            validation_result["errors"].append("TextGrid file does not exist")
            return validation_result
        
        validation_result["exists"] = True
        
        # Try to load the TextGrid
        try:
            tg = praatio.textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)
            validation_result["has_content"] = True
            validation_result["num_tiers"] = len(tg.tierNames)
            
            if validation_result["num_tiers"] > 0:
                validation_result["has_intervals"] = True
                
                # Get statistics from the first tier
                first_tier_name = tg.tierNames[0]
                first_tier = tg.getTier(first_tier_name)
                validation_result["num_intervals"] = len(first_tier.entries)
                
                if validation_result["num_intervals"] > 0:
                    validation_result["total_duration"] = first_tier.maxTimestamp - first_tier.minTimestamp
                    
                    # Count non-empty intervals
                    non_empty = 0
                    for entry in first_tier.entries:
                        if hasattr(entry, 'label') and entry.label.strip():
                            non_empty += 1
                        elif len(entry) >= 3 and entry[2].strip():  # Fallback for tuple format
                            non_empty += 1
                    validation_result["non_empty_intervals"] = non_empty
                    
                    if non_empty == 0:
                        validation_result["errors"].append("No non-empty intervals found")
                else:
                    validation_result["errors"].append("No intervals found in TextGrid")
            else:
                validation_result["errors"].append("No tiers found in TextGrid")
                
        except Exception as e:
            validation_result["errors"].append(f"Error loading TextGrid: {str(e)}")
            
    except ImportError:
        validation_result["errors"].append("praatio library not available for validation")
    except Exception as e:
        validation_result["errors"].append(f"Unexpected error: {str(e)}")
    
    return validation_result


def run_inference_workflow(examples_dir: str = "examples", output_dir: str = "outputs"):
    """
    Run the complete inference workflow
    """
    print("==> Starting Wav2TextGrid inference workflow")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Examples directory: {examples_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover example pairs
    pairs = discover_example_pairs(examples_dir)
    print(f"Found {len(pairs)} wav/lab pairs:")
    for wav, lab in pairs:
        print(f"  * {os.path.basename(wav)} + {os.path.basename(lab)}")
    
    if not pairs:
        print("ERROR: No wav/lab pairs found!")
        sys.exit(1)
    
    print("\nInitializing models...")
    
    # Download NLTK data if needed
    download_nltk_data()
    
    try:
        # Initialize models
        xvx = xVecExtractor(method="xvector")
        aligner = xVecSAT_forced_aligner("pkadambi/Wav2TextGrid", satvector_size=512)
        print("Models initialized successfully")
    except Exception as e:
        print(f"ERROR: Error initializing models: {e}")
        sys.exit(1)
    
    print("\nRunning inference...")
    results = []
    successful_alignments = 0
    failed_alignments = 0
    
    for i, (wav_file, lab_file) in enumerate(pairs, 1):
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.TextGrid")
        
        print(f"[{i}/{len(pairs)}] Processing {base_name}...")
        
        try:
            # Run alignment with models passed as parameters
            align_file_with_models(wav_file, lab_file, output_file, xvx, aligner)
            successful_alignments += 1
            print(f"  SUCCESS: Alignment successful: {output_file}")
            
            # Validate output
            validation = validate_textgrid(output_file)
            results.append(validation)
            
            if validation["errors"]:
                print(f"  WARNING: Validation warnings: {'; '.join(validation['errors'])}")
            else:
                print(f"  PASS: Validation passed: {validation['num_intervals']} intervals, "
                      f"{validation['non_empty_intervals']} non-empty, "
                      f"{validation['total_duration']:.2f}s duration")
                
        except Exception as e:
            failed_alignments += 1
            print(f"  ERROR: Alignment failed: {e}")
            
            # Add failed validation result
            validation = {
                "file": output_file,
                "exists": False,
                "has_content": False,
                "has_intervals": False,
                "num_tiers": 0,
                "num_intervals": 0,
                "total_duration": 0.0,
                "non_empty_intervals": 0,
                "errors": [f"Alignment failed: {str(e)}"]
            }
            results.append(validation)
    
    # Generate validation report
    print("\nGenerating validation report...")
    report_path = "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("Wav2TextGrid Inference Validation Report\n")
        f.write("========================================\n\n")
        f.write(f"Platform: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Total pairs processed: {len(pairs)}\n")
        f.write(f"Successful alignments: {successful_alignments}\n")
        f.write(f"Failed alignments: {failed_alignments}\n")
        f.write(f"Success rate: {(successful_alignments/len(pairs)*100):.1f}%\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"\nFile: {os.path.basename(result['file'])}\n")
            f.write(f"  Exists: {result['exists']}\n")
            f.write(f"  Has Content: {result['has_content']}\n")
            f.write(f"  Tiers: {result['num_tiers']}\n")
            f.write(f"  Intervals: {result['num_intervals']}\n")
            f.write(f"  Non-empty intervals: {result['non_empty_intervals']}\n")
            f.write(f"  Duration: {result['total_duration']:.2f}s\n")
            if result['errors']:
                f.write(f"  Errors: {'; '.join(result['errors'])}\n")
            f.write(f"  Status: {'PASS' if not result['errors'] else 'FAIL'}\n")
    
    print(f"Validation report saved to: {report_path}")
    
    # Summary
    print("\nSummary:")
    print(f"  Successful alignments: {successful_alignments}/{len(pairs)}")
    print(f"  Failed alignments: {failed_alignments}/{len(pairs)}")
    print(f"  Success rate: {(successful_alignments/len(pairs)*100):.1f}%")
    
    valid_outputs = sum(1 for r in results if not r['errors'])
    print(f"  Valid outputs: {valid_outputs}/{len(pairs)}")
    print(f"  Validation rate: {(valid_outputs/len(pairs)*100):.1f}%")
    
    if failed_alignments > 0:
        print("\nERROR: Some alignments failed. Check the validation report for details.")
        sys.exit(1)
    elif valid_outputs < len(pairs):
        print("\nWARNING: Some outputs failed validation. Check the validation report for details.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All alignments completed successfully and passed validation!")


def main():
    parser = argparse.ArgumentParser(description="Run Wav2TextGrid inference workflow")
    parser.add_argument("--examples-dir", default="examples", 
                       help="Directory containing wav/lab pairs (default: examples)")
    parser.add_argument("--output-dir", default="outputs",
                       help="Directory to save TextGrid outputs (default: outputs)")
    
    args = parser.parse_args()
    
    # Ensure examples directory exists
    if not os.path.exists(args.examples_dir):
        print(f"ERROR: Examples directory not found: {args.examples_dir}")
        sys.exit(1)
    
    run_inference_workflow(args.examples_dir, args.output_dir)


if __name__ == "__main__":
    main()