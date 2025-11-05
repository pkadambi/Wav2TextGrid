"""
Author: Prad Kadambi
Paper: https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00347

"""

import argparse
import glob
import os
import platform
from pathlib import Path

import torch
from tqdm import tqdm

from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner
from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor


def align_file(
    wavfilepath,
    transcriptfilepath,
    outfilepath,
    xvec_extractor=None,
    forced_aligner=None,
    target_phns=None,
    postproc=False,
    use_speaker_adaptation=True,
):
    # Only extract and pass an x-vector when speaker adaptation is enabled
    ixvector = None
    if use_speaker_adaptation and xvec_extractor is not None:
        xvector = xvec_extractor.extract_xvector(wavfilepath)
        xvector = xvector[0][0].view(1, -1)
        if torch.cuda.is_available():
            xvector = xvector.cuda()
        ixvector = xvector

    transcript = open(transcriptfilepath).readlines()[0]
    transcript = transcript.replace("\n", "")
    forced_aligner.serve(
        audio=wavfilepath,
        text=transcript,
        save_to=outfilepath,
        ixvector=ixvector,
        target_phones=target_phns,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavfile_or_dir", type=str)
    parser.add_argument("transcriptfile_or_dir", type=str)
    parser.add_argument("outfile_or_dir", type=str)
    parser.add_argument("--filetype", default="wav")
    parser.add_argument("--aligner_model", type=str, default="pkadambi/Wav2TextGrid")
    parser.add_argument(
        "--disable-speaker-adaptation",
        action="store_true",
        help="Disable x-vector speaker adaptation (do not extract or pass x-vectors).",
    )
    args = parser.parse_args()

    use_speaker_adaptation = not args.disable_speaker_adaptation

    # Only create an xVecExtractor if speaker adaptation is enabled
    xvx = xVecExtractor(method="xvector") if use_speaker_adaptation else None

    if os.path.isdir(args.wavfile_or_dir):
        align_dirs(
            args.wavfile_or_dir,
            args.transcriptfile_or_dir,
            args.outfile_or_dir,
            xvx,
            args.aligner_model,
            args.filetype,
            use_speaker_adaptation=use_speaker_adaptation,
        )
    else:
        # satvector_size 0 disables adaptation in the aligner construction
        sat_size = 512 if use_speaker_adaptation else 0
        aligner = xVecSAT_forced_aligner(args.aligner_model, satvector_size=sat_size)
        align_file(
            args.wavfile_or_dir,
            args.transcriptfile_or_dir,
            args.outfile_or_dir,
            xvec_extractor=xvx,
            forced_aligner=aligner,
            use_speaker_adaptation=use_speaker_adaptation,
        )


def align_dirs(
    wavfile_or_dir,
    transcriptfile_or_dir,
    outfile_or_dir,
    xvx=None,
    aligner_model=None,
    filetype="wav",
    postproc=False,
    use_speaker_adaptation=True,
):
    # TODO: Remove redundancy with main() in terms of parameter passing
    if xvx is None and use_speaker_adaptation:
        xvx = xVecExtractor(method="xvector")
    if aligner_model is None:
        aligner_model = "pkadambi/Wav2TextGrid"

    sat_size = 512 if use_speaker_adaptation else 0
    aligner = xVecSAT_forced_aligner(aligner_model, satvector_size=sat_size)

    if platform.system() == "Windows":
        # Use pathlib for Windows, especially with UNC paths
        wav_files = [str(p) for p in Path(wavfile_or_dir).rglob(f"*.{filetype}")]
    else:
        wav_files = glob.glob(
            os.path.join(wavfile_or_dir, "**", f"*.{filetype}"),
            recursive=True,
        )

    success_count = 0
    failure_count = 0
    missing_lab_files = []
    # Iterate over .wav files
    os.makedirs(outfile_or_dir, exist_ok=True)

    for wav_file in tqdm(wav_files):
        # Generate corresponding .lab file path
        rel_path = os.path.relpath(wav_file, wavfile_or_dir)
        lab_file = os.path.join(transcriptfile_or_dir, os.path.splitext(rel_path)[0] + ".lab")
        outfpath = os.path.join(outfile_or_dir, os.path.splitext(rel_path)[0] + ".TextGrid")

        # Check if .lab file exists
        if os.path.exists(lab_file):
            try:
                # Align .wav and .lab files
                align_file(
                    wav_file,
                    lab_file,
                    outfpath,
                    xvx,
                    aligner,
                    postproc=postproc,
                    use_speaker_adaptation=use_speaker_adaptation,
                )  # always avoid downsampling because it occurs earlier
                success_count += 1
            except Exception as e:
                print(f"Alignment failed for {wav_file}: {e}")
                failure_count += 1
        else:
            missing_lab_files.append(lab_file)
            print(f"Did not find transcript at {lab_file} for wav file {wav_file}")

    # Write to alignment log
    with open(os.path.join(outfile_or_dir, "alignment.log"), "w") as log_file:
        log_file.write(f"Successfully aligned: {success_count}\n")
        log_file.write(f"Alignment failures: {failure_count}\n")

        if missing_lab_files:
            log_file.write("\nMissing transcript .lab files:\n")
            for missing_lab_file in missing_lab_files:
                log_file.write(f"- {missing_lab_file}\n")


if __name__ == "__main__":
    main()