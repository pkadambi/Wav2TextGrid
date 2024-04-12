#!/usr/bin/env python3
import glob
import os
import pickle as pkl
import torch
from tqdm import tqdm
import pdb
from .aligner_core.xvec_extractor import xVecExtractor
from .aligner_core.aligner import xVecSAT_forced_aligner
import argparse




def align_file(wavfilepath, transcriptfilepath, outfilepath, downsample=False, target_phns=None):
    
    if downsample==True:
        xvx.downsample(wavfilepath)

    xvector = xvx.extract_xvector(wavfilepath)
    xvector = xvector[0][0].view(1, -1)
    if torch.cuda.is_available():
        xvector = xvector.cuda()

    transcript = open(transcriptfilepath, 'r').readlines()[0]
    transcript = transcript.replace('\n', '')
    aligner.serve(audio=wavfilepath, text=transcript, save_to=outfilepath, ixvector=xvector, target_phones=target_phns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wavfile_or_dir', type=str)
    parser.add_argument('transcriptfile_or_dir', type=str)
    parser.add_argument('outfile_or_dir', default=str)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('aligner_model', type=str)
    args = parser.parse_args()

    global xvx, aligner

    xvx = xVecExtractor(method='xvector')
    aligner = xVecSAT_forced_aligner('pkadambi/Wav2TextGrid', satvector_size=512)

    if os.path.isdir(args.wavfile_or_dir):
        align_dirs(args)
    else:
        align_file(args.wavfile_or_dir, args.transcriptfile_or_dir, args.outfile_or_dir, args.downsample)



def align_dirs(args):
    
    if args.downsample==True:
        xvx.downsample(args.wavfile_or_dir)
    
    # Get list of .wav files in directory1 and its subdirectories
    wav_files = glob.glob(os.path.join(args.wavfile_or_dir, '**/*.wav'), recursive=True)
    # Get list of .lab files in directory2 and its subdirectories
    lab_files = glob.glob(os.path.join(args.transcriptfile_or_dir, '**/*.lab'), recursive=True)
    success_count = 0
    failure_count = 0
    missing_lab_files = []
    # Iterate over .wav files
    os.makedirs(args.outfile_or_dir, exist_ok=True)

    for wav_file in tqdm(wav_files):
        # Generate corresponding .lab file path
        rel_path = os.path.relpath(wav_file, args.wavfile_or_dir)
        lab_file = os.path.join(args.transcriptfile_or_dir, os.path.splitext(rel_path)[0] + '.lab')
        outfpath = os.path.join(args.outfile_or_dir, os.path.splitext(rel_path)[0]+'.TextGrid')
        # Check if .lab file exists
        if os.path.exists(lab_file):
            try:
                # Align .wav and .lab files
                align_file(wav_file, lab_file, outfpath)
                success_count += 1
            except Exception as e:
                print(f"Alignment failed for {wav_file}: {e}")
                failure_count += 1
        else:
            missing_lab_files.append(lab_file)
            print(f"Did not find transcript at {lab_file} for wav file {wav_file}")

        # Write to alignment log
    with open(os.path.join(args.outfile_or_dir, 'alignment.log'), 'w') as log_file:
        log_file.write(f"Successfully aligned: {success_count}\n")
        log_file.write(f"Alignment failures: {failure_count}\n")
        if missing_lab_files:
            log_file.write("\nMissing transcript files:\n")
            for missing_lab_file in missing_lab_files:
                log_file.write(f"- {missing_lab_file}\n")

    pass


if __name__=='__main__':
    main()