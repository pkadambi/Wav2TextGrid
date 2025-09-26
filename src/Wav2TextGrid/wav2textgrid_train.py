#!/usr/bin/env python3

'''
Author: Prad Kadambi
Paper: https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00347

'''
if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # import Wav2TextGrid  # triggers absolute import resolution


import nltk
from Wav2TextGrid.utils.args import parse_args
from Wav2TextGrid.utils.processor import load_processor
from Wav2TextGrid.utils.dataset_utils import match_audio_textgrids, create_dataset
from Wav2TextGrid.utils.training_routine import perform_train_test_split_run
from Wav2TextGrid.aligner_core.utils import get_all_filetype_in_dir


def main():
    nltk.download('averaged_perceptron_tagger_eng')
    args = parse_args()
    processor = load_processor(args.TOKENIZER_NAME)

    train_audio_files = get_all_filetype_in_dir(args.TRAIN_AUDIO_DIR, ".wav")
    train_audio_files, train_textgrids = match_audio_textgrids(train_audio_files, args.TRAIN_TEXTGRID_DIR)


    train_dataset = create_dataset(
        dataset_audiofiles=train_audio_files,
        dataset_textgrids=train_textgrids,
        args=args,
        processor=processor,
        suffix='train'
    )

    eval_dataset = None
    if args.HAS_EVAL_DATASET:
        try:
            eval_audio_files = get_all_filetype_in_dir(args.EVAL_AUDIO_DIR, ".wav")
            if len(eval_audio_files) == 0:
                print(f"\n⚠️  No audio files found in eval directory: {args.EVAL_AUDIO_DIR}")
                print("   Proceeding without evaluation dataset")
            else:
                eval_audio_files, eval_textgrids = match_audio_textgrids(eval_audio_files, args.EVAL_TEXTGRID_DIR, allow_empty=True)
                
                if len(eval_audio_files) > 0:
                    eval_dataset = create_dataset(
                        dataset_audiofiles=eval_audio_files,
                        dataset_textgrids=eval_textgrids,
                        args=args,
                        processor=processor,
                        suffix='eval'
                    )
                else:
                    print("   No matching eval pairs found. Proceeding without evaluation dataset")
        except Exception as e:
            print(f"\n⚠️  Error processing evaluation dataset: {e}")
            print("   Proceeding without evaluation dataset")
            eval_dataset = None

    perform_train_test_split_run(args=args, train_dataset=train_dataset,
                                  eval_dataset=eval_dataset, processor=processor)

if __name__ == '__main__':
    main()
