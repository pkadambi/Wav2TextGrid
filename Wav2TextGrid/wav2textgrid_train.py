#!/usr/bin/env python3

'''
Author: Prad Kadambi
Paper: https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00347

'''
import nltk
from .utils.args import parse_args
from .utils.processor import load_processor
from .utils.dataset_utils import match_audio_textgrids, create_dataset
from .utils.training_routine import perform_train_test_split_run
from .aligner_core.utils import get_all_filetype_in_dir
import tqdm


def main():
    nltk.download('averaged_perceptron_tagger_eng')
    args = parse_args()
    processor = load_processor(args.TOKENIZER_NAME)

    train_audio_files = get_all_filetype_in_dir(args.TRAIN_AUDIO_DIR, ".wav")
    train_textgrids = match_audio_textgrids(train_audio_files, args.TRAIN_TEXTGRID_DIR)


    train_dataset = create_dataset(
        dataset_audiofiles=train_audio_files,
        dataset_textgrids=train_textgrids,
        args=args,
        processor=processor,
        suffix='train'
    )

    eval_dataset = None
    if args.HAS_EVAL_DATASET:
        eval_audio_files = get_all_filetype_in_dir(args.EVAL_AUDIO_DIR, ".wav")
        eval_textgrids = match_audio_textgrids(eval_audio_files, args.EVAL_TEXTGRID_DIR)

        eval_dataset = create_dataset(
            dataset_audiofiles=eval_audio_files,
            dataset_textgrids=eval_textgrids,
            args=args,
            processor=processor,
            suffix='eval'
        )

    perform_train_test_split_run(args=args, train_dataset=train_dataset,
                                  eval_dataset=eval_dataset, processor=processor)


if __name__ == '__main__':
    main()
