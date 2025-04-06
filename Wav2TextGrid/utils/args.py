# File: utils/args.py
import argparse
import os
from dataclasses import dataclass

@dataclass
class W2TextgridTrainerArgs:
    CLEAN: bool
    TRAIN_AUDIO_DIR: str
    EVAL_AUDIO_DIR: str
    TRAIN_TEXTGRID_DIR: str
    EVAL_TEXTGRID_DIR: str
    MODEL_OUTPUT_DIR: str
    TOKENIZER_NAME: str
    MODEL_NAME: str
    BASELINE_TG_DIR: str
    TG_OUTPUT_DIR: str
    OUTPUT_FOLDER: str
    DATASET_DIR: str
    DEVICE: str
    SAT_METHOD: str
    PHONE_KEY: str
    WORDS_KEY: str
    SATVECTOR_DCTPATH: str
    SEED: int
    NTRAIN_EPOCHS: int
    WRITE_BASE_ALIGNMENTS: bool
    HAS_EVAL_DATASET: bool
    RETRAIN: bool = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_audio_dir', type=str,
                        default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')
    parser.add_argument('--train_textgrids_dir',
                        default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')
    parser.add_argument('--sat_vectors_dctfile', default='satvectors.pkl')
    parser.add_argument('--run_output_folder', default='./Wav2TextGridResults')
    parser.add_argument('--eval_audio_dir', type=str,
                        default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')
    parser.add_argument('--eval_textgrids_dir',
                        default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST', )
    parser.add_argument('--model_output_dir', default='trained_model')
    parser.add_argument('--dataset_dir', default='./data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sat_method', default='xvec')
    parser.add_argument('--phone_key', default='phones')
    parser.add_argument('--words_key', default='words')
    parser.add_argument('--ntrain_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--write_base_alignments', default=True, action='store_false')
    parser.add_argument('--clean', default=False, action='store_true')
    parser.add_argument('--tokenizer_name', default='charsiu/tokenizer_en_cmu')
    parser.add_argument('--model_name', default='pkadambi/Wav2TextGrid')
    parser.add_argument('--alignments_dir', default='alignments_baseline')

    args = parser.parse_args()

    output_folder = args.run_output_folder
    model_output_dir = os.path.join(output_folder, args.model_output_dir)
    baseline_dir = os.path.join(output_folder, args.alignments_dir, 'eval_base')
    tg_output_dir = os.path.join(output_folder, 'eval_trained')

    # os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(tg_output_dir, exist_ok=True)

    return W2TextgridTrainerArgs(
        TRAIN_AUDIO_DIR=args.train_audio_dir,
        EVAL_AUDIO_DIR=args.eval_audio_dir,
        TRAIN_TEXTGRID_DIR=args.train_textgrids_dir,
        EVAL_TEXTGRID_DIR=args.eval_textgrids_dir,
        MODEL_OUTPUT_DIR=model_output_dir,
        TOKENIZER_NAME=args.tokenizer_name,
        MODEL_NAME=args.model_name,
        BASELINE_TG_DIR=baseline_dir,
        TG_OUTPUT_DIR=tg_output_dir,
        OUTPUT_FOLDER=output_folder,
        DATASET_DIR=args.dataset_dir,
        DEVICE=args.device,
        SAT_METHOD=args.sat_method,
        PHONE_KEY=args.phone_key,
        WORDS_KEY=args.words_key,
        SATVECTOR_DCTPATH=args.sat_vectors_dctfile,
        SEED=args.seed,
        NTRAIN_EPOCHS=args.ntrain_epochs,
        WRITE_BASE_ALIGNMENTS=args.write_base_alignments,
        HAS_EVAL_DATASET=bool(args.eval_textgrids_dir),
        CLEAN=args.clean
    )
