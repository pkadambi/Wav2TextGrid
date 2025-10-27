"""
Author: Prad Kadambi
Paper: https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00347

"""

import nltk

from Wav2TextGrid.aligner_core.utils import get_all_filetype_in_dir
from Wav2TextGrid.utils.args import parse_args
from Wav2TextGrid.utils.dataset_utils import create_dataset, match_audio_textgrids
from Wav2TextGrid.utils.processor import load_processor
from Wav2TextGrid.utils.training_routine import perform_train_test_split_run


def train_aligner(
    train_audio_dir,
    train_textgrid_dir,
    tokenizer_name,
    model_output_dir,
    tg_output_dir,
    model_name="pkadambi/Wav2TextGrid",
    dataset_dir="./DATASET",
    words_key="words",
    sat_method="xvec",
    device="cpu",
    ntrain_epochs=50,
    phone_key="phones",
    satvector_dctpath="satvectors.pkl",
    has_eval_dataset=False,
    eval_audio_dir=None,
    eval_textgrid_dir=None,
    write_base_alignments=False,
    baseline_tg_dir=None,
    retrain=False,
    download_nltk=True,
):
    """
    Programmatic interface for training the aligner.
    
    Args:
        train_audio_dir: Directory containing training audio files (.wav)
        train_textgrid_dir: Directory containing training TextGrid files
        tokenizer_name: Name of the tokenizer to use
        dataset_dir: Directory to store/cache the dataset
        phone_key: Key for phone tier in TextGrid
        words_key: Key for words tier in TextGrid
        sat_method: Speaker adaptation method
        model_name: Name of the model to use
        device: Device to train on (e.g., 'cuda', 'cpu')
        model_output_dir: Directory to save trained model
        ntrain_epochs: Number of training epochs
        tg_output_dir: Directory to save output TextGrids
        satvector_dctpath: Path to SAT vector DCT (optional)
        has_eval_dataset: Whether to use evaluation dataset
        eval_audio_dir: Directory containing evaluation audio files
        eval_textgrid_dir: Directory containing evaluation TextGrid files
        write_base_alignments: Whether to write baseline alignments
        baseline_tg_dir: Directory for baseline TextGrids
        retrain: Whether to retrain existing model
        download_nltk: Whether to download NLTK data
    """
    if download_nltk:
        nltk.download("averaged_perceptron_tagger_eng")
    
    processor = load_processor(tokenizer_name)

    train_audio_files = get_all_filetype_in_dir(train_audio_dir, ".wav")
    train_audio_files, train_textgrids = match_audio_textgrids(
        train_audio_files, train_textgrid_dir
    )

    train_dataset = create_dataset(
        dataset_audiofiles=train_audio_files,
        dataset_textgrids=train_textgrids,
        dataset_dir=dataset_dir,
        phone_key=phone_key,
        words_key=words_key,
        sat_method=sat_method,
        satvector_dctpath=satvector_dctpath,
        clean=False,
        processor=processor,
        suffix="train",
    )

    eval_dataset = None
    if has_eval_dataset:
        try:
            eval_audio_files = get_all_filetype_in_dir(eval_audio_dir, ".wav")
            if len(eval_audio_files) == 0:
                raise ValueError(f"No audio files found in eval directory: {eval_audio_dir}")
            
            eval_audio_files, eval_textgrids = match_audio_textgrids(
                eval_audio_files, eval_textgrid_dir, allow_empty=True
            )

            if len(eval_audio_files) == 0:
                raise ValueError("No matching eval pairs found")
            
            eval_dataset = create_dataset(
                dataset_audiofiles=eval_audio_files,
                dataset_textgrids=eval_textgrids,
                dataset_dir=dataset_dir,
                phone_key=phone_key,
                words_key=words_key,
                sat_method=sat_method,
                satvector_dctpath=satvector_dctpath,
                clean=False,
                processor=processor,
                suffix="eval",
            )
        except Exception as e:
            raise RuntimeError(f"Error processing evaluation dataset: {e}") from e

    perform_train_test_split_run(
        train_dataset=train_dataset,
        processor=processor,
        SAT_METHOD=sat_method,
        MODEL_NAME=model_name,
        DEVICE=device,
        WRITE_BASE_ALIGNMENTS=write_base_alignments,
        BASELINE_TG_DIR=baseline_tg_dir,
        MODEL_OUTPUT_DIR=model_output_dir,
        NTRAIN_EPOCHS=ntrain_epochs,
        RETRAIN=retrain,
        TG_OUTPUT_DIR=tg_output_dir,
        eval_dataset=eval_dataset,
    )


def main():
    args = parse_args()
    
    train_aligner(
        train_audio_dir=args.TRAIN_AUDIO_DIR,
        train_textgrid_dir=args.TRAIN_TEXTGRID_DIR,
        tokenizer_name=args.TOKENIZER_NAME,
        dataset_dir=args.DATASET_DIR,
        phone_key=args.PHONE_KEY,
        words_key=args.WORDS_KEY,
        sat_method=args.SAT_METHOD,
        model_name=args.MODEL_NAME,
        device=args.DEVICE,
        model_output_dir=args.MODEL_OUTPUT_DIR,
        ntrain_epochs=args.NTRAIN_EPOCHS,
        tg_output_dir=args.TG_OUTPUT_DIR,
        satvector_dctpath=args.SATVECTOR_DCTPATH,
        has_eval_dataset=args.HAS_EVAL_DATASET,
        eval_audio_dir=args.EVAL_AUDIO_DIR,
        eval_textgrid_dir=args.EVAL_TEXTGRID_DIR,
        write_base_alignments=args.WRITE_BASE_ALIGNMENTS,
        baseline_tg_dir=args.BASELINE_TG_DIR,
        retrain=args.RETRAIN,
        download_nltk=True,
    )


if __name__ == "__main__":
    # Required = argparse arguments:
    # train_audio_dir
    # train_textgrid_dir

    train_aligner(
        train_audio_dir="/Volumes/kchustad/ASUTransfer/AdultCorpus/adult_wavs_subset",
        train_textgrid_dir="/Volumes/kchustad/ASUTransfer/AdultCorpus/adult_w2tg_subset",
        tokenizer_name="charsiu/tokenizer_en_cmu",
        model_output_dir="/Users/beckett/trained_model",
        tg_output_dir="/Users/beckett/aligner_output",
    )
