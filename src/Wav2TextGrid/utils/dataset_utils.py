# File: utils/dataset_utils.py
import os

from datasets import load_from_disk

from Wav2TextGrid.aligner_core.alignermodel import (
    Wav2Vec2ForFrameClassification,
)
from Wav2TextGrid.aligner_core.utils import (
    get_all_filetype_in_dir,
    get_filename_with_upper_dirs,
    get_matching_file_in_list,
)

from .aligner_dataset import AlignerDataset


def match_audio_textgrids(audio_files, textgrid_dir, allow_empty=False):
    import tqdm

    tg_files = get_all_filetype_in_dir(textgrid_dir, ".TextGrid")
    matched_audio = []
    matched_textgrids = []
    skipped_count = 0

    for audio in tqdm.tqdm(audio_files, desc="Matching TextGrids"):
        target = get_filename_with_upper_dirs(audio, 1).replace(".wav", ".TextGrid")
        try:

            matched_tg = get_matching_file_in_list(target, tg_files)
            matched_audio.append(audio)
            matched_textgrids.append(matched_tg)
        except Exception:
            skipped_count += 1
            # Print warning for first few files, then summarize
            if skipped_count <= 5:
                print(f"\nWarning: Skipping {os.path.basename(audio)} - no matching TextGrid")
            elif skipped_count == 6:
                print("\n... (suppressing further individual warnings)")
            continue

    # Summary statistics
    success_rate = len(matched_audio) / len(audio_files) * 100 if audio_files else 0
    print("\n✅ Dataset Matching Summary:")
    print(f"   • Successfully matched: {len(matched_audio)} pairs")
    print(f"   • Skipped (no TextGrid): {skipped_count} files")
    print(f"   • Success rate: {success_rate:.1f}%")

    if len(matched_audio) == 0 and not allow_empty:
        raise ValueError(
            "No matching audio-TextGrid pairs found! Check your file paths and naming conventions."
        )

    return matched_audio, matched_textgrids


def prepare_framewise_dataset(batch, mapping, unk_method="ignore"):
    # phoneset = list(mapping.keys())
    batch["input_values"] = batch["audio"]
    batch["frame_phones"] = [phone.upper() for phone in batch["frame_phones"]]

    if "ignore" in unk_method.lower():
        batch["labels"] = [mapping.get(phone.upper(), -100) for phone in batch["frame_phones"]]
    elif "unk_token" in unk_method.lower():
        batch["labels"] = [
            mapping.get(phone.upper(), mapping["[UNK]"]) for phone in batch["frame_phones"]
        ]
    return batch


def create_dataset(dataset_audiofiles, dataset_textgrids, args, processor, suffix="train"):
    tokenizer = processor.tokenizer
    # model = Wav2Vec2ForFrameClassificationSAT.from_pretrained(
    #     args.MODEL_NAME,
    model = Wav2Vec2ForFrameClassification.from_pretrained(
        "charsiu/en_w2v2_fc_10ms",
        # gradient_checkpointing=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer.decoder),
    )
    os.getcwd()
    dataset_dir = os.path.abspath(args.DATASET_DIR)
    dataset_dir = os.path.join(dataset_dir, suffix)
    os.makedirs(dataset_dir, exist_ok=True)
    satvector_path = os.path.join(dataset_dir, args.SATVECTOR_DCTPATH)
    DS_DIR_IS_EMPTY = not os.listdir(dataset_dir)

    if DS_DIR_IS_EMPTY or args.CLEAN:
        ald = AlignerDataset(
            args=args,
            audio_paths=dataset_audiofiles,
            textgrid_paths=dataset_textgrids,
            phone_key=args.PHONE_KEY,
            words_key=args.WORDS_KEY,
            adaptation_type=args.SAT_METHOD,
            satvector_path=satvector_path,
            model=model,
        )
        dataset = ald.return_as_datsets()
        dataset = dataset.map(lambda x: prepare_framewise_dataset(x, mapping=tokenizer.encoder))
        dataset.save_to_disk(dataset_dir)
    else:
        dataset = load_from_disk(dataset_dir)

    del model
    return dataset
