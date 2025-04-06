import argparse
import torch
from datasets import load_from_disk
from dataclasses import dataclass
import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Trainer, TrainingArguments
from Wav2TextGrid.utils.aligner_dataset import AlignerDataset
from aligner_core.aligner import *
from aligner_core.utils import *
from aligner_core.alignermodel import  Wav2Vec2ForFrameClassificationSAT, Wav2Vec2ForFrameClassification
from typing import Dict, List, Optional, Union

#TODO: Explore if making training arguments manually usable is worth it
parser = argparse.ArgumentParser()
parser.add_argument('--train_audio_dir', type=str, default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')
parser.add_argument('--train_textgrids_dir', default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')
parser.add_argument('--sat_vectors_dctfile', default='satvectors.pkl')  # joint satvectors for both train and test

parser.add_argument('--run_output_folder', default='./Wav2TextGridResults', type=str, help='where to save models and generated TextGrids')
parser.add_argument('--eval_audio_dir', type=str, default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST')

parser.add_argument('--eval_textgrids_dir', default='/media/prad/data/datasets/ChildSpeechDataset/traintest60Kids/TEST', )
parser.add_argument('--model_output_dir', default='trained_model', help='The directory to save the model to')
parser.add_argument('--dataset_dir', default='./data', help='Will try to load the train/eval dataset from this directory. \nIf not found there, will create the trian/eval dataset and save in the dataset_dir directory')  #

parser.add_argument('--device', default='cuda')
parser.add_argument('--sat_method', default='xvec')

parser.add_argument('--phone_key', default='phones')
parser.add_argument('--words_key', default='words')

parser.add_argument('--ntrain_epochs', default=1)
parser.add_argument('--seed', default=1337)
parser.add_argument('--write_base_alignments', default=True, action='store_false')

parser.add_argument('--tokenizer_name', default='charsiu/tokenizer_en_cmu') #
parser.add_argument('--model_name', default='charsiu/en_w2v2_fc_10ms') # the seed model
parser.add_argument('--baseline_tg_dir', default='alignments_baseline') #where to save the baseline alignment results

'''
Step 0: Extract training utils
'''
from dataclasses import dataclass


@dataclass
class W2TextgridTrainerArgs:
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

    def add_arg(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)


def extract_args_from_parser(parser):
    '''
    ./RESULTS_FOLDER/
    ./RESULTS_FOLDER/

    :param parser:
    :return:
    '''

    '''    
        This is done in case we want to add additional arguments to the dataclass
    '''

    args = parser.parse_args()

    MODEL_OUTPUT_DIR = os.path.join(args.run_output_folder, args.model_output_dir)
    OUTPUT_FOLDER = args.run_output_folder
    OUTPUT_TG_DIR = os.path.join(OUTPUT_FOLDER, 'trained_alignments')
    BASELINE_RESULTS_DIR = os.path.join(OUTPUT_FOLDER, args.baseline_tg_dir)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TG_DIR, exist_ok=True)
    os.makedirs(BASELINE_RESULTS_DIR, exist_ok=True)

    script_args = W2TextgridTrainerArgs(
        TRAIN_AUDIO_DIR=args.train_audio_dir,
        EVAL_AUDIO_DIR=args.eval_audio_dir,
        TRAIN_TEXTGRID_DIR=args.train_textgrids_dir,
        EVAL_TEXTGRID_DIR=args.eval_textgrids_dir,
        HAS_EVAL_DATASET= args.eval_textgrids_dir is not None,
        MODEL_OUTPUT_DIR=MODEL_OUTPUT_DIR,
        TOKENIZER_NAME=args.tokenizer_name,
        MODEL_NAME=args.model_name,
        BASELINE_TG_DIR=BASELINE_RESULTS_DIR,
        TG_OUTPUT_DIR=OUTPUT_TG_DIR,
        OUTPUT_FOLDER=OUTPUT_FOLDER,
        DATASET_DIR=args.dataset_dir,
        DEVICE=args.device,
        SAT_METHOD=args.sat_method,
        SATVECTOR_DCTPATH=args.sat_vectors_dctfile,
        PHONE_KEY=args.phone_key,
        WORDS_KEY=args.words_key,
        SEED=args.seed,
        NTRAIN_EPOCHS=args.ntrain_epochs,
        WRITE_BASE_ALIGNMENTS=args.write_base_alignments
    )
    # # ./OUTPUT_FOLDER/BASELINE_TG_DIR  - baseline tg results output
    # # ./OUTPUT_FOLDER/RESULT_CSVPATH.csv # the result csv
    # # ./OUTPUT_FOLDER/MODEL_OUTPUT_DIR/pytorch_model.bin
    # # ./OUTPUT_FOLDER/TG_OUTPUT_DIR/

    return script_args

'''
Step 2: pass arguments to dataset class
'''
@dataclass
class DataCollatorClassificationWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    args: W2TextgridTrainerArgs
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        if args.SAT_METHOD is not None:
            input_features = [{"input_values": feature["input_values"], "ixvector": feature["ixvector"]} for feature in features]
        else:
            input_features = [{"input_values": feature["input_values"]} for feature in features]

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )

        ''' Mask out of vocabulary indices'''

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def prepare_framewise_dataset(batch, mapping, unk_method='ignore'):
    '''
    :param batch:
    :param mapping:
    :param unk_method: Labeling mode for unknown phoneme token. In mode 'ignore' the label is set to -100 (thus
                        default ignore_index for nn.CrossEntropyLoss). Or 'unk_token', when the unknown phoneme
                        labels will be set to the [UNK] token label
    :return:
    '''

    supported_unk_methods = ['ignore', 'unk_token']
    if not any([unk_method.lower() in _unkmethod for _unkmethod in supported_unk_methods]):
        raise Exception(f"Parameter 'unk_method' must be in {supported_unk_methods}")
    batch['input_values'] = batch['audio']
    batch['labels'] = []
    # for phone in batch['phone_alignments']['utterance']:
    #     batch['labels'].append(mapping[phone])
    phoneset = list(mapping.keys())
    # if statement deals with any 'sp' tokens
    batch["frame_phones"] = [phone.upper() for phone in batch["frame_phones"]]

    if 'ignore' in unk_method.lower():
        batch['labels'] = [mapping[phone.upper()] if phone in phoneset else -100
                       for phone in batch["frame_phones"]]
    elif 'unk_token' in unk_method.lower():
        batch['labels'] = [mapping[phone.upper()] if phone in phoneset else mapping["[UNK]"]
                           for phone in batch["frame_phones"]]

    return batch

def create_dataset(dataset_audiofiles, dataset_textgrids, processor, tokenizer, args:W2TextgridTrainerArgs, suffix='train'):
    model = Wav2Vec2ForFrameClassificationSAT.from_pretrained(
        args.MODEL_NAME,
        gradient_checkpointing=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer.decoder),)

    mapping_phone2id = tokenizer.encoder
    mapping_id2phone = tokenizer.decoder

    dataset_dir = args.DATASET_DIR + suffix
    os.makedirs(os.path.join(args.OUTPUT_FOLDER,suffix), exist_ok=True)
    satvector_path = os.path.join(args.OUTPUT_FOLDER,suffix, args.SATVECTOR_DCTPATH)
    _phone_key = args.phone_key
    _words_key = args.word_key

    if not os.path.exists(dataset_dir):

        ''' train dataset '''
        print(f"Dataset not found at:\t {dataset_dir} \nCreating and saving dataset")
        ald = AlignerDataset(audio_paths=dataset_audiofiles, textgrid_paths=dataset_textgrids, phone_key=args.PHONE_KEY,
                             words_key=args.WORDS_KEY, adaptation_type=args.SAT_METHOD, satvector_path=satvector_path,
                             split="train", model=model)
        dataset = ald.return_as_datsets()
        _prepare_framewise_dataset = lambda x: prepare_framewise_dataset(x, mapping=mapping_phone2id)
        dataset = dataset.map(_prepare_framewise_dataset)

        dataset.save_to_disk(dataset_dir)
    else:
        print(f"Found dataset at:\t {dataset_dir} \nLoading...")
        dataset = load_from_disk(dataset_dir)

    del model
    return dataset

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    #    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    ntimesteps = pred_ids.shape[1]
    lbs = pred.label_ids[:, :ntimesteps]
    # comparison = pred_ids.equal(pred.label_ids)
    comparison = np.equal(pred_ids, lbs)
    comparison = comparison[lbs != -100].flatten()
    acc = np.sum(comparison) / len(comparison)

    return {"phone_accuracy": acc}

def write_textgrid_alignments_for_dataset(aligner, holdout_dataset, output_dir):
    ''' performs the alignment for all files in the dataset (to the output dir)'''
    audiofiles = holdout_dataset['file']
    transcripts = holdout_dataset['sentence']
    speaker_id = holdout_dataset['speaker_id'][0]

    if 'ixvector' in holdout_dataset.features.keys():
        ixvector = holdout_dataset['ixvector']
        if type(ixvector[0]) is list:
            ixvector = [torch.Tensor(iv).to(aligner.aligner.device).reshape(1, -1) for iv in ixvector]
    else:
        ixvector = None

    print('Generating Alignments...')
    # print('Writing alignments to directory:\t', output_tg_dir)
    import time
    for ii, audiofilepath in enumerate(audiofiles):
        # file_id = holdout_dataset['id'][ii]
        output_tg_path = os.path.join(output_dir, get_filename_with_upper_dirs(audiofilepath, 2).replace('.wav', '.TextGrid'))
        output_tg_dir = os.path.dirname(output_tg_path)

        if not os.path.exists(output_tg_dir):
            os.makedirs(output_tg_dir, exist_ok=True)
        try:
            if ixvector is not None and type(aligner) == xVecSAT_forced_aligner:
                aligner.serve(audio=audiofilepath, text=transcripts[ii], save_to=output_tg_path,
                                      ixvector=ixvector[ii])
            else:
                aligner.serve(audio=audiofilepath, text=transcripts[ii], save_to=output_tg_path)
        except:
            print('Error could not generate alignment for file:', audiofilepath)

        print(f'Aligning {ii}/{len(audiofiles)}...\r', end='')
        time.sleep(.02)
def perform_train_test_split_run(args, train_dataset, processor, eval_dataset=None):
    # avoids having to load the whole dataset into memory
    data_collator = DataCollatorClassificationWithPadding(args=args, processor=processor, padding=True)

    '''Step 1: Calculate baseline accuracy (no fine tuning)'''

    if args.SAT_METHOD is not None:
        SATVECTOR_SIZE = len(train_dataset['ixvector'][0])
        model = Wav2Vec2ForFrameClassificationSAT.from_pretrained(
            args.MODEL_NAME,
            gradient_checkpointing=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.decoder),
            satvector_size=SATVECTOR_SIZE
        )
    else:
        model = Wav2Vec2ForFrameClassification.from_pretrained(
            args.MODEL_NAME,
            gradient_checkpointing=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.decoder), )

    model = model.to(args.DEVICE)
    model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
    model.config.conv_stride[-1] = 1
    model.freeze_feature_extractor()

    alignment_system = xVecSAT_forced_aligner(args.MODEL_NAME, satvector_size=SATVECTOR_SIZE)
    alignment_system.aligner = model

    if args.WRITE_BASE_ALIGNMENTS and eval_dataset is not None:
        print('Writing baseline alignments (no training)...')
        write_textgrid_alignments_for_dataset(aligner=alignment_system, holdout_dataset=eval_dataset,
                                              output_dir=args.BASELINE_TG_DIR)
    else:
        print('WARNING: Was not provided an evaluation dataset, no evaluation will be performed.')

    if eval_dataset is None:
        eval_strategy = None
    else:
        eval_strategy = 'steps'

    ''' Step 2: train model'''

    model = model.to(args.DEVICE)
    ''' --- training --- '''
    training_args = TrainingArguments(
        output_dir=args.MODEL_OUTPUT_DIR,
        gradient_checkpointing=True,
        group_by_length=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        evaluation_strategy=eval_strategy,
        num_train_epochs=args.NTRAIN_EPOCHS,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.0001,
        warmup_steps=300,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    if not os.path.exists(args.MODEL_OUTPUT_DIR) or args.RETRAIN:
        trainer.train()
        trainer.save_model(args.MODEL_OUTPUT_DIR)
        torch.save(trainer.model.state_dict(), os.path.join(args.MODEL_OUTPUT_DIR, 'pytorch_model.bin'))
        del model

    '''Evaluate trained results '''
    if args.SAT_METHOD is not None:
        SATVECTOR_SIZE = len(train_dataset['ixvector'][0])
        model = Wav2Vec2ForFrameClassificationSAT.from_pretrained(
            args.MODEL_OUTPUT_DIR,
            # gradient_checkpointing=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.decoder),
            satvector_size=SATVECTOR_SIZE
        )
        alignment_system = xVecSAT_forced_aligner(args.MODEL_NAME, satvector_size=SATVECTOR_SIZE)
    else:
        model = Wav2Vec2ForFrameClassification.from_pretrained(
            args.MODEL_OUTPUT_DIR,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.decoder),        )
        alignment_system = charsiu_forced_aligner(args.MODEL_NAME) #TODO: merge into xvec sat aligner class

    alignment_system.aligner = model.cuda()

    print('Writing trained alignments...')
    write_textgrid_alignments_for_dataset(aligner=alignment_system, holdout_dataset=eval_dataset,
                                          output_dir=args.TG_OUTPUT_DIR)
    del alignment_system


def main():
    args = extract_args_from_parser(parser)
    args.phone_key = 'phones'
    args.word_key = 'words'

    '''
    Step 1: Load tokenizer, feature_extractor, processor 
    '''
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.TOKENIZER_NAME)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    '''
    Step 3: define dataset helper functions
    '''
    # data_collator = DataCollatorClassificationWithPadding(processor=processor, padding=True)

    unmatched_manual_textgrid_files_train = get_all_filetype_in_dir(args.TRAIN_TEXTGRID_DIR, ".TextGrid")
    train_audio_files = get_all_filetype_in_dir(args.TRAIN_AUDIO_DIR, ".wav")

    manual_textgrid_files_train = []
    print('Matching textgrid and audio files...')
    for audfile in tqdm.tqdm(train_audio_files):
        targetpth = get_filename_with_upper_dirs(audfile, num_upper_dirs=1).replace('.wav', '.TextGrid')
        try:
            match = get_matching_file_in_list(targetpth, unmatched_manual_textgrid_files_train, verbose=False)
        except:
            print(f'ERROR: Did not find corresponding textgrid for audio file {audfile}.\nExpected Textgrid at path {targetpth}')
            raise Exception()

        manual_textgrid_files_train.append(match)

    train_dataset = create_dataset(dataset_audiofiles=train_audio_files, dataset_textgrids=manual_textgrid_files_train,
                                   args=args, processor=processor, tokenizer=tokenizer, suffix='train')

    if args.HAS_EVAL_DATASET:
        unmatched_manual_textgrid_files_eval = get_all_filetype_in_dir(args.EVAL_TEXTGRID_DIR, ".TextGrid")
        eval_audio_files = get_all_filetype_in_dir(args.EVAL_AUDIO_DIR, ".wav")

        manual_textgrid_files_eval = []
        for audfile in eval_audio_files:
            targetpth = get_filename_with_upper_dirs(audfile, num_upper_dirs=1).replace('.wav', '.TextGrid')
            try:
                match = get_matching_file_in_list(targetpth, unmatched_manual_textgrid_files_eval, verbose=False)
            except:
                print(targetpth)
            manual_textgrid_files_eval.append(match)

        eval_dataset = create_dataset(dataset_audiofiles=eval_audio_files, dataset_textgrids=manual_textgrid_files_train,
                                      args=args, processor=processor, tokenizer=tokenizer, suffix='evalc')
    else:
        eval_dataset = None

    perform_train_test_split_run(args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                 processor=processor)


if __name__=='__main__':
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    main()
