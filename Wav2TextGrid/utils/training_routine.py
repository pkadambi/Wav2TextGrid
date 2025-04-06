import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from Wav2TextGrid.aligner_core.alignermodel import Wav2Vec2ForFrameClassificationSAT, Wav2Vec2ForFrameClassification
from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner, charsiu_forced_aligner
from Wav2TextGrid.aligner_core.utils import get_filename_with_upper_dirs


def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids[:, :pred_ids.shape[1]]
    mask = label_ids != -100
    correct = np.sum((pred_ids == label_ids) & mask)
    total = np.sum(mask)
    return {"phone_accuracy": correct / total if total > 0 else 0.0}


def write_textgrid_alignments_for_dataset(aligner, dataset, output_dir):
    import time
    audiofiles = dataset['file']
    transcripts = dataset['sentence']
    ixvector = dataset['ixvector'] if 'ixvector' in dataset.features else None

    for ii, audiofile in enumerate(audiofiles):
        output_path = os.path.join(output_dir, get_filename_with_upper_dirs(audiofile, 2).replace('.wav', '.TextGrid'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            if ixvector is not None and isinstance(aligner, xVecSAT_forced_aligner):
                vec = torch.Tensor(ixvector[ii]).reshape(1, -1).to(aligner.aligner.device)
                aligner.serve(audio=audiofile, text=transcripts[ii], save_to=output_path, ixvector=vec)
            else:
                aligner.serve(audio=audiofile, text=transcripts[ii], save_to=output_path)
        except:
            print("Error aligning:", audiofile)
            try:
                aligner.serve(audio=audiofile, text=transcripts[ii], save_to=output_path, ixvector=vec)
            except:
                print('Asdf')


        print(f'Aligning {ii+1}/{len(audiofiles)}...', end='\r')
        time.sleep(0.02)


def perform_train_test_split_run(args, train_dataset, processor, eval_dataset=None):
    from Wav2TextGrid.utils.data_collator import DataCollatorClassificationWithPadding

    data_collator = DataCollatorClassificationWithPadding(args=args, processor=processor)
    tokenizer = processor.tokenizer

    sat_size = 512 if args.SAT_METHOD else None
    model_cls = Wav2Vec2ForFrameClassificationSAT if sat_size else Wav2Vec2ForFrameClassification

    model = model_cls.from_pretrained(
        args.MODEL_NAME,
        # gradient_checkpointing=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer.decoder),
        **({'satvector_size': sat_size} if sat_size else {}))

    model = model.to(args.DEVICE)
    model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
    model.config.conv_stride[-1] = 1
    model.freeze_feature_extractor()

    aligner_cls = xVecSAT_forced_aligner if sat_size else charsiu_forced_aligner
    aligner = aligner_cls(args.MODEL_NAME, satvector_size=sat_size) if sat_size else aligner_cls(args.MODEL_NAME)
    aligner.aligner = model

    if args.WRITE_BASE_ALIGNMENTS and eval_dataset:
        write_textgrid_alignments_for_dataset(aligner=aligner, dataset=eval_dataset, output_dir=args.BASELINE_TG_DIR)
    else:
        print('No eval dataset found, no baseline alignments to write')

    eval_strategy = 'steps' if eval_dataset else 'no'
    warmup_steps = int(len(train_dataset)/2) if len(train_dataset)/64<600 else 300

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
        save_steps=-1,
        eval_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.0001,
        warmup_steps=warmup_steps,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor
    )

    MODEL_DIR_IS_EMPTY = not os.listdir(args.MODEL_OUTPUT_DIR)
    if MODEL_DIR_IS_EMPTY or args.RETRAIN:
        print('***************************************************************')
        print('***************************************************************')
        print('\t\t\t BEGAN TRAINING \t\t\t')
        print('***************************************************************')
        print('***************************************************************')
        trainer.train()
        trainer.save_model(args.MODEL_OUTPUT_DIR)
        torch.save(trainer.model.state_dict(), os.path.join(args.MODEL_OUTPUT_DIR, 'pytorch_model.bin'))
        del model

    # Reload trained model and write alignments
    model = model_cls.from_pretrained(
        args.MODEL_OUTPUT_DIR,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer.decoder),
        **({'satvector_size': sat_size} if sat_size else {})
    ).to(args.DEVICE)

    aligner.aligner = model
    write_textgrid_alignments_for_dataset(aligner, eval_dataset, args.TG_OUTPUT_DIR)
