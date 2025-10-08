# File: utils/data_collator.py
import torch
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from ..utils.args import W2TextgridTrainerArgs


class DataCollatorClassificationWithPadding:
    def __init__(
        self,
        args: W2TextgridTrainerArgs,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        return_attention_mask: Optional[bool] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.args = args
        self.processor = processor
        self.padding = padding
        self.return_attention_mask = return_attention_mask
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if self.args.SAT_METHOD is not None:
            input_features = [
                {"input_values": f["input_values"], "ixvector": f["ixvector"]} for f in features
            ]
        else:
            input_features = [{"input_values": f["input_values"]} for f in features]

        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
