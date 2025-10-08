# File: utils/processor.py
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def load_processor(tokenizer_name: str) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_name)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
