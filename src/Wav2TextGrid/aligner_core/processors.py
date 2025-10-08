"""
All code in this file is attributed to:
https://github.com/lingjzhu/charsiu
Primary Author: lingjzhu
MIT license
"""

import re
import unicodedata
from builtins import str as unicode
from itertools import chain, groupby

import librosa.core
import numpy as np
import soundfile as sf
from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from nltk.tokenize import TweetTokenizer
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

word_tokenize = TweetTokenizer().tokenize


class CharsiuPreprocessor:

    def __init__(self):
        pass

    def get_phones_and_words(self):
        raise NotImplementedError

    def get_phone_ids(self):
        raise NotImplementedError

    def mapping_phone2id(self, phone):
        """
        Convert a phone to a numerical id

        Parameters
        ----------
        phone : str
            A phonetic symbol

        Returns
        -------
        int
            A one-hot id for the input phone

        """
        return self.processor.tokenizer.convert_tokens_to_ids(phone)

    def mapping_id2phone(self, idx):
        """
        Convert a numerical id to a phone

        Parameters
        ----------
        idx : int
            A one-hot id for a phone

        Returns
        -------
        str
            A phonetic symbol

        """

        return self.processor.tokenizer.convert_ids_to_tokens(idx)

    def audio_preprocess(self, audio, sr=16000):
        """
        Load and normalize audio
        If the sampling rate is incompatible with models, the input audio will be resampled.

        Parameters
        ----------
        path : str
            The path to the audio
        sr : int, optional
            Audio sampling rate, either 16000 or 32000. The default is 16000.

        Returns
        -------
        torch.Tensor [(n,)]
            A list of audio sample as an one dimensional torch tensor

        """
        if type(audio) == str:
            if sr == 16000:
                features, fs = sf.read(audio)
                # Resample to 16000 Hz if the sample rate is different
                if fs != 16000:
                    features = librosa.core.resample(features, orig_sr=fs, target_sr=16000)
            else:
                features, _ = librosa.core.load(audio, sr=sr)
        elif isinstance(audio, np.ndarray):
            features = audio
        else:
            raise Exception("The input must be a path or a numpy array!")
        return self.processor(
            features, sampling_rate=16000, return_tensors="pt"
        ).input_values.squeeze()


"""
English g2p processor
"""


class CharsiuPreprocessor_en(CharsiuPreprocessor):

    def __init__(self):

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("charsiu/tokenizer_en_cmu")
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2p()
        self.sil = "[SIL]"
        self.sil_idx = self.mapping_phone2id(self.sil)
        #        self.punctuation = set('.,!?')
        self.punctuation = set()

    def get_phones_and_words(self, sen):
        """
        Convert texts to phone sequence

        Parameters
        ----------
        sen : str
            A str of input sentence

        Returns
        -------
        sen_clean : list
            A list of phone sequence without stress marks
        sen : list
             A list of phone sequence with stress marks


        xxxxx should sen_clean be deleted?

        """

        phones = self.g2p(sen)
        words = self._get_words(sen)

        phones = list(tuple(g) for k, g in groupby(phones, key=lambda x: x != " ") if k)

        aligned_phones = []
        aligned_words = []
        for p, w in zip(phones, words):
            if re.search(r"\w+\d?", p[0]):
                aligned_phones.append(p)
                aligned_words.append(w)
            elif p in self.punctuation:
                aligned_phones.append((self.sil,))
                aligned_words.append(self.sil)

        assert len(aligned_words) == len(aligned_phones)

        return aligned_phones, aligned_words

        assert len(words) == len(phones)

        return phones, words

    def get_phone_ids(self, phones, append_silence=True):
        """
        Convert phone sequence to ids

        Parameters
        ----------
        phones : list
            A list of phone sequence
        append_silence : bool, optional
            Whether silence is appended at the beginning and the end of the sequence.
            The default is True.

        Returns
        -------
        ids: list
            A list of one-hot representations of phones

        """
        phones = list(chain.from_iterable(phones))
        ids = [self.mapping_phone2id(re.sub(r"\d", "", p)) for p in phones]

        # append silence at the beginning and the end
        if append_silence:
            if ids[0] != self.sil_idx:
                ids = [self.sil_idx] + ids
            if ids[-1] != self.sil_idx:
                ids.append(self.sil_idx)
        return ids

    def _get_words(self, text):
        """
        from G2P_en
        https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

        Parameters
        ----------
        sen : TYPE
            DESCRIPTION.

        Returns
        -------
        words : TYPE
            DESCRIPTION.

        """

        text = unicode(text)
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)

        return words

    def align_words(self, preds, phones, words):

        words_rep = [w for ph, w in zip(phones, words) for p in ph]
        phones_rep = [re.sub(r"\d", "", p) for ph, w in zip(phones, words) for p in ph]
        assert len(words_rep) == len(phones_rep)

        # match each phone to its word
        word_dur = []
        count = 0
        for dur in preds:
            if dur[-1] == "[SIL]":
                word_dur.append((dur, "[SIL]"))
            else:
                while dur[-1] != phones_rep[count]:
                    count += 1
                word_dur.append((dur, words_rep[count]))  # ((start,end,phone),word)

        # merge phone-to-word alignment to derive word duration
        words = []
        for key, group in groupby(word_dur, lambda x: x[-1]):
            group = list(group)
            entry = (group[0][0][0], group[-1][0][1], key)
            words.append(entry)

        return words
