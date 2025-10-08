"""
The code in this file has been adapted from:
https://github.com/lingjzhu/charsiu
Authors: lingjzhu, pkadambi
MIT license
"""

import sys
from itertools import groupby

import torch

sys.path.append("src/")
import numpy as np

# sys.path.insert(0,'src')
from .alignermodel import Wav2Vec2ForFrameClassification, Wav2Vec2ForFrameClassificationSAT
from .processors import CharsiuPreprocessor_en
from .utils import duration2textgrid, forced_align, seq2duration, word2textgrid


class base_aligner:

    def __init__(
        self,
        lang="en",
        sampling_rate=16000,
        device=None,
        recognizer=None,
        processor=None,
        resolution=0.01,
    ):

        self.lang = lang

        if processor is not None:
            self.processor = processor
        else:
            self.base_processor = CharsiuPreprocessor_en()

        self.resolution = resolution

        self.sr = sampling_rate

        self.recognizer = recognizer

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _freeze_model(self):
        self.aligner.eval().to(self.device)
        if self.recognizer is not None:
            self.recognizer.eval().to(self.device)

    def align(self, audio, text):
        print("WHAT?")
        raise NotImplementedError()

    def serve(self, audio, save_to, output_format="variable", text=None):
        raise NotImplementedError()

    def _to_textgrid(self, phones, save_to):
        """
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        duration2textgrid(phones, save_path=save_to)
        # print('Alignment output has been saved to %s' % (save_to))

    def _to_tsv(self, phones, save_to):
        """
        Convert output tuples to a tab-separated file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        with open(save_to, "w") as f:
            for start, end, phone in phones:
                f.write("%s\t%s\t%s\n" % (start, end, phone))
        # print('Alignment output has been saved to %s' % (save_to))


class xVecSAT_forced_aligner(base_aligner):
    def __init__(self, aligner, satvector_size, sil_threshold=4, **kwargs):
        super(xVecSAT_forced_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassificationSAT.from_pretrained(
            aligner, satvector_size=satvector_size
        )
        self.sil_threshold = sil_threshold
        self._freeze_model()

    def align(self, audio, text, ixvector, target_phones=None, return_logits=False, TEMPERATURE=1):
        """
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        """
        audio = self.base_processor.audio_preprocess(audio, sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        phones, words = self.base_processor.get_phones_and_words(text)

        if target_phones is not None:
            try:
                if not (np.array(phones) == np.array(target_phones)).all():
                    print("target")
                    print(target_phones)
                    print("g2p")
                    print(phones)
            except:
                print("text")
                print(text)
                print("target")
                print(target_phones)
                print("g2p")
                phnsg2p = [item for t in phones for item in t]
                print(phnsg2p)
            phones = target_phones

        phone_ids = self.base_processor.get_phone_ids(phones)

        with torch.no_grad():
            out = self.aligner(input_values=audio, ixvector=ixvector)
        cost = torch.softmax(out.logits, dim=-1).detach().cpu().numpy().squeeze()

        sil_mask = self._get_sil_mask(cost)

        nonsil_idx = np.argwhere(sil_mask != self.base_processor.sil_idx).squeeze()
        if nonsil_idx is None:
            raise Exception("No speech detected! Please check the audio file!")

        aligned_phone_ids = forced_align(cost[nonsil_idx, :], phone_ids[1:-1])

        aligned_phones = [
            self.base_processor.mapping_id2phone(phone_ids[1:-1][i]) for i in aligned_phone_ids
        ]

        pred_phones = self._merge_silence(aligned_phones, sil_mask)

        pred_phones = seq2duration(pred_phones, resolution=self.resolution)

        pred_words = self.base_processor.align_words(pred_phones, phones, words)

        return pred_phones, pred_words

    def serve(
        self,
        audio,
        text,
        ixvector,
        save_to,
        target_phones=None,
        output_format="textgrid",
        verbose=False,
    ):
        """
         A wrapper function for quick inference

        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file.
            The default is 'textgrid'.

        Returns
        -------
        None.

        """
        phones, words = self.align(audio, text, ixvector, target_phones=target_phones)

        if output_format == "tsv":
            if save_to.endswith(".tsv"):
                save_to_phone = save_to.replace(".tsv", "_phone.tsv")
                save_to_word = save_to.replace(".tsv", "_word.tsv")
            else:
                save_to_phone = save_to + "_phone.tsv"
                save_to_word = save_to + "_word.tsv"

            self._to_tsv(phones, save_to_phone)
            self._to_tsv(words, save_to_word)

        elif output_format == "textgrid":
            self._to_textgrid(phones, words, save_to)
        else:
            raise Exception("Please specify the correct output format (tsv or textgrid)!")

    def _to_textgrid(self, phones, words, save_to):
        """
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        word2textgrid(phones, words, save_path=save_to)
        # print('Alignment output has been saved to %s'%(save_to))

    def _get_sil_mask(self, cost):
        # single out silent intervals

        preds = np.argmax(cost, axis=-1)
        sil_mask = []
        for key, group in groupby(preds):
            group = list(group)
            if key == self.base_processor.sil_idx and len(group) < self.sil_threshold:
                sil_mask += [-1 for i in range(len(group))]
            else:
                sil_mask += group

        return np.array(sil_mask)

    def _merge_silence(self, aligned_phones, sil_mask):
        # merge silent and non-silent intervals
        pred_phones = []
        count = 0
        for i in sil_mask:
            if i == self.base_processor.sil_idx:
                pred_phones.append("[SIL]")
            else:
                pred_phones.append(aligned_phones[count])
                count += 1
        assert len(pred_phones) == len(sil_mask)
        return pred_phones


class charsiu_forced_aligner(base_aligner):

    def __init__(self, aligner, sil_threshold=4, **kwargs):
        super(charsiu_forced_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)
        self.sil_threshold = sil_threshold

        self._freeze_model()

    def align(self, audio, text, target_phones=None, return_logits=False, TEMPERATURE=1):
        """
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        """
        audio = self.charsiu_processor.audio_preprocess(audio, sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        phones, words = self.charsiu_processor.get_phones_and_words(text)

        if target_phones is not None:
            phones = target_phones

        phone_ids = self.charsiu_processor.get_phone_ids(phones)

        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits / TEMPERATURE, dim=-1).detach().cpu().numpy().squeeze()

        sil_mask = self._get_sil_mask(cost)

        nonsil_idx = np.argwhere(sil_mask != self.charsiu_processor.sil_idx).squeeze()
        if nonsil_idx is None:
            raise Exception("No speech detected! Please check the audio file!")

        aligned_phone_ids = forced_align(cost[nonsil_idx, :], phone_ids[1:-1])

        aligned_phones = [
            self.charsiu_processor.mapping_id2phone(phone_ids[1:-1][i]) for i in aligned_phone_ids
        ]

        pred_phones = self._merge_silence(aligned_phones, sil_mask)

        pred_phones = seq2duration(pred_phones, resolution=self.resolution)

        pred_words = self.charsiu_processor.align_words(pred_phones, phones, words)
        if return_logits:
            return pred_phones, pred_words, out.logits.detach().cpu().squeeze()
        else:
            return pred_phones, pred_words

    def serve(self, audio, text, save_to, target_phones=None, output_format="textgrid"):
        """
         A wrapper function for quick inference

        Parameters
        ----------
        audio : TYPE
            DESCRIPTION.
        text : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : str, optional
            Output phone-taudio alignment as a "tsv" or "textgrid" file.
            The default is 'textgrid'.

        Returns
        -------
        None.

        """
        phones, words = self.align(audio, text, target_phones=target_phones)

        if output_format == "tsv":
            if save_to.endswith(".tsv"):
                save_to_phone = save_to.replace(".tsv", "_phone.tsv")
                save_to_word = save_to.replace(".tsv", "_word.tsv")
            else:
                save_to_phone = save_to + "_phone.tsv"
                save_to_word = save_to + "_word.tsv"

            self._to_tsv(phones, save_to_phone)
            self._to_tsv(words, save_to_word)

        elif output_format == "textgrid":
            self._to_textgrid(phones, words, save_to)
        else:
            raise Exception("Please specify the correct output format (tsv or textgird)!")

    def _to_textgrid(self, phones, words, save_to):
        """
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        word2textgrid(phones, words, save_path=save_to)
        # print('Alignment output has been saved to %s'%(save_to))

    def _merge_silence(self, aligned_phones, sil_mask):
        # merge silent and non-silent intervals
        pred_phones = []
        count = 0
        for i in sil_mask:
            if i == self.charsiu_processor.sil_idx:
                pred_phones.append("[SIL]")
            else:
                pred_phones.append(aligned_phones[count])
                count += 1
        assert len(pred_phones) == len(sil_mask)
        return pred_phones

    def _get_sil_mask(self, cost):
        # single out silent intervals

        preds = np.argmax(cost, axis=-1)
        sil_mask = []
        for key, group in groupby(preds):
            group = list(group)
            if key == self.charsiu_processor.sil_idx and len(group) < self.sil_threshold:
                sil_mask += [-1 for i in range(len(group))]
            else:
                sil_mask += group

        return np.array(sil_mask)
