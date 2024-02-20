import numpy as np
import copy
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import VAD
import tqdm
import torch
import pickle as pkl

class xVecExtractor:

    def __init__(self, method, batch_size=128):
        self.VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")

        if "xvec" in method:
            self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
        elif "ecapa" in method:
            self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.batch_size = batch_size


    def extract_all_xvecs(self, filename, minlen=1):
        pass

    def extract_xvector(self, filename):
        signal, fs = torchaudio.load(filename)
        target_sample_rate = 16000
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)(signal)
        assert fs==16000, 'ERROR: YOU MUST SUPPLY A FILE WITH 16kHz sampling rate'

        try:
            vadout = self.VAD.get_speech_segments(filename, large_chunk_size=1.5, small_chunk_size=0.5)
            if len(vadout.ravel()) > 2:
                start = int(vadout[0][0] * fs)
                end = int(vadout[-1][1] * fs)
                print(f'Warning multiple active speech segments found for {filename}')

            elif len(vadout) == 1:
                start = int(vadout[0][0] * fs)
                end = int(vadout[0][1] * fs)
                # print(vadout, start, end)
            else:
                start = 0
                end = len(signal[0])
                print(f'Warning VAD found no active speech segments for {filename}')

            vadsig = signal[0][start:end]
        except:
            print(f'Warning VAD internal failure for {filename}. Calculating x-vector with full utterance')
            vadsig = signal[0]

        output_emb = self.classifier.encode_batch(vadsig)
        return output_emb


if __name__=='__main__':
    filename = './examples/test.wav'
    xvx = xVecExtractor(method='xvec')
    xvx.extract_xvector(filename)

    xvector = xvx.extract_xvector(filename)
    print()