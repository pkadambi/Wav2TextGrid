import librosa
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.VAD import VAD


class xVecExtractor:

    def __init__(self, method, batch_size=128, device="cpu"):

        assert device in ["cuda", "cpu"], "Error: device must be `cuda` or `cpu`"

        self.VAD = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty", run_opts={"device": device}
        )
        if "xvec" in method:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device}
            )
        elif "ecapa" in method:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
            )
        self.batch_size = batch_size

        self.device = device

    def extract_xvector(self, filename):
        signal, fs = librosa.load(filename, sr=None)
        signal = torch.Tensor(signal).view(1, -1)
        target_sample_rate = 16000
        if fs != target_sample_rate:
            signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)(
                signal
            )
        signal = signal.to(self.device)

        try:
            vadout = self.VAD.get_speech_segments(
                filename, large_chunk_size=1.5, small_chunk_size=0.5
            )
            if len(vadout.ravel()) > 2:
                start = int(vadout[0][0] * fs)
                end = int(vadout[-1][1] * fs)
                # print(f'Warning multiple active speech segments found for {filename}')

            elif len(vadout) == 1:
                start = int(vadout[0][0] * fs)
                end = int(vadout[0][1] * fs)
                # print(vadout, start, end)
            else:
                start = 0
                end = len(signal[0])
                # print(f'Warning VAD found no active speech segments for {filename}')

            vadsig = signal[0][start:end]
        except Exception:
            print(
                f"Warning VAD internal failure for {filename}."
                "Calculating x-vector with full utterance"
            )
            vadsig = signal[0]

        output_emb = self.classifier.encode_batch(vadsig)
        return output_emb


if __name__ == "__main__":
    filename = "./examples/test.wav"
    xvx = xVecExtractor(method="xvec")
    xvx.extract_xvector(filename)

    xvector = xvx.extract_xvector(filename)
    print()
