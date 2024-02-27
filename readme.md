# Wav2TextGrid
**Alpha 0.0.19 version** of the Wav2TextGrid phonetic forced aligner. 

Wav2TextGrid been tested on Ubuntu18.04, Ubuntu22.04, and Windows 11.

The aligner is currently in development, please contact pkadami@asu.edu if you have any questions or issues in using the aligner

## Installation + quick start

`pip install Wav2TextGrid==0.0.19 --force-reinstall`

Make sure to include the flags and the correct version number, check PyPI for the most recent version: 
https://pypi.org/project/Wav2TextGrid/

`git clone git@github.com:pkadambi/Wav2TextGrid.git`

`cd Wav2TextGrid`

`wav2textgrid ./Wav2TextGrid/examples/test.wav ./Wav2TextGrid/examples/test.lab ./test.TextGrid`
Output text grid stored in test.TextGrid

Python 3.10 required

------
## Demo/Usage

Please refer to the **Aligner Scope and Applicability** section to verify that your data and use case are a good fit for this alignment system. 

------


## Aligning a directory of .wav files

Say your audio is in `/path/to/wavfiles` and your transcripts are in `/path/to/transcripts` (note that these two 
directories _can_ be the same), and you want the output TextGrids to be in `/path/to/outputTextGrids`.

You can simply use the command:

`wav2textgrid /path/to/wavfiles /path/to/transcripts /path/to/outputTextGrids`


**<u>Expected location and format of transcript files</u>**:

It is assumed that an example `.wav` file `/path/to/wavfiles/directoryA/subdirB/file1.wav`
will also have a corresponding transcript `/path/to/transcripts/directoryA/subdirB/file1.lab`.

Otherwise the transcript will not be found and alignment will fail.


## Aligning Individual files
For an input `.wav` file:  `example.wav` With transcript contained in `example.lab`

Use the following command: `wav2textgrid example.wav example.lab example.TextGrid`

---
# Wav2TextGrid Application

---

Coming soon!

----
## Input data format

----
### Input Audio: `.wav` files
All `.wav` files are expected to be 16kHz mono.

### Transcripts: `.lab` files
All `.lab` files are expected to contain only one line with the string transcript.
For example a `.lab` file may look like:

file1.lab: 
`SHE HAD YOUR DARK SUIT IN GREASY WASH WATER ALL YEAR`

---
## Aligner Scope of Applicability

---

If your data and use case do not align with the training data described below (short child speech utterances) for Wav2TextGrid, take care to verify the alignments produced by Wav2TextGrid.

**Training data used**
The aligner was trained on manual alignments from a corpus of children 3-7 years old, speaking sentences from the Test of Childhood Stuttering (TOCS), using both single word utterances and short sentences. The duration of each utterance was ~2-5s in length, and a total of ~3700 utterances (~2 hours of data) were used to train the model. The aligner likely will work best on similar data (short child speech utterances from TOCS).
This training training on child speech was done after initializing using an existing system trained on adult speech (CommonVoice+LibriSpeech as training data). However, after fine-tuning on our child speech corpus, we did not verify Wav2TextGrid on adult speakers again.

Thus, Wav2TextGrid likely performs best on children from this same age range and on audio collected in this same context. Use in conversational speech or speech including both children and adults has not been validated.

**Usage on adult speakers**
As previously mentioned, Wav2TextGrid has been fine-tuned on child speech using an initialization model trained to align CommonVoice and LibriSpeech utterances. But, Wav2TextGrid wasn't verified on adult speech data again after training on child speech data. As the fine-tuning child speech dataset contained short utterances, Wav2TextGrid would likely work best on shorter adult specech utterances. 



------
## Algorithm Details

------
### Pronunciation Dictionary
The CMUdict in `g2p_en` python package is used to create the target phoneme sequence from the provided textgrid

### x-vector extraction

1. First a VAD is applied (`speechbrain/vad-crdnn-libriparty` from the huggingface model hub)
2. xVectors are extracted using `speechbrain/spkrec-ecapa-voxceleb` from the huggingface model hub

### Forced Alignment

Forced alignments are generated with a 10ms granularity. We use a Wav2Vec2 network to predict 
a frame-wise phoneme posterior, and Viterbi decoding to calculate the optimal alignment path.   


## Acknowledgements
We would like to thank the authors of the "Charsiu" aligner: https://github.com/lingjzhu/charsiu. 
Parts our code have been adapted from the repo above. 
