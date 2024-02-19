# Wav2TextGrid
**Alpha 0.1 version** of the Wav2TextGrid phonetic forced aligner.

## Installation

`pip install Wav2TextGrid`

------
## Demo/Usage

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



------
## Algorithm Details

------
### Pronunciation Dictionary
The CMUdict in `g2p_en` python package is used to create the target phoneme sequence from the 

### x-vector extraction

1. First a VAD is applied (`speechbrain/vad-crdnn-libriparty` from the huggingface model hub)
2. xVectors are extracted using `speechbrain/spkrec-ecapa-voxceleb` from the huggingface model hub

### Forced Alignment

Forced alignments are generated with a 10ms granularity. We use a Wav2Vec2 network to predict 
a frame-wise phoneme posterior, and Viterbi decoding to calculate the optimal alignment path.   


## Acknowledgements
We would like to thank the authors of the "Charsiu" alignment system found here: https://github.com/lingjzhu/charsiu. 

Portions of this codebase have been adapted from the repo above. 
