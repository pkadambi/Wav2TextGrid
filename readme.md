### Wav2TextGrid
--- 

*Version 0.1.2 (Beta)* — A phonetic forced aligner

Wav2TextGrid has been tested on Ubuntu 18.04, Ubuntu 22.04, and Windows 11.

⚠️ This aligner is currently in development. If you encounter issues, please open an issue or email me directly: pkadambi@asu.edu


# 🚀 Installation + Quick Start

### Wav2TextGridGUI
- A Linux friendly (tested on Ubuntu 22.04 and 18.04) version of the aligner [can be found here](https://drive.google.com/file/d/1ZaRlLNE17y6OAivL_11NRLxlDbDWnOTh/view?usp=sharing)

---
### Required Dependencies

 <u>Pytorch/Torchaudio/Torchvision</u>

⚠️ Note: The package installs the CPU version of pytorch by default.
Install your preferred pytorch GPU support (v2.5>) BEFORE installing if you would like to use a GPU.




### Installation 
`pip install Wav2TextGrid==0.01.02`

Make sure to include the version number. Check for the latest release on PyPI:
https://pypi.org/project/Wav2TextGrid/

Python 3.10 required


# Usage

------
# Quickstart Example: Alignment without training

1. **Download and extract `examples.zip` in this repo**
2. **Align an single file** (in unix systems): 
    ```
    w2tg wavfile_or_dir=/path/to/examples/test.wav  transcriptfile_or_dir=/path/to/examples/test.lab outfile_or_dir=./output.TextGrid
    ```
     OR more simply.
    ```
    w2tg /path/to/examples/test.wav  /path/to/examples/test.lab ./output.TextGrid
    ```
    This aligns the file `test.wav` using `test.lab` to output `output.TextGrid` 

3. **Align an entire directory**
    ```
    w2tg /path/to/examples/ /path/to/examples/ ./outputs
    ```
   This aligns all `.wav` and `.lab` files in the folder `examples` to the `outputs` folder 

[//]: # (Output text grid stored in test.TextGrid)

## 📁 Best Practice: Usage and Data Format For `w2tg `
Follow the kaldi style
```
│/Dataset/folder/
├── Speaker1
│ ├── file1.lab
│ ├── file1.wav
│ ├── file2.wav
│ ├── file2.lab
├── Speaker2
│ ├── file3.lab
│ ├── file3.wav
│ ├── file4.wav
│ ├── file4.lab
```


### Best Practice - `w2tg_train`
Follow the kaldi style
```
│/Dataset/folder/
├── Speaker1
│ ├── file1.lab
│ ├── file1.wav
│ ├── file2.wav
│ ├── file2.lab
├── Speaker2
│ ├── file3.lab
│ ├── file3.wav
│ ├── file4.wav
│ ├── file4.lab
```
# ⚙️ Quickstart Example: Training your own alignment & perform alignment with the trained model 
In this section we will see how to train a new model, and perform alignment with the trained model 

Steps
1. Prepare dataset
2. Train the model using `w2tg_train`
3. Align using your trained model using `w2tg`

### 1. Prepare dataset
Follow the kaldi style
```
│/Dataset/folder/
├── Speaker1
│ ├── file1.lab
│ ├── file1.wav
│ ├── file1.TextGrid
│ ├── file2.wav
│ ├── file2.lab
│ ├── file2.TextGrid
├── Speaker2
│ ├── file3.lab
│ ├── file3.wav
│ ├── file3.TextGrid
│ ├── file4.wav
│ ├── file4.lab
│ ├── file4.TextGrid
```

### 2. Train the Model
      
      w2tg_train \
      --train_audio_dir=/Dataset/folder \
      --train_textgrids_dir=/Dataset/folder \
      --eval_audio_dir=/EvalDataset/folder \
      --eval_textgrids_dir=/EvalDataset/folder \
      --model_output_dir=./OUTPUT_MODEL \
      --dataset_dir=./DATA \
      --run_output_folder=./RESULTS

      
   
   Note that the `--eval_audio_dir` and `--eval_dataset_dir` are optional. 

   **<u>Eval dataset must include textgrids.</u>**

   **<u>If you want to generate alignments for a target dataset (which doesn't have textgrids) using a trained model, you must use step 3. </u>** 

Training produces the following 
```
│--run_output_folder
├── alignments_baseline
│   └── alignments on the eval dataset (if provided) using the base model
├── eval_trained
│   └── alignments on the eval dataset (if provided) using the trained model 
└── OUTPUT_MODEL
    ├── trained model files
```


### 3. **Align an entire directory**

    ```
    w2tg /Data/to/align /Data/to/align ./output_folder --aligner_model=./RESULTS/OUTPUT_MODEL
    ```
Where:

- `--aligner_model` is your trained model directory
  
- `./output_folder` is the target alignment output
- `/Data/to/align` for format see Quickstart Example: Alignment without training
- 
   Where `./output folder`, `--aligner_model`, `/Data/to/align` can be specified based on user data, and `./RESULTS/OUTPUT_MODEL` is the user trained model.
    



# 🎵 Data Format

----

## Audio Format

#### `w2tg` likely works for `mp3` and `wav` files, just make sure to specify `--filetype=mp3`

#### `w2tg_train` only works with `.wav` 

#### Input Audio: `.wav` or `.mp3` files will be 

## Transcripts: `.lab` files format
All `.lab` files are expected to contain only one line with the string transcript.
For example a `.lab` file may look like:

file1.lab: 
`SHE HAD YOUR DARK SUIT IN GREASY WASH WATER ALL YEAR`

## Textgrid Format
Must contain `IntervalTier` of `phones`.


## 🧪 Wav2TextGrid Demo

---

Coming soon!

---
## ⚠️ Aligner Scope of Applicability

---

Wav2TextGrid was trained on a corpus of children (ages 3–7, 3700 utterances) reading short utterances (~2–5s) from the Test of Childhood Stuttering (TOCS). It works best for:

- Child speech

- Short utterances

- Clean, non-conversational audio

The aligner was initialized using a model trained on adult speech (CommonVoice + LibriSpeech), but was not re-validated on adult speakers after fine-tuning. Use with caution on:

- Long utterances

- Conversational audio

- Adult speech (especially unvalidated)

If your data and use case do not align with the training data described abive (short child speech utterances) for Wav2TextGrid, take care to verify the alignments produced by Wav2TextGrid.


Thus, Wav2TextGrid likely performs best on children from this same age range and on audio collected in this same context. Use in conversational speech or speech including both children and adults has not been validated.



## 📝 TODOs

---

### Functionality
The TODOs left will be completed by prior to the publication of the article.
- [x] ~~Aligner functionality~~
- [x] ~~Quickstart Demo~~
- [x] ~~Training code + functionality~~
- [ ] ~~Demo for training aligner system [By 04/20]~~
- [ ] ~~GUI Application for Linux [By 04/20]~~
- [ ] GUI Application for Windows (ongoing, will be complete by 05/31)
- [ ] Add training functionality to GUI (longer term, likely by 06/31) 

## Attribution and Citation

---
- Our paper can be found here: https://doi.org/10.1044/2024_JSLHR-24-00347

BibTeX
```
@article{kadambi2025tunable,
  title={A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech},
  author={Kadambi, Prad and Mahr, Tristan J and Hustad, Katherine C and Berisha, Visar},
  journal={Journal of Speech, Language, and Hearing Research},
  pages={1--19},
  year={2025},
  publisher={American Speech-Language-Hearing Association}
}
```
MLA

 ``` 
 Kadambi, Prad, et al. "A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech." Journal of Speech, Language, and Hearing Research (2025): 1-19.
 ``` 

APA
```
Kadambi, P., Mahr, T. J., Hustad, K. C., & Berisha, V. (2025). A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech. Journal of Speech, Language, and Hearing Research, 1-19.
```


------
## Algorithm Details

------
### Pronunciation Dictionary
Uses `g2p_en` (CMUdict) for text-to-phoneme conversion.


### x-vector extraction

1. Voice Activity Detection (VAD): `speechbrain/vad-crdnn-libriparty` (Hugging Face)
2. xVector speaker embeddings: `speechbrain/spkrec-ecapa-voxceleb` (Hugging Face)

### Forced Alignment
- Frame-level phoneme prediction via Wav2Vec2, phoneme posterior distribution

- Alignment via Viterbi decoding (10ms granularity)

## Acknowledgements

---
Portions of this project adapt code from the excellent Charsiu aligner https://github.com/lingjzhu/charsiu.



