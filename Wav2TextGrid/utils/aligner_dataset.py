import torch
import os
import tqdm
import librosa
import numpy as np
import pickle as pkl
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor
from Wav2TextGrid.aligner_core.utils import textgridpath_to_phonedf

def get_satvector_dict(audio_paths, adaptation_type:str, xvextractor: xVecExtractor):
    SUPPORTED_SPEAKER_EMBEDDINGS = ['xvec', 'ecapa']

    if adaptation_type is None or adaptation_type.lower() not in SUPPORTED_SPEAKER_EMBEDDINGS:
        raise Exception(f'Error: adaptation method must be one of: {SUPPORTED_SPEAKER_EMBEDDINGS}')
    else:
        satvector_dict = {}
        print('Extracting xVectors...')
        for pth in tqdm.tqdm(audio_paths):
            satvector_dict[pth] = xvextractor.extract_xvector(pth)[0][0].cpu().numpy()

    return satvector_dict


class AlignerDataset(torch.utils.data.Dataset):
    '''
    Extracts the textgrids, audios,
    '''
    def __init__(self, args, audio_paths: list, textgrid_paths: list, phone_key:str='phones', words_key:str='words',
                 adaptation_type:str='xvec', satvector_path:str=None, model:Wav2Vec2ForCTC = None):

        self.audio_paths = audio_paths
        self.textgrid_paths = textgrid_paths
        self.words_key = words_key
        self.phone_key = phone_key
        self.model = model

        if (satvector_path is not None and adaptation_type is None) or (satvector_path is None and adaptation_type is not None):
            raise Exception(f'Error: both both satvector_path and adaptation_type must be specified. Only one cannot be specified. Found {satvector_path} for satvector_path and {adaptation_type} for adaptation_type.')

        if satvector_path is not None: # make this a dictionary instead, there's no reason why this should be a csv
            print('\nExtracting SAT Vectors')
            # satvector_path = f'{split}_{satvector_path}' if split is not None else satvector_path
            if not os.path.exists(satvector_path) or args.CLEAN:
                xvx = xVecExtractor(adaptation_type, batch_size=64, device='cuda')
                self.satvector_dict = get_satvector_dict(audio_paths=audio_paths, adaptation_type=adaptation_type, xvextractor=xvx)

                if not os.path.exists(satvector_path):
                    pkl.dump(self.satvector_dict, open(satvector_path, 'wb'))
            else:
                self.satvector_dict = pkl.load(open(satvector_path, 'rb'))

            try:
                self.satvectors = [self.satvector_dict[pth] for pth in self.audio_paths]
            except:
                print('Did not find expected files in previously speaker adaptation vectors... extracting again...')
                xvx = xVecExtractor(adaptation_type, batch_size=64, device='cuda')
                self.satvector_dict = get_satvector_dict(audio_paths=audio_paths, adaptation_type=adaptation_type, xvextractor=xvx)

                pkl.dump(self.satvector_dict, open(satvector_path, 'wb'))

            self.satvectors =[self.satvector_dict[pth] for pth in self.audio_paths]

        # satvector_path = f'{split}_{satvector_path}' if split is not None else satvector_path

        else:
            self.satvectors = [None for _path in self.audio_paths]


        # Extract Audios
        print('Loading audio files')
        self.audios = self.extract_audio()
        self.audio_lens = [len(_audio) for _audio in self.audios]
        # Extract speaker ids and file ids
        self.speaker_ids = [_path.split('/')[-2] for _path in self.audio_paths]
        self.ids = [_path.split('/')[-1].split('.')[0] for _path in self.audio_paths]

        #Step 1: extract word transcripts
        print('\nExtracting transcripts, phone and word bounds')
        self.text_transcripts = [self.extract_text_transcript(_path) for _path in tqdm.tqdm(self.audio_paths)]

        #Step 2: extract phone transcripts/alignments
        print('\nExtracting alignments')
        self.phone_alignments, self.word_alignments = self.extract_phone_words_bounds()

        #step3: extract frame labels
        print('\nExtracting Framewise Labels')
        self.frame_phn_labels, self.frame_times = self.gen_frame_labels_from_alignment()

        # if sat_method is not None:
        #     get_satvector_dict(audio_paths, adaptation_type)



    def save_satvector_dict(self, satvector_path):
        pass

    # assumes that the directory is in the file
    def extract_text_transcript(self, path):
        transcript_path = path.split('.')[0] + '.lab'
        f = open(transcript_path, 'rb')
        transcript = f.read()
        transcript = transcript.decode('utf-8').replace('\n', '')
        return transcript

    def extract_audio(self):
        return [librosa.load(_audio_path, sr=16000)[0] for _audio_path in tqdm.tqdm(self.audio_paths)]

    def extract_phone_transcripts(self):
        transcripts = []
        for _path in self.audio_paths:
            _transcript_path = _path.split('.') + 'lab'
            f = open(_transcript_path, 'rb')
            transcripts.append(f.read())
            f.close()
        return transcripts

    def extract_phone_words_bounds(self):
        phone_alignments = []
        word_alignments = []
        frame_phone_lables = []
        frame_phone_times = []
        phones = []
        for ii, (_path, _tgpath) in enumerate(tqdm.tqdm(zip(self.audio_paths, self.textgrid_paths))):
            # _tgpath = _path[:-3] + 'TextGrid'
            # tgfilename = _path.split('/')[-1][:-3] + 'TextGrid'
            # splitpath = _path.split('/')[:5]
            # speaker_dir = _path.split('/')[-2]
            # splitpath.append('manually-aligned-text-grids')
            # splitpath.append(speaker_dir)
            # splitpath.append(tgfilename)
            # manual_tg_path = '/'.join(splitpath)
            # manual_tg_path = _path.replace('child_speech_16_khz', 'manually-aligned-text-grids').replace('.wav', '.TextGrid')
            manual_tg_path = _tgpath
            phone_df = textgridpath_to_phonedf(manual_tg_path, phone_key=self.phone_key, remove_numbers=True)
            words_df = textgridpath_to_phonedf(manual_tg_path, phone_key=self.words_key, remove_numbers=True)
            phone_df = phone_df.replace('sil', '[SIL]')
            phone_df = phone_df.replace('<U>', '[]')
            phonedict = {'start': list(phone_df.iloc[:,0]),
                         'stop':list(phone_df.iloc[:,1]),
                         'utterance': list(phone_df.iloc[:,2])}

            worddict = {'start': list(words_df.iloc[:,0]),
                         'stop': list(words_df.iloc[:,1]),
                         'utterance': list(words_df.iloc[:,2])}
            phone_alignments.append(phonedict)
            word_alignments.append(worddict)
            phones.extend(list(phone_df.iloc[:, 2]))
            self.unique_phones = list(np.unique(phones))
        return phone_alignments, word_alignments

    def gen_frame_labels_from_alignment(self):
        '''
        RUN AFTER 'extract_phone_words_bounds'

        Returns
        -------

        '''
        frame_phn_labels = []
        frame_times = []
        w2v2_time_step = .02001/2

        def _find_framewise_phn_labels(timestep, phn_dict):
            framewise_phn_labels = []

            for jj, _time in enumerate(timestep):
                gt_start_indicator = _time>np.array(phn_dict['start'])
                lt_end_indicator = _time<np.array(phn_dict['stop'])
                match_indicator = np.logical_and(gt_start_indicator, lt_end_indicator)
                phone_ind = np.argwhere(match_indicator).ravel()

                if len(phone_ind)==0:
                    _phn='[SIL]'
                    # np.concatenate([np.array(phn_dict['start']).reshape(-1, 1), np.array(phn_dict['stop']).reshape(-1, 1), np.array(phn_dict['utterance']).reshape(-1, 1)], axis=1)
                elif len(phone_ind)>1:
                    Exception('Error found more than 1 matching phone index!\n Manual Transcript dictionary:\n', phn_dict)
                else:
                    _phn = phn_dict['utterance'][phone_ind[0]]
                framewise_phn_labels.append(_phn)
            return framewise_phn_labels

        for ii, _path in enumerate(tqdm.tqdm(self.audio_paths)):
            _audiolen = len(self.audios[ii])/16000

            if self.model is not None:
                # _audiolen = get_aligner_frame_seq_len_model_processor(audio_filepath=_path, model=self.model, processor=self.processor)
                audiolen_samples = len(librosa.load(_path, sr=16000)[0])
                n_timesteps = int(self.model._get_feat_extract_output_lengths(audiolen_samples))
                timesteps = np.arange(0, n_timesteps)*w2v2_time_step
                # timesteps = np.linspace(0, _audiolen, n_timesteps)

            else:
                timesteps = np.arange(0, _audiolen, step=w2v2_time_step)[1:] #No embedding for t=0
            _phn_dct = self.phone_alignments[ii]

            frame_times.append(timesteps)
            frame_phn_labels.append(_find_framewise_phn_labels(timesteps, _phn_dct))

            # phone_df = textgridpath_to_phonedf(manual_tg_path, phone_key='ha phones', remove_numbers=True)

        return frame_phn_labels, frame_times

    def return_as_dict(self):
        dct = DatasetDict({'file': self.audio_paths, 'audio':self.audios, 'phone_alignments': self.phone_alignments,
               'word_alignments': self.word_alignments, 'speaker_id': self.speaker_ids, 'audio_len': self.audio_lens,
                           'id': self.ids, 'sentence': self.text_transcripts, 'frame_phones': self.frame_phn_labels,
                           'frame_times': self.frame_times, 'ixvector': self.satvectors})
        return dct

    def return_as_datsets(self):
        self.dct = self.return_as_dict()
        self.dataset = Dataset.from_dict(self.dct)
        return self.dataset

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.audio_paths)
