import numpy as np
import pandas as pd
import copy
from g2p_en import G2p
g2p = G2p()

def is_start_phone(phn, phonelist):
    '''
        each phoneme is associated with words in the transcript
        This function returns which phonemes in the textgrid occur at the start of a word

        Example:

        transcript - 'good day'
        phonelist = ['G', 'UH', 'D', ' ', 'D', 'AE', 'Y']

        1. the first phoneme at [0] is a start phone
        2a. find the locations of the spaces ' '
        2b. any phonemes occuring after a space are also start phones

    '''
    space_locs = np.argwhere(phonelist==' ').ravel()
    start_idxs = np.concatenate(([0], space_locs+1))
    phn_idx = np.argwhere(phonelist==phn)
    return any([_idx in start_idxs for _idx in phn_idx])

def is_end_phone(phn, phonelist):
    '''
    find the location of end phonemes
    does nearly the same thing as the is_start_phone function, except for end phones
    '''
    space_locs = np.argwhere(phonelist==' ').ravel()
    end_idxs = np.concatenate((space_locs-1, [len(phonelist)]))
    phn_idx = np.argwhere(phonelist==phn)
    return any([_idx in end_idxs for _idx in phn_idx])

def collapse_repeated_phones(input_df, phonekey='phone'):
    '''
    input:
        input_df - the input textgrid

    logic:
        - just concatenates consecutive identical phonemes

        for each line in the input textgrid df, check if the next phoneme is the same as the current phoneme
        if they are, simply

    '''
    # keepdata = []
    inp_df = copy.deepcopy(input_df)
    ii=0
    while ii<len(inp_df)-1:
        if inp_df.at[ii, phonekey]==inp_df.at[ii+1, phonekey]:
            newend = inp_df.at[ii+1, 'end']
            inp_df.at[ii, 'end'] = newend
            inp_df = inp_df.drop(ii+1, axis=0).reset_index(drop=True)
        else:
            ii+=1
    return inp_df

def process_silences(inp_df, transcript: str, silphone='sil'):
    phonelist = g2p(transcript)
    tgdf = copy.deepcopy(inp_df)
    ''' flag silences in the middle of a word with nan'''
    for ii in range(len(tgdf)):
        if ii < len(tgdf) - 1 and ii > 0:
            # print(tgdf[phonekey][ii])
            if tgdf['phone'][ii] == silphone:
                prevphone = tgdf['phone'].iloc[ii - 1]
                nextphone = tgdf['phone'].iloc[ii + 1]

                if prevphone == nextphone and not (
                        is_end_phone(prevphone, phonelist) and is_start_phone(nextphone, phonelist)):
                    tgdf.at[ii, 'phone'] = np.nan

    ''' remove the silences'''
    # tgdf = tgdf[~pd.isna(tgdf[phonekey])].reset_index(drop=True).drop(columns=['index'])
    tgdf = tgdf[~pd.isna(tgdf['phone'])].reset_index(drop=True)
    ''' collapse the repeated phonemes '''
    return collapse_repeated_phones(tgdf, phonekey='phone')