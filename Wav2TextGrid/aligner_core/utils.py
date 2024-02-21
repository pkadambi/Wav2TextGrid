'''
The code in this file has been adapted from:
https://github.com/lingjzhu/charsiu
Primary Author: lingjzhu, henrynomeland
MIT license
'''

import numpy as np
import re
import os
from praatio import textgrid
from itertools import groupby
from librosa.sequence import dtw

def seq2duration(phones, resolution=0.01):
    """
    xxxxx convert phone sequence to duration

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.

    Returns
    -------
    out : list
        xxxxx A list of duration values.

    """

    counter = 0
    out = []
    for p, group in groupby(phones):
        length = len(list(group))
        out.append((round(counter * resolution, 2), round((counter + length) * resolution, 2), p))
        counter += length
    return out


def duration2textgrid(duration_seq, save_path=None):
    """
    Save duration values to textgrids

    Parameters
    ----------
    duration_seq : list
        xxxxx A list of duration values.
    save_path : str, optional
        The path to save the TextGrid files. The default is None.

    Returns
    -------
    tg : TextGrid file?? str?? xxxxx?
        A textgrid object containing duration information.

    """

    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    if save_path:
        tg.save(save_path, format="short_textgrid", includeBlankSpaces=False)
    return tg


def word2textgrid(duration_seq, word_seq, save_path=None):
    """
    Save duration values to textgrids

    Parameters
    ----------
    duration_seq : list
        xxxxx A list of duration values.
    save_path : str, optional
        The path to save the TextGrid files. The default is None.

    Returns
    -------
    tg : TextGrid file?? str?? xxxxx?
        A textgrid object containing duration information.

    """

    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    wordTier = textgrid.IntervalTier('words', word_seq, 0, word_seq[-1][1])
    tg.addTier(wordTier)
    if save_path:
        # if the subdirectory does not exist in the specified textgrids directory, they will be made
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # saves the textgrids in the corresponding directory/subdirectory
        tg.save(save_path, format="short_textgrid", includeBlankSpaces=False)
    return tg


def forced_align(cost, phone_ids):
    """
    Force align text to audio.

    Parameters
    ----------
    cost : float xxxxx
        xxxxx.
    phone_ids : list
        A list of phone IDs.

    Returns
    -------
    align_id : list
        A list of IDs for aligned phones.

    """

    D, align = dtw(C=-cost[:, phone_ids],
                   step_sizes_sigma=np.array([[1, 1], [1, 0]]))

    align_seq = [-1 for i in range(max(align[:, 0]) + 1)]
    for i in list(align):
        #    print(align)
        if align_seq[i[0]] < i[1]:
            align_seq[i[0]] = i[1]

    align_id = list(align_seq)
    return align_id










