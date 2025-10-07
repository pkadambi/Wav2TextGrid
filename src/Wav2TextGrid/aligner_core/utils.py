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
from praatio.data_classes.textgrid import Textgrid
from itertools import groupby
from librosa.sequence import dtw
import pandas as pd
from pathlib import Path

import platform

PLATFORM = platform.system()

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
        tg.save(save_path, format="long_textgrid", includeBlankSpaces=False)
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
        tg.save(save_path, format="long_textgrid", includeBlankSpaces=False)
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




def textgridpath_to_phonedf(txtgrid_path: str, phone_key: str, remove_numbers=False, replace_silence=False,
                            silreplace_str='SIL', addsil=True):
    '''
    txtgrid_path - the path to the textgridfile
    phone_key - the key in the textgrid for the phoneme column
    '''
    txtgrid = textgrid.openTextgrid(txtgrid_path, False)
    phndf = extract_phone_df_from_textgrid(txtgrid=txtgrid, phone_key=phone_key, remove_numbers=remove_numbers)

    if replace_silence:
        phndf = phndf.replace('[SIL]', silreplace_str)
        phndf = phndf.replace('sp', silreplace_str)

    # Function to uppercase string columns
    def uppercase_strings(x):
        if isinstance(x, str):
            return x.upper()
        return x

    # Apply the function to each element in the DataFrame
    phndf = phndf.applymap(uppercase_strings)

    return phndf


def extract_phone_df_from_textgrid(txtgrid: Textgrid, phone_key, remove_numbers=False,
                                   silchar='[SIL]', replace_SP=True):
    '''
        txtgrid - praatio textgrid
        phone_key - the key for the phonemes
    '''
    try:
        phonelist = txtgrid._tierDict[phone_key].entries
    except:
        phonelist = txtgrid.tierDict[phone_key].entryList

    phonedf = []
    for interval in phonelist:
        _phone = interval.label
        if remove_numbers:
            _phone = re.sub(r'[0-9]+', '', _phone)
        phonedf.append([interval.start, interval.end, _phone])


    # why is this silence replace code duplicated? Because the output of this function will
    # contain textgrids with the silence character as [SIL] always
    phonedf = pd.DataFrame(phonedf, columns=['start', 'end', 'phone'])
    phonedf = phonedf.replace('sil', silchar)
    if replace_SP:
        phonedf = phonedf.replace('sp', silchar)
        phonedf = phonedf.replace('spn', silchar)

    return phonedf


def get_all_filetype_in_dir(directory, extension):
    extension = f'.{extension}' if '.' not in extension else extension
    files = []
    for path in Path(directory).rglob(f'*{extension}'):
        print(str(path.resolve()))
        files.append(str(path.resolve()))
    return files


def get_filename_with_upper_dirs(path, num_upper_dirs):
    """
    Extracts the filename along with a specified number of upper directories from the given path.

    Args:
        path (str or Path): The path from which to extract the filename.
        num_upper_dirs (int): The number of upper directories to include.

    Returns:
        str: The filename with the specified number of upper directories.
    """
    path = Path(path)
    # Get the desired upper directories
    upper_dirs = path.parts[-(num_upper_dirs + 1):-1]
    # Join the upper directories and the filename
    filename_with_upper_dirs = '/'.join(upper_dirs + (path.name,))


    return filename_with_upper_dirs

def get_matching_file_in_list(file_match_str: str, file_paths_to_search, verbose=True):
    filestem = get_filename_with_upper_dirs(file_match_str, num_upper_dirs=1)

    if PLATFORM == "Windows":
        filestem = filestem.replace("/", "\\")

    corresponding_files = [file for file in file_paths_to_search if filestem in file]
    if len(corresponding_files) > 1:
        if verbose:
            print(f'Error found more than one matching file in file_paths_to_search for filename {file_match_str}')
        raise Exception(f'Error found more than one matching file in file_paths_to_search for filename {file_match_str}')

    elif len(corresponding_files) == 0:
        if verbose:
            print(f'Error did not find any matching files in file_paths_to_search for filename {file_match_str}')
        raise Exception('Error did not find any matching files in file_paths_to_search')

    else:
        return corresponding_files[0]


