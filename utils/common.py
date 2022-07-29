import os
import librosa
import soundfile as sf
import numpy as np
import re
import utils.common

def normalize(data , axis = None):
    
    mean_val = np.mean(data , axis = axis)
    std_val = np.std(data,axis = axis)
    if axis :
        mean_val = np.expand_dims(mean_val,axis = axis)
        std_val = np.expand_dims(std_val,axis = axis)

    return (data-mean_val)/std_val


def walk_filter(input_dir, file_extension=None):
    files = []

    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])

    return files


def featch_files_labels(input_dir):
    files = []
    labels = []

    for r, _, fs in os.walk(input_dir, followlinks=True):

        for f in fs:
            files.append(os.path.join(r, f))
            file_basename = os.path.splitext(os.path.basename(f))[0]
            tokens = re.split("-",file_basename)
            labels.append(int(tokens[2]))

    return files,labels


def load_audio(file_path, sr=44100):
    '''
    load an audio file to mono and the specified sampling rate
    '''
    try:
        audio, fs = sf.read(file_path, dtype='float32')
    except Exception:
        audio, fs = librosa.load(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if fs != sr:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=sr)
    return audio

def test_train_split(data,labels):
    number_of_datapoints = data.shape[0]
    # shuffle indexs
    indexes = np.random.permutation(number_of_datapoints)
    train_idx = indexes[:int(number_of_datapoints*0.8)]
    test_idx = indexes[int(number_of_datapoints*0.8):]

    X = {}
    y ={}
    X['train'],X['test'] = data[train_idx], data[test_idx]
    y['train'],y['test'] = labels[train_idx],  labels[test_idx]

    return X,y

def read_features(input_dir):
    ## read features

    file_lists = utils.common.walk_filter(input_dir)
    print(len(file_lists))
    label_list = []
    mel_list = []
    pitch_list = []
    egmaps_list =[]
    mfcc_list =[]
    for file in file_lists:
        zip_arr = np.load(file)
        mel_list.append(zip_arr['mel'])
        pitch_list.append(zip_arr['pitch'])
        egmaps_list.append(zip_arr['egmaps'])
        label_list.append(zip_arr['label'])
        mfcc_list.append(zip_arr['mfcc'])

    res_dict ={}
    res_dict['mel'] = np.array(mel_list).squeeze()
    res_dict['pitch'] = np.array(pitch_list)
    res_dict['egmaps'] = np.array(egmaps_list)
    res_dict['label'] = np.array(label_list)
    res_dict['mfcc'] = np.array(mfcc_list).squeeze()

    return res_dict