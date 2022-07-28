import os
import functools
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
import librosa
import soundfile as sf
import parselmouth
import opensmile
import numpy as np
import re

def walk_filter(input_dir, file_extension=None):
    files = []
    labels = []

    for r, _, fs in os.walk(input_dir, followlinks=True):

        # if file_extension:
        #     files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        # else:
        for f in fs:
            files.append(os.path.join(r, f))
            file_basename = os.path.splitext(os.path.basename(f))[0]
            tokens = re.split("-",file_basename)
            labels.append(int(tokens[2]))


            # files.extend([os.path.join(r, f) for f in fs])

    return files,labels

def extract_mel_spectogram(audio):
    x_audio = tf.constant(audio)
    stft = tf.signal.stft(
        x_audio,
        400, # frame_length , the window length
        160, # originally 160, reduces the upsampling discrepancy in encoder # frame_step , hop size
        fft_length= 512,
        window_fn= functools.partial(window_ops.hann_window, periodic=True),
        pad_end=False,
        name=None
    )
    stft = tf.abs(stft)
    # Returns a matrix to warp linear scale spectrograms to the mel scale
    mel_spect_input = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=tf.shape(stft)[1], #257
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.float32,
        name=None
    )
    input_data = tf.tensordot(stft, mel_spect_input, 1)
    input_data = tf.math.log(input_data + 1e-6)
    input_data = tf.expand_dims(input_data, -1)
    return input_data.numpy()

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

def extract_pitch_from_file(signal):

    '''
    extract the pitch of an audio file given the path
    the algorithm used for pitch extraction is PYin
    '''

    # print("extracting pitch")
    # sample = np.load(filepath)
    # signal = sample['audio']

    sr = 16000
    snd = parselmouth.Sound(signal,sampling_frequency = sr)
    pitch_parl = snd.to_pitch_ac()
    pitch = pitch_parl.selected_array['frequency']
    
    pitch_val = np.concatenate((pitch,np.zeros(1))) # zero pad pitch values to match dimensions  
    
    # print('out_file_path: ',out_file_path)
    return pitch_val

def extract_egmaps_from_file(signal):

    '''
    extract eGMAPS fetures 
    '''

    # print("extracting egmaps")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    y = smile.process_signal(signal, 16000)
    features = y.to_numpy()
    features_val = np.concatenate((features,np.zeros((2,25)))) # zero pad pitch values to match dimensions

    return features_val

def extract_features_from_file(audio):
    # file_name = os.path.splitext(os.path.basename(filepath))[0]
    # out_file_path = "{}/{}_features.npz".format(output_dir,file_name)
    # if os.path.isfile(out_file_path): 
    #   print("file exists")
    # #   return
    # file = np.load(filepath)
    pitch_val = extract_pitch_from_file(audio)
    egmaps_val = extract_egmaps_from_file(audio)

    return pitch_val,egmaps_val
    # save 
    # np.savez_compressed(out_file_path,pitch = pitch_val,egmaps = egmaps_val)
