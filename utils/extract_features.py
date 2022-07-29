import argparse
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
import parselmouth
import opensmile
import functools
import numpy as np
import utils.common as common
import concurrent.futures as ft
import os

def extract_mel_spectogram(audio):
    x_audio = tf.constant(audio)
    stft = tf.signal.stft(
        x_audio,
        400, # frame_length , the window length
        160, #frame_step , hop size
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
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(input_data)[:,:40]
    # print("shape?: ",mfccs.numpy().shape)
    mel_arr = np.mean(input_data.numpy(),axis = 0)
    mfcc_arr = np.mean(mfccs.numpy(),axis = 0)


    return  mel_arr, mfcc_arr


def extract_pitch_from_file(signal):

    '''
    extract the pitch of an audio file given the path
    the algorithm used for pitch extraction is PYin
    '''
    sr = 16000
    snd = parselmouth.Sound(signal,sampling_frequency = sr)
    pitch_parl = snd.to_pitch_ac()
    pitch = pitch_parl.selected_array['frequency']
    
    pitch_val = np.concatenate((pitch,np.zeros(1))) # zero pad pitch values to match dimensions  
    
 
    return np.mean(pitch_val,axis = 0)

def extract_egmaps_from_file(signal):

    '''
    extract eGMAPS fetures 
    '''
    # print("extracting egmaps!")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    y = smile.process_signal(signal, 16000)
    features = y.to_numpy()
    features_val = np.concatenate((features,np.zeros((2,25)))) # zero pad pitch values to match dimensions

    return np.mean(features_val,axis =0)

def save_features(file,label,sr=16000,window_len=2.56,augment_factor =1):
    signal = common.load_audio(file,sr=16000)
    # print("signal shape: ",signal.shape)
    n_samples = int(window_len * sr)
    mel_spec_list = []
    label_list = []
    egmap_list = []
    pitch_list = []

    # n_windows = int(signal.shape[0] // n_samples) * augment_factor

    # print("n_windows: ",n_windows)
    file_base_name =  os.path.splitext(os.path.basename(file))[0]
    # for i in range(n_windows):
    # file_name = file_base_name+'_'+str(n_windows)+'.npz'
    file_name = file_base_name+'.npz'
    path_to_save = os.path.join('features',file_name)
    print(path_to_save)

    if not os.path.exists(path_to_save):
        print("generating features!")

        # audio_frames = signal[ int(n_samples * i / augment_factor):int(n_samples * (1 + i / augment_factor))]

        mel, mfcc = extract_mel_spectogram(signal)
        # print("mfcc??: ", mfcc.shape)
        # print("mel??: ", mel.shape)
        egmaps = extract_egmaps_from_file(signal)
        pitch = extract_pitch_from_file(signal)

            

            # save features
        np.savez_compressed(path_to_save,mel=mel,egmaps=egmaps,pitch=pitch,label=label,mfcc=mfcc)
    
    # out_dict = {}
    # out_dict['label'] = np.array(label_list)
    # out_dict['mel'] = np.array(mel_spec_list)
    # mel = np.array(mel_spec_list)
    # print("mel shape: ",mel.shape)
    # out_dict['pitch'] = np.log(np.array(pitch_list) +1e-8)
    # out_dict['egmaps'] = np.array(egmap_list)
    # return out_dict
    
    # return np.array(label_list),np.array(mel_spec_list),np.array(egmap_list),np.array(pitch_list)
        


def extract_features_from_files(input_dir,file_extension = ".wav",sr=16000,window_len=3.5,augment_factor =1):
    file_paths, labels = common.featch_files_labels(input_dir)

    # pitch_val = extract_pitch_from_file(audio)
    # egmaps_val = extract_egmaps_from_file(audio)
    # mel_spec_val = extract_mel_spectogram(audio)
    # window_len = 3.5
    # augment_factor = 1


    # n_samples = int(window_len * sr)
    # mel_spec_list = []
    # label_list = []
    # egmap_list = []
    # pitch_list = []
    # audio_list =[]
    jobs =[]
    print("Number of files: ",len(file_paths))
    # print("this is new !")
    # with ft.ProcessPoolExecutor(max_workers=8) as worker:
    for file,label in zip(file_paths,labels):
            # print("file: ",file)
            # print("label: ",label)
            # jobs.append(
            #     worker.submit(save_features,file,label)
            # )
            save_features(file,label)

    print("completed!")
    #     results = [worker.submit(extract_features,file,label) for file,label in zip(file_paths,labels)]
        
    #     for f in ft.as_completed(results):
    #         res_dict = f.result()    
    #         mel_spec_list.append(res_dict['mel'])
    #         label_list.append(res_dict['label'])
    #         egmap_list.append(res_dict['egmaps'])
    #         pitch_list.append(res_dict['pitch'])
    # for file,label in zip(file_paths,labels):
    #         signal = common.load_audio(file,sr=16000)

    #         n_windows = int(signal.shape[0] // n_samples) * augment_factor

    #         # print("n_windows: ",n_windows)

    #         for i in range(n_windows):
    #             audio_frames = signal[ int(n_samples * i / augment_factor):int(n_samples * (1 + i / augment_factor))]
    #             audio_list.append(audio_frames)
    #             label_list.append(label)
    #             mel_spec_list.append(extract_mel_spectogram(audio_frames))
    #             egmap_list.append(extract_egmaps_from_file(audio_frames))
    #             pitch_list.append(extract_pitch_from_file(audio_frames))

    # out_dict = {}
    # out_dict['label'] = np.array(label_list)
    # out_dict['mel'] = np.array(mel_spec_list)
    # out_dict['pitch'] = np.log(np.array(pitch_list) +1e-8)
    # out_dict['egmaps'] = np.array(egmap_list)
    # return out_dict
    # save 
    # np.savez_compressed(out_file_path,pitch = pitch_val,egmaps = egmaps_val)
