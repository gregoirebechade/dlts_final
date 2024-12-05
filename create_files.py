


import os
import numpy as np
from scipy.io import wavfile
import tqdm
from scipy.signal import stft, istft


# conversion en npy : 
convert_files = True  
if convert_files: 
    print('begin creation')
    test_files_noise = os.listdir('./data/Audio/denoising/test/test/')
    train_files_noise = os.listdir('./data/Audio/denoising/train/train/')
    test_files_clean = os.listdir('./data/Audio/voice_origin/test/test/')
    train_files_clean = os.listdir('./data/Audio/voice_origin/train/train/')
    np.random.seed(0)
    for i in tqdm.tqdm(range(len(test_files_noise))):
        file = test_files_noise[i]
        samplerate, data = wavfile.read('./data/Audio/denoising/test/test/' + file)
        frequencies, times, Zxx = stft(data, fs=samplerate, nperseg=500)
        np.save('./data/spectrogrammes/test/noisy/' + file + '.npy', Zxx)

    for i in tqdm.tqdm(range(len(test_files_clean))):
        file = test_files_noise[i]
        samplerate, data = wavfile.read('./data/Audio/voice_origin/test/test/' + file)
        frequencies, times, Zxx = stft(data, fs=samplerate, nperseg=500)
        np.save('./data/spectrogrammes/test/origin/' + file + '.npy', Zxx)
        
    for i in tqdm.tqdm(range(len(train_files_noise))):
        file = train_files_noise[i]
        samplerate, data = wavfile.read('./data/Audio/denoising/train/train/' + file)
        samplerate_clean, data_clean=wavfile.read('./data/Audio/voice_origin/train/train/' + file)
        frequencies, times, Zxx = stft(data, fs=samplerate, nperseg=500)
        frequencies_clean, times_clean, Zxx_clean = stft(data_clean, fs=samplerate_clean, nperseg=500)
        if np.random.random() > 0.8 :
            np.save('./data/spectrogrammes/validation/noisy/' + file + '.npy', Zxx)
            np.save('./data/spectrogrammes/validation/origin/' + file + '.npy', Zxx_clean)
        else : 
            np.save('./data/spectrogrammes/train/noisy/' + file + '.npy', Zxx)
            np.save('./data/spectrogrammes/train/origin/' + file + '.npy', Zxx_clean)
