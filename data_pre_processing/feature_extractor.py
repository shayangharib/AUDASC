#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import glob
import pickle

import os
from os import path

import scipy
from scipy.signal.windows import *

import soundfile as sf
import librosa

__author__ = 'Shayan Gharib -- TUT'


def spectrogram(y, n_fft=1024, win_length=0.04, hop_length=0.02, window=scipy.signal.hamming(1024, sym=False),
                center=True, spectrogram_type='magnitude'):
    """

    :param y: audio signal
    :param n_fft: length of the FFT window
    :param win_length: the length of used window
    :param hop_length: number of samples between successive frames
    :param window: window type to be used for framing the audio
    :param center:If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`
    :param spectrogram_type: calculating the power or magnitude of spectrogram
    :return: D[f, t] is the magnitude/power of frequency bin f at frame t
    """
    eps = np.spacing(1)

    if spectrogram_type == 'magnitude':
        return np.abs(librosa.stft(y + eps,
                                   n_fft=n_fft,
                                   win_length=win_length,
                                   hop_length=hop_length,
                                   center=center,
                                   window=window))

    elif spectrogram_type == 'power':
        return np.abs(librosa.stft(y + eps,
                                   n_fft=n_fft,
                                   win_length=win_length,
                                   hop_length=hop_length,
                                   center=center,
                                   window=window)) ** 2
    else:
        raise ValueError('Unknown spectrum type [%s]' % spectrogram_type)


def main():
    
    """Main function of the feature_extractor script.

    Loading the audio files and extracting the mel band features.
    """
    with open(path.join('feature_params.yml'), 'r') as f:
        feature_params = yaml.load(f)

    audio_file_dir = feature_params['general']['audio_files_dir']

    feature_set_dir = feature_params['general']['feature_set_dir']

    if not os.path.exists(path.join(feature_set_dir)):
        os.makedirs(path.join(feature_set_dir))

    # defining the initial parameters for extracting the features
    smpl_rate = feature_params['feature']['sampling_rate']
    n_fft_ = feature_params['feature']['n_fft']
    win_length = feature_params['feature']['win_length']
    hop_length = feature_params['feature']['hop_length']
    n_mel_bands = feature_params['feature']['n_mel_bands']
    f_min = feature_params['feature']['f_min']
    f_max = feature_params['feature']['f_max']
    htk_ = feature_params['feature']['htk_']
    spectrogram_type = feature_params['feature']['spectrogram_type']
    window_name = feature_params['feature']['window_name']
    symmetric = feature_params['feature']['symmetric']

    if window_name in scipy.signal.windows.__all__:
        win_func = eval(window_name)
    else:
        raise ValueError('Unknown window function')

    window = win_func(win_length, sym=symmetric)

    mel_basis = librosa.filters.mel(sr=smpl_rate, n_fft=n_fft_, n_mels=n_mel_bands,
                                    fmin=f_min, fmax=f_max, htk=htk_)

    audio_data_path = glob.glob(os.path.join(audio_file_dir, '*.wav'))

    eps = np.spacing(1)

    for ind, smpl in enumerate(audio_data_path):

        wav_smpl, sr_ = sf.read(smpl, always_2d=True)

        # to converts wav_smpl[samples, channels] to wav_smpl[channels, samples]
        wav_smpl = wav_smpl.T

        feature_matrix_container = []
        stat_container = []

        for channel in range(wav_smpl.shape[0]):
            spectrogram_ = spectrogram(y=wav_smpl[channel, :],
                                       n_fft=n_fft_,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       spectrogram_type=spectrogram_type,
                                       center=True,
                                       window=window)

            feature_matrix = np.dot(mel_basis, spectrogram_)

            feature_matrix = np.log(feature_matrix + eps)

            feature_matrix_container.append(feature_matrix)

            stat_container.append({
                        'mean': np.mean(feature_matrix.T, axis=0),
                        'std': np.std(feature_matrix.T, axis=0)
            })

        pickle.dump({
            'feat': feature_matrix_container,
            'stat': stat_container
        }, open(path.join(feature_set_dir, smpl.split('/')[-1].replace('.wav', '.p')), "wb"), protocol=2)

    with open(path.join(feature_set_dir, 'feature_set_params.yaml'), 'w') as f:
        yaml.dump(
            {
                'sample rate': smpl_rate,
                'minimum frequency': f_min,
                'maximum frequency': f_max,
                'window': window_name,
                'window length': win_length,
                'hop length': hop_length,
                'fft length': n_fft_,
                'mel bands': n_mel_bands,
                'spectrogram type': spectrogram_type,
                'htk': htk_
            }, f, default_flow_style=True
        )


if __name__ == '__main__':
    main()
