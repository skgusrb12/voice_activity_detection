import os
import librosa
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split


def split_data(file_list, test_ratio=0, random_seed=0):
    np.random.seed(random_seed)
    np.random.shuffle(file_list)

    num_test = int(len(file_list) * test_ratio)
    num_train = len(file_list) - num_test

    train_fn = file_list[:num_train]
    test_fn = file_list[num_train:]

    return train_fn, test_fn


def load_data(data_dir, label_dir, file_list):

    data = []
    label = []

    for fn in file_list:
        data_path = fn.split('-')
        data_fn = os.path.join(data_dir, data_path[0], data_path[1], '{}.flac'.format(fn))

        try:
            signal, sampling_rate = librosa.load(data_fn, sr=16000)
        except RuntimeError:
            print('!!! Skipped signal !!!')
            continue

        seg_data, seg_label = data_segmention(signal, label_dir, fn)

        data.extend(seg_data)
        label.extend(seg_label)

    return data, label


def data_segmention(signal, label_dir, fn, seg_len=1024):
    label_fn = os.path.join(label_dir, '{}.json'.format(fn))
    with open(label_fn, 'r') as f:
        label = json.load(f)

    seg_data = []
    seg_label = []

    start = 0
    end = seg_len

    for speech_interval in label['speech_segments']:
        next_start, next_end = speech_interval['start_time'], speech_interval['end_time']

        while end < next_start:
            seg_data.append(signal[start:end])
            seg_label.append(0)

            start += seg_len
            end += seg_len


        seg_data.append(signal[next_start:next_end])
        seg_label.append(1)

        start = next_start + seg_len
        end = next_end + seg_len


    while end < len(signal):
        seg_data.append(signal[start:end])
        seg_label.append(0)

        start += seg_len
        end += seg_len

    return seg_data, seg_label


def extract_features(signal, freq=16000, n_mfcc=5, size=512, step=16, n_mels=40):

    # Mel Frequency Cepstral Coefficents
    mfcc = librosa.feature.mfcc(y=signal, sr=freq, n_mfcc=n_mfcc, n_fft=size, hop_length=step)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Root Mean Square Energy
    mel_spectogram = librosa.feature.melspectrogram(y=signal, n_mels=n_mels, sr=freq, n_fft=size, hop_length=step)

    rmse = librosa.feature.rms(S=mel_spectogram, frame_length=n_mels*2-1, hop_length=step)

    mfcc = np.asarray(mfcc)
    mfcc_delta = np.asarray(mfcc_delta)
    mfcc_delta2 = np.asarray(mfcc_delta2)
    rmse = np.asarray(rmse)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
    features = np.transpose(features)

    return features


def prepare_input_img(data, label, train_val_ratio, batch_size):

    x_tensor = torch.from_numpy(data).float()
    y_tensor = torch.from_numpy(label).long()

    dataset = TensorDataset(x_tensor, y_tensor)

    train_len = int(len(dataset)*train_val_ratio)
    val_len = len(dataset) - train_len

    # split the training and validation dataset
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def check_length(train_loader, num_epochs):

    num_total_batch = len(train_loader)
    str_total_batch = len(str(num_total_batch))

    str_epochs = len(str(num_epochs))

    return num_total_batch, str_total_batch, str_epochs