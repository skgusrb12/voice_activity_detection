import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import multiprocessing
import pickle
import torch
from data_tools import extract_features


def smoothing_v1(label):

    smoothed_label = []
    # Smooth with 3 consecutive windows
    for i in range(2, len(label), 3):
        cur_pred = label[i]
        if cur_pred == label[i - 1] == label[i - 2]:
            smoothed_label.extend([cur_pred, cur_pred, cur_pred])
        else:
            if len(smoothed_label) > 0:
                smoothed_label.extend([smoothed_label[-1], smoothed_label[-1], smoothed_label[-1]])
            else:
                smoothed_label.extend([0, 0, 0])

    n = 0
    while n < len(smoothed_label):
        cur_pred = smoothed_label[n]
        if cur_pred == 1:
            if n > 0:
                smoothed_label[n - 1] = 1
            if n < len(smoothed_label) - 1:
                smoothed_label[n + 1] = 1
            n += 2
        else:
            n += 1

    for idx in range(len(label) - len(smoothed_label)):
        smoothed_label.append(smoothed_label[-1])

    return smoothed_label


def smoothing_v2(label):

    smoothed_label = []
    for i in range(2, len(label)):
        cur_pred = label[i]
        if cur_pred == label[i - 1] == label[i - 2]:
            smoothed_label.append(cur_pred)
        else:
            if len(smoothed_label) > 0:
                smoothed_label.append(smoothed_label[-1])
            else:
                smoothed_label.append(0)

    n = 0
    while n < len(smoothed_label):
        cur_pred = smoothed_label[n]
        if cur_pred == 1:
            if n > 0:
                smoothed_label[n - 1] = 1
            if n < len(smoothed_label) - 1:
                smoothed_label[n + 1] = 1
            n += 2
        else:
            n += 1

    for idx in range(len(label) - len(smoothed_label)):
        smoothed_label.append(smoothed_label[-1])

    return smoothed_label


def visualize(signal, label, fig_path, fn='predict_VAD'):

    sr = 16000
    fig = plt.figure(figsize=(15, 10))
    sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / sr for i in range(len(signal))], signal)
    for idx, predictions in enumerate(label):
        color = 'r' if predictions == 0 else 'g'
        ax.axvspan((idx * 1024) / sr, ((idx+1) * 1024) / sr, alpha=0.5, color=color)
    plt.title('Prediction on signal {}, speech in green'.format(fn), size=20)
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.xticks(size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()
    fig.savefig(fig_path + fn + '.png', dpi=fig.dpi)
    plt.close(fig)

# check the speech interval (SI)
def check_si(label):

    length = 1024
    interval = [[0, 0]]
    pre_label = 0

    for idx in range(len(label)):

        if label[idx]:
            if pre_label == 1:
                interval[-1][1] = (idx + 1) * length
            else:
                interval.append([idx * length, (idx + 1) * length])

        pre_label = label[idx]

    return interval[1:]


# extract only speech interval (SI) to ust in Speech to Text (STT)
def extract_si(data, label):

    interval = check_si(label)
    speech = []
    for start, end in interval:
        speech.append(data[start:end])

    return speech


def prediction(params):

    data_path = params['test_data']
    model_path = params['model_path']
    is_smoothing = params['smoothing']
    is_visualize = params['visualize']
    is_parallel = params['parallel']
    fig_path = params['fig_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path)
    data_list = os.listdir(data_path)

    p = multiprocessing.Pool(multiprocessing.cpu_count()-16)

    speech_interval = []
    total_time = 0

    # using GPU
    model.eval()
    for idx in range(len(data_list)):

        feature_list, frames = [], []
        start = time.time()
        file_n = data_list[idx]
        signal, _ = librosa.load(os.path.join(data_path, file_n), sr=16000)

        start_idx, end_idx = 0, 1024

        while end_idx < len(signal):
            frames.append(signal[start_idx:end_idx])

            start_idx = end_idx
            end_idx += 1024

        if is_parallel:
            feature = p.map(extract_features, frames)
        else:
            feature = [extract_features(frames[idx]) for idx in range(len(frames))]

        feature = np.array(feature)
        feature = np.expand_dims(feature, 1)
        test_input = torch.from_numpy(feature).float().to(device)

        # prediction for the test dataset
        val_output = model(test_input)
        _, predict_label = torch.max(val_output.data, 1)

        labels = predict_label.to(torch.device('cpu')).detach().numpy()

        if is_smoothing:
            labels = smoothing_v1(labels)
            # labels = smoothing_v2(labels)

        # For input of STT (in our scenario)
        # if you do not need it, remove it
        speech_interval.append(extract_si(signal, labels))

        # inference time check
        single_time = time.time() - start
        total_time += single_time

        if is_visualize:
            visualize(signal, predict_label, fig_path, fn=file_n)

        print('idx : ', str(idx), 'inference time : ', single_time)

    print('======================================')
    print('\ntotal inference time : ', total_time)
    p.close()
    p.join()

    with open('./result/VAD_output.pickle', 'wb') as f:
        pickle.dump(speech_interval, f)

    return speech_interval
