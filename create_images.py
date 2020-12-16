import os
import numpy as np
import multiprocessing
from data_tools import split_data, load_data, extract_features


def creating_data(params):

    file_list = [fn.split('.')[0] for fn in os.listdir(params['labels']) if 'json' in fn]

    train_fn, test_fn = split_data(file_list, params['test_img_ratio'])

    train_data, train_label = load_data(params['raw_data'], params['labels'], train_fn)
    test_data, test_label = load_data(params['raw_data'], params['labels'], test_fn)

    p = multiprocessing.Pool(multiprocessing.cpu_count()-8)

    print('start feature extraction')

    if params['parallel']:
        train_out = p.map(extract_features, train_data)
        test_out = p.map(extract_features, test_data)
        p.close()
        p.join()

    else:
        train_out = [extract_features(train_data[idx]) for idx in range(len(train_data))]
        test_out = [extract_features(test_data[idx]) for idx in range(len(test_data))]

    # save the input images and labels with extracted features
    np.save(os.path.join(params['save_dir'], 'train_imgs.npy'), train_out)
    np.save(os.path.join(params['save_dir'], 'train_labels.npy'), train_label)
    np.save(os.path.join(params['save_dir'], 'test_imgs.npy'), test_out)
    np.save(os.path.join(params['save_dir'], 'test_labels.npy'), test_label)

    print('create the input images and labels')