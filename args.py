import argparse


parser = argparse.ArgumentParser(description='Voice Activity Detection (VAD) using pytorch')

# mode to run the program
parser.add_argument('--mode', default='prediction', type=str, choices=['creating_data', 'training', 'prediction'])

# a set of dataset paths
parser.add_argument('--training-speech-dir', type=str, default='./dataset/raw_data/data', help='path where to find training speech data')
parser.add_argument('--test-speech-dir', type=str, default='./dataset/test_data', help='path where to find test speech data')
parser.add_argument('--label-dir', type=str, default='./dataset/raw_data/labels', help='path where to find segmented label of raw data')
parser.add_argument('--features-dir', type=str, default='./dataset/features', help='path where to save the feature and to load that')
parser.add_argument('--model-dir', type=str, default='./model', help='pretrained model directory')
parser.add_argument('--fig-path', type=str, default='./result/', help='ratio dividing Nb of training dataset and Nb of test dataset')


# hyperparameters of creating the input images
parser.add_argument('--parallel', type=bool, default=False, help='using the cpu parallel processing')
parser.add_argument('--test_img_ratio', type=float, default=0.1, help='ratio dividing Nb of training dataset and Nb of test dataset')

# hyperparameters of model
parser.add_argument('--train_val_ratio', type=float, default=0.8, help='ratio dividing Nb of training dataset and Nb of validation dataset')
parser.add_argument('--baseline_val_loss', type=float, default=0.01, help='early stopping parameter to stop the training')
parser.add_argument('--batch-size', '-bs', type=int, default=32, help='batch size of training: (?, 16, 65)')
parser.add_argument('--epochs', '-e', type=int, default=20, help='Nb of epochs for training')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.00001, help='learning rate for training')
parser.add_argument('--n-filters', type=str, default='32-64-128')
parser.add_argument('--n-kernels', type=str, default='8-5-3')
parser.add_argument('--n-fc-units', type=str, default='2048-2048')
parser.add_argument('--n-classes', '-n', type=int, default=2, help='the number of classes')

# hyperparameters of creating the input images
parser.add_argument('--smoothing', type=bool, default=True, help='using the cpu parallel processing')
parser.add_argument('--visualize', type=bool, default=False, help='ratio dividing Nb of training dataset and Nb of test dataset')