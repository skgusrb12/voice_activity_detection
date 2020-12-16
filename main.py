import os
from args import parser
from multiprocessing import freeze_support

from create_images import creating_data
from train import train_model
from predict import prediction


if __name__ == '__main__':

    # multiprocessing
    freeze_support()
    args = parser.parse_args()

    mode = args.mode

    # if you want to implement this code,
    # write to 'python main.py --mode prediction'

    if mode == 'creating_data':

        params = {
            'raw_data': args.training_speech_dir,
            'labels': args.label_dir,
            'save_dir': args.features_dir,
            'test_img_ratio': args.test_img_ratio,
            'parallel': args.parallel,
        }

        creating_data(params)

        print('creating the data')

    elif mode == 'training':

        params = {
            'path': args.features_dir,
            'model_path': args.model_dir,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'n_cnn_filters': [int(x) for x in args.n_filters.split('-')],
            'n_cnn_kernels': [int(x) for x in args.n_kernels.split('-')],
            'n_fc_units': [int(x) for x in args.n_fc_units.split('-')],
            'n_classes': args.n_classes,
            'train_val_ratio': args.train_val_ratio,
            'baseline_val_loss': args.baseline_val_loss,
        }

        train_model(params)

        print('model training complete')

    elif mode == 'prediction':

        params = {
            'test_data': args.test_speech_dir,
            'model_path': os.path.join(args.model_dir, 'vad_model.pt'),
            'smoothing': args.smoothing,
            'visualize': args.visualize,
            'parallel': args.parallel,
            'fig_path': args.fig_path
        }

        prediction(params)

        print('prediction complete')