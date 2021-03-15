# voice_activity_detection (VAD)
>Hyeon Kyu Lee : skgusrb12@gmail.com
>
>Coworker : https://github.com/Jihackstory (email : dlwlgkr159@gmail.com)
>
>This project is Pytorch version of this project : [Voice Activity Detection based on Deep Learning & TensorFlow](https://github.com/filippogiruzzi/voice_activity_detection)

## 1. Introduction

**To detect real-time voice based on korean speech, this project was utilized in 2020 Artificial Intelligence (AI) Grand Challenge Speech Recognition(Track2) (http://www.ai-challenge.kr/sub0101).**

The main purpose of this competition is to design deep learning algorithms that detect five class of korean violence speech(threat, extortion, workplace harassment, out-of-work harassment, and nonviolence). 
In our secenario, we used this project to improve the performance of Speech to Text (STT). 

The project consists of three modes: `creating_data`, `training` and `prediction`.

## 2. Download Dataset

We used the LibriSpeech ASR corpus dataset(https://openslr.org/12/) at sampling rate 16kHz consist of approximately 1000 hours and English speech from audiobook.

The `test-clean` in website was used as training dataset in our project and the label can download [here](https://drive.google.com/drive/folders/1ZPQ6wnMhHeE7XP5dqpAEmBAryFzESlin) provided by the [reference project](https://github.com/filippogiruzzi/voice_activity_detection).

In other words, in this project, the english voice dataset was used in the training step, and the korean voice dataset was used in the test step. 

## 3. Run this project

As mentioned above, this project is comprised of three modes : `creating_data`, `training` and `prediction`.

**Run script (Command Line)**

```python
python main.py --mode prediction
# if you want to run training mode : --mode training
```

### 3.1 Creating_data 

This mode is used to create input image data including the four features(MFCC, MFCC-△, MFCC-△2, RMSE of mel-spectogram) of VAD model.\

**The parameter lists and defaults of this mode**
```
- raw_data : raw data (e.g. LibriSpeech ASR corpus dataset)
- labels : labels of voice and non-voice
- save_dir : the file for saving the input images
- test_img_ratio : the ratio of test data from total input data
- parallel : using the cpu parallel processing, 'True' and 'False'
```

### 3.2 Training 
```
- path : the path to input dataset
- model_path : pretrained model directory
- batch_size : batch size of training (?, 16, 65)
- epochs : Nb of epochs for training
- learning_rate : learning rate
- n_cnn_filters : 32-64-128
- n_cnn_kernels : 8-5-3
- n_fc_units : 2048-2048
- n_classes : Nb of classes
- train_val_ratio : ratio dividing Nb of training dataset and Nb of validation dataset
- baseline_val_loss : early stopping parameter to stop the training
```

### 3.3 Prediction 
```
- test_data :
- model_path :
- smoothing :
- visualize :
- parallel :
- fig_path :
```

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
