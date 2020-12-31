# voice_activity_detection
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

In other words, in this project, the english voice dataset was used in the training step, and the test step is performed with korean voice dataset. 

## 3. Run this project


## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
