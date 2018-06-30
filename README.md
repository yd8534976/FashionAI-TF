# FashionAI Global Challenge: Key points Detection
- Code for [FashionAI KeyPoint Detection](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.4ccc289bCzDJXu&raceId=231648&_lang=en_US)
- Score of this code : 4.25%
- Team ranked 32/2322
- Mainly base on [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319).

## Folder Structure
- `nets`: store modified ResNet
- `model`: store checkpoint files
- `outputs` : store predicted files
- `summary`: store files for tensorboard
- `train_set`: place training data here
- `fashion_evaluator.py`: evaluator script
- `fashion_generator.py`: data generator
- `fashion_helper.py`: store a bunch of helper functions
- `fashion_stacked.py`: main file to define model
- `train_script.py`: train script
- `test_script.py`: test script

## Prerequisites
### Docker is recommended:
- nvidia-docker
- pull image from Docker Hub\
`docker pull yd8534976/tf-aiden`

### Alternatively:
- python
- tensorflow-gpu (>= 1.4)
- numpy
- pandas
- opencv-python

### Pre-trained ResNet50 model
 - You can download pre-trained models from tensorflow offical
  [slim model zoo](https://github.com/tensorflow/models/tree/master/research/slim).
 - Put checkpoint files into model/ .