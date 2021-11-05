

# NPC-diagnosis-based-on-deep-learnig

This is a PyTorch implementation of the paper   'Computer Aided Pathological Diagnosis of Nasopharyngeal Carcinoma Based on Deep Learning',  and We'll refine the data and code over time.


![overview](https://github.com/SH-Diao123/NPC-diagnosis-based-on-deep-learnig/blob/master/assets/overview.png)

## 🤝 Authorization 
If you would like to use our data, please contact us first and obtain authorization to use it.


## 🤝 Citation

If you find this code is useful for your research, please consider citing:

```javascript 
@article{
title={Computer Aided Pathological Diagnosis of Nasopharyngeal Carcinoma Based on  Deep Learning}，
author={Songhui Diao, Jiaxin Hou, Hong Yu, Xia Zhao, Yikang Sun, Ricardo Lewis Lambo, Yaoqin Xie, Lei Liu, Weiren Luo, Wenjian Qin}，
journal={The American Journal of Pathology}，
year={2020},
}
```

## Setup
### Prerequisites
- PyTorch 1.0
- python 3.6.4
- Torchvision 0.4
- numpy and so on
### Data
- train data
- validation data
- test data

## Training
Training a network with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within experiments/models and experiments/logs respectively after starting training.
If conditions permit, it will be better to pre-train the model first.
```javascript 
python main.py
```

## Validation and Testing
You can run validation and testing on the checkpointed best model by:
```javascript 
python test.py
```









