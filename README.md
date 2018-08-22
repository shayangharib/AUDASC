# AUDASC -- Adversarial Unsupervised Domain Adaptation for Acoustic Scene Classification

### Welcome to the repository of the AUDASC method. 

Full description of the method along with obtained 
results can be found at the corresponding [paper](https://arxiv.org/abs/1808.05777). The paper is submitted to the [Detection
and Classification of Acoustic Scenes and Events workshop (DCASE)](http://dcase.community/) 2018.

Pre-trained weights together with extracted features and labels are at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164585.svg)](https://zenodo.org/record/1401995#.W31Zaxx9iK4)

This project is implemented using [PyTorch](https://pytorch.org/)

## How to set up AUDASC

### Clone repository and setup project

### Dependencies
If you are going to use the feature extraction part of this repository you need the following libraries:
```
LibROSA
PySoundFile
scikit-learn
```
To run the core method, following is necessary:
```
Torch
```
## Extract and pre-process features
* To extract the features, you can use data_pre_processing/feature_extractor.py.
* To prepare the features and labels as input to the model, you can use data_pre_processing/data_preprocessing.py

To change the setup of both files, you can use data_pre_processing/feature_params.yml

### Training and test procedures
The setup of pre-training, adaptation, and test can be assigned and changed using scripts/learning_params.yml
### Pre_training and adaptation
To train the model on source domain data and adapt the pre-trained model, you need to run 
scripts/training.py

### Test the adapted model
To test the adapted as well as non adapted model on source and target data, you simply need to run 
scripts/test.py

