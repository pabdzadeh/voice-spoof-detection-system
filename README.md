# voice-spoof-detection-system
This is the implementation of our work titled "A Countermeasure Based on CQT Spectrogram for Deepfake Speech Detection" was presented in the 7th International Conference on Signal Processing and Intelligent Systems (ICSPIS), Dec 2021. 

We are using CQT spectrogram as input and a ResNet-18 with self-attention for feature extraction.
A three Layer MLP is used for classification and for a better discrimination of genuine samples from fake ones we use One Class Softmax.

Some part of the codes are borrowed from https://github.com/yzyouzhang/AIR-ASVspoof and https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts.
