# voice-spoof-detection-system
A voice spoofing detection system, based on our used system in ASVSpoof2021

We are using CQT spectrogram as input and a ResNet-18 with self-attention for feature extraction.
A three Layer MLP is used for classification and for a better discrimination of genuine samples from fake ones we use One Class Softmax.
