import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
import feature_layers
from CQCC2.cqcc import cqcc
import math
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, \
    meshgrid, ceil, linspace
from CQCC2.CQT_toolbox_2013.cqt import cqt
from scipy.signal import lfilter
from scipy.interpolate import interpn
from scipy.fft import dct
from resnet import *
from loss import *
import librosa


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        super(Model, self).__init__()
        # self.frame_length = 320
        # self.frame_hops = 160
        self.n_fft = 2048
        self.target_sampling_rate = 16000
        self.device = device

        self.lfcc_dim = 60 * 3

        self.cqt_layer = feature_layers.CQT(sampling_rate=self.target_sampling_rate)
        self.spectrogram_layer = feature_layers.Spectrogram(n_fft=self.n_fft).to(
            device)

        # self.lfcc_layer = feature_layers.LFCC(self.frame_length,
        #                                       self.frame_hops,
        #                                       self.n_fft,
        #                                       self.target_sampling_rate,
        #                                       self.lfcc_dim,
        #                                       with_energy=True,
        #                                       with_delta=True).to(device)

        self.resnet = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)
        self.resnet2 = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)
        self.mlp_layer1 = nn.Linear(num_classes, 256).to(device)
        self.mlp_layer2 = nn.Linear(256, 256).to(device)
        self.mlp_layer3 = nn.Linear(256, 256).to(device)
        self.drop_out = nn.Dropout(0.5)
        self.oc_softmax = OCSoftmax(256).to(device)

    # feature extraction functions
    def cqccDeltas(x, hlen=2):
        win = list(range(hlen, -hlen - 1, -1))
        norm = 2 * (arange(1, hlen + 1) ** 2).sum()
        xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
        xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
        xx = concatenate([xx_1, x, xx_2], axis=-1)
        D = lfilter(win, 1, xx) / norm
        return D[:, hlen * 2:]

    def extract_cqcc(self, sig, fs=16000, fmin=96, fmax=8000, B=12, cf=19, d=16):
        # cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
        kl = B * math.log(1 + 1 / d, 2)
        gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))
        eps = 2.2204e-16
        scoeff = 1

        new_fs = 1 / (fmin * (2 ** (kl / B) - 1))
        ratio = 9.562 / new_fs

        Xcq = cqt(sig[:, None], B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
        absCQT = abs(Xcq['c'])

        TimeVec = arange(1, absCQT.shape[1] + 1).reshape(1, -1)
        TimeVec = TimeVec * Xcq['xlen'] / absCQT.shape[1] / fs

        FreqVec = arange(0, absCQT.shape[0]).reshape(1, -1)
        FreqVec = fmin * (2 ** (FreqVec / B))

        LogP_absCQT = log(absCQT ** 2 + eps)

        n_samples = int(ceil(LogP_absCQT.shape[0] * ratio))
        Ures_FreqVec = linspace(FreqVec.min(), FreqVec.max(), n_samples)

        xi, yi = meshgrid(TimeVec[0, :], Ures_FreqVec)
        Ures_LogP_absCQT = interpn(points=(TimeVec[0, :], FreqVec[0, :]), values=LogP_absCQT.T, xi=(xi, yi),
                                   method='splinef2d')

        CQcepstrum = dct(Ures_LogP_absCQT, type=2, axis=0, norm='ortho')
        CQcepstrum_temp = CQcepstrum[scoeff - 1:cf + 1, :]
        deltas = self.cqccDeltas(CQcepstrum_temp.T).T

        CQcc = concatenate([CQcepstrum_temp, deltas, self.cqccDeltas(deltas.T).T], axis=0)

        return CQcc.T

    def forward(self, x, labels):
        x = x.to(self.device)

        cqt_features = self.cqt_layer(x, self.device)
        # spectrogram_features = self.spectrogram_layer(x, self.device)

        # spectrogram_features = spectrogram_features.unsqueeze(1).float().to(self.device)
        # spectrogram_hidden_features = self.resnet(spectrogram_features)

        # mfcc_features = self.mfcc_layer(x)
        # mfcc_features = mfcc_features.unsqueeze(1).float().to(self.device)
        # mfcc_hidden_features = self.resnet(mfcc_features)

        # cqcc_features = self.extract_cqcc(x)
        # cqcc_features = cqcc_features.unsqueeze(1).float().to(self.device)
        # cqcc_hidden_features = self.resnet(cqcc_features)

        # lfcc_features = self.lfcc_layer(x)
        # lfcc_features = lfcc_features.unsqueeze(1).float().to(self.device)
        # lfcc_hidden_features = self.resnet2(lfcc_features)

        # x = torch.cat((spectrogram_hidden_features, mfcc_hidden_features), 1)
        # x = torch.cat((spectrogram_hidden_features, cqcc_hidden_features), 1)
        # x = torch.cat((spectrogram_hidden_features, lfcc_hidden_features), 1)

        # features = torch.cat((cqt_features, spectrogram_features), 1)
        features = cqt_features
        x = self.resnet(features.unsqueeze(1).float().to(self.device))

        x = F.relu(self.mlp_layer1(x))
        self.drop_out(x)
        x = F.relu(self.mlp_layer2(x))
        self.drop_out(x)
        x = F.relu(self.mlp_layer3(x))
        feat = x
        
        return self.oc_softmax(feat, labels)