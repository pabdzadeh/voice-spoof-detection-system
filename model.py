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


# class Block(nn.Module):
#     def __init__(
#             self, in_channels, intermediate_channels, device, identity_downsample=None, stride=1
#     ):
#         super(Block, self).__init__()
#         self.expansion = 4
#         self.conv1 = nn.Conv2d(
#             in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
#         ).to(device)
#         self.bn1 = nn.BatchNorm2d(intermediate_channels).to(device)
#         self.conv2 = nn.Conv2d(
#             intermediate_channels,
#             intermediate_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False
#         ).to(device)
#         self.bn2 = nn.BatchNorm2d(intermediate_channels).to(device)
#         self.conv3 = nn.Conv2d(
#             intermediate_channels,
#             intermediate_channels * self.expansion,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=False
#         ).to(device)
#         self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion).to(device)
#         self.relu = nn.ReLU()
#         self.identity_downsample = identity_downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x.clone()
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#
#         if self.identity_downsample is not None:
#             identity = self.identity_downsample(identity)
#
#         x += identity
#         x = self.relu(x)
#         return x
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, image_channels, num_classes, device):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.device = device
#         self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
#         self.bn1 = nn.BatchNorm2d(64).to(device)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # Essentially the entire ResNet architecture are in these 4 lines below
#         self.layer1 = self._make_layer(
#             block, layers[0], intermediate_channels=64, stride=1, device=device
#         ).to(device)
#         self.layer2 = self._make_layer(
#             block, layers[1], intermediate_channels=128, stride=2, device=device
#         ).to(device)
#         self.layer3 = self._make_layer(
#             block, layers[2], intermediate_channels=256, stride=2, device=device
#         ).to(device)
#         self.layer4 = self._make_layer(
#             block, layers[3], intermediate_channels=512, stride=2, device=device
#         ).to(device)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * 4, num_classes).to(device)
#
#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#
#         return x
#
#     def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride, device):
#         identity_downsample = None
#         layers = []
#
#         # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
#         # we need to adapt the Identity (skip connection) so it will be able to be added
#         # to the layer that's ahead
#         if stride != 1 or self.in_channels != intermediate_channels * 4:
#             identity_downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_channels,
#                     intermediate_channels * 4,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False
#                 ),
#                 nn.BatchNorm2d(intermediate_channels * 4),
#             )
#
#         layers.append(
#             block(self.in_channels, intermediate_channels, device, identity_downsample, stride)
#         )
#
#         # The expansion size is always 4 for ResNet 50,101,152
#         self.in_channels = intermediate_channels * 4
#
#         # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
#         # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
#         # and also same amount of channels.
#         for i in range(num_residual_blocks - 1):
#             layers.append(block(self.in_channels, intermediate_channels, device))
#
#         return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        super(Model, self).__init__()
        self.frame_length = 320
        self.frame_hops = 160
        self.n_fft = 512
        self.target_sampling_rate = 16000
        self.device = device

        self.lfcc_dim = 60 * 3
        # self.mfcc_layer = torchaudio.transforms.MFCC(sample_rate=self.target_sampling_rate)
        self.spectrogram_layer = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.frame_length).to(
            device)
        self.lfcc_layer = feature_layers.LFCC(self.frame_length,
                                              self.frame_hops,
                                              self.n_fft,
                                              self.target_sampling_rate,
                                              self.lfcc_dim,
                                              with_energy=True,
                                              with_delta=True).to(device)

        self.resnet = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)
        self.resnet2 = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)
        self.mlp_layer1 = nn.Linear(num_classes * 2, 256).to(device)
        self.mlp_layer2 = nn.Linear(256, 256).to(device)
        self.mlp_layer3 = nn.Linear(256, 256).to(device)
        self.drop_out = nn.Dropout(0.5)

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

    def forward(self, x):
        x = x.to(self.device)
        spectrogram_features = self.spectrogram_layer(x)
        spectrogram_features = spectrogram_features.unsqueeze(1).float().to(self.device)
        spectrogram_hidden_features = self.resnet(spectrogram_features)

        # mfcc_features = self.mfcc_layer(x)
        # mfcc_features = mfcc_features.unsqueeze(1).float().to(self.device)
        # mfcc_hidden_features = self.resnet(mfcc_features)

        # cqcc_features = self.extract_cqcc(x)
        # cqcc_features = cqcc_features.unsqueeze(1).float().to(self.device)
        # cqcc_hidden_features = self.resnet(cqcc_features)

        lfcc_features = self.lfcc_layer(x)
        lfcc_features = lfcc_features.unsqueeze(1).float().to(self.device)
        lfcc_hidden_features = self.resnet2(lfcc_features)

        # x = torch.cat((spectrogram_hidden_features, mfcc_hidden_features), 1)
        # x = torch.cat((spectrogram_hidden_features, cqcc_hidden_features), 1)
        x = torch.cat((spectrogram_hidden_features, lfcc_hidden_features), 1)

        x = F.relu(self.mlp_layer1(x))
        self.drop_out(x)
        x = F.relu(self.mlp_layer2(x))
        self.drop_out(x)
        feat = x
        x = F.relu(self.mlp_layer3(x))

        return feat, x

# def ResNet50(img_channel=1, num_classes=1000):
#     return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)
#
#
# def ResNet101(img_channel=1, num_classes=1000):
#     return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)
#
#
# def ResNet152(img_channel=1, num_classes=1000):
#     return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)
#
#
# def test():
#     net = ResNet101(img_channel=3, num_classes=1000)
#     # y = net(torch.randn(4, 3, 224, 224)).to("cuda")
#     print(y.size())
