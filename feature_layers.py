import torch.nn as nn
import torch
import tools.audio_utils
import librosa
import numpy as np


class CQT(nn.Module):
    def __init__(self, sampling_rate):
        super(CQT, self).__init__()
        self.sampling_rate = sampling_rate

    def forward(self, x, device):
        batch_size = x.shape[0]
        batch_output = torch.zeros(batch_size, 84, 126)
        batch_count = 0
        for item in x:
            numpy_item = item.numpy()
            item_cqt = librosa.cqt(numpy_item, sr=self.sampling_rate)
            item_cqt = librosa.amplitude_to_db(np.abs(item_cqt), ref=np.max)
            item_torch_cqt = torch.from_numpy(item_cqt).to(device)
            batch_output[batch_count] = item_torch_cqt
            batch_count += 1

        return batch_output.to(device)


class Spectrogram(nn.Module):
    def __init__(self, n_fft):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft

    def forward(self, x, device):
        batch_size = x.shape[0]
        batch_output = torch.zeros(batch_size, 1025, 126)
        batch_count = 0
        for item in x:
            numpy_item = item.numpy()
            item_stft = librosa.stft(numpy_item, n_fft=self.n_fft)
            item_stft = librosa.amplitude_to_db(np.abs(item_stft), ref=np.max)
            item_torch_stft = torch.from_numpy(item_stft).to(device)
            batch_output[batch_count] = item_torch_stft
            batch_count += 1

        return batch_output.to(device)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = tools.audio_utils.dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = tools.audio_utils.idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = tools.audio_utils.dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = tools.audio_utils.idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


class LFCC(nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """

    def __init__(self, fl, fs, fn, sr, filter_num,
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_LFB=False):
        """ Initialize LFCC

        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank
          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta

          for_LFB: bool (default False), reserved for LFB feature
        """
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num

        # build the triangle filter bank
        f = (sr / 2) * torch.linspace(0, 1, fn // 2 + 1)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([fn // 2 + 1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = tools.audio_utils.trimf(
                f, [filter_bands[idx],
                    filter_bands[idx + 1],
                    filter_bands[idx + 2]])
        self.lfcc_fb = nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')

        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_LFB = flag_for_LFB
        return

    def forward(self, x):
        """

        input:
        ------
         x: tensor(batch, length), where length is waveform length

        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]

        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl,
                            window=torch.hamming_window(self.fl).to(x.device),
                            onesided=True, pad_mode="constant")
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) +
                                 torch.finfo(torch.float32).eps)

        # DCT (if necessary, remove DCT)
        lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature

        # Add energy
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2) +
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = tools.audio_utils.delta(lfcc)
            lfcc_delta_delta = tools.audio_utils.delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output
