import torch
import collections
import os
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed

LOGICAL_DATA_ROOT = './data/'

ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """

    def __init__(self,
                 transform=None,
                 is_train=True,
                 is_eval=False,
                 is_eval2021=False
                 ):
        super(ASVDataset, self).__init__()

        data_root = LOGICAL_DATA_ROOT

        self.is_eval = is_eval

        self.data_root = data_root

        self.dset_name = 'eval2021' if is_eval2021 else 'eval' if is_eval else 'train' if is_train else 'train'

        self.protocols_fname = os.path.join(self.data_root, self.dset_name + '.protocol.txt')

        self.files_dir = os.path.join(self.data_root, '{}'.format(self.dset_name))
        self.transform = transform

        self.files_meta = self.parse_protocols_file(self.protocols_fname)
        self.data_files = os.listdir(self.files_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        idx = idx % len(self.files_meta)
        meta = self.files_meta[idx]
        data_x, data_y, _ = self.read_file(meta)
        data_x = self.transform(data_x)
        x = data_x
        y = data_y
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')

        if self.is_eval:
            return ASVFile(speaker_id='',
                           file_name=tokens[0],
                           path=os.path.join(self.files_dir, tokens[0] + '.flac'),
                           sys_id=0,
                           key=0)
        return ASVFile(speaker_id=tokens[0],
                       file_name=tokens[1],
                       path=os.path.join(self.files_dir, tokens[1] + '.flac'),
                       sys_id=0,
                       key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)
