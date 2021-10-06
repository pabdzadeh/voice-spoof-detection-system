import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import evaluation_metrics as em
import numpy as np
from model import Model
from loss import *
import librosa
import torchvision
from torch import Tensor

from tools.dataset_loader import ASVDataset


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def test_model(model_path, device, batch_size):
    transforms = torchvision.transforms.Compose([
        lambda x: pad(x),
        lambda x: librosa.util.normalize(x),
        lambda x: Tensor(x)
    ])

    model = Model(input_channels=1, num_classes=256, device=device).to(device)
    oc_softmax = OCSoftmax(256, r_real=0.9, r_fake=0.2, alpha=20).to(device)

    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    oc_softmax.load_state_dict(torch.load(loss_model_path))

    test_set = ASVDataset(is_train=False, is_eval=True, is_eval2021=False, transform=transforms)
    test_set_2021 = ASVDataset(is_train=False, is_eval=True, is_eval2021=True, transform=transforms)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_data_loader_2021 = DataLoader(test_set_2021, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    oc_softmax.eval()

    with open('./scores/cm_score.txt', 'w') as cm_score_file:
        for batch_x, batch_y, batch_meta in test_data_loader:
            batch_x = batch_x.to(device)
            labels = batch_y.to(device)

            feats, outputs = model(batch_x)

            _, score = oc_softmax(feats, labels)

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s %s\n' % (batch_meta.file_name[j],
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    with open('./scores/cm_score_2021.txt', 'w') as cm_score_file_2021:
        for batch_x, batch_y, batch_meta in test_data_loader_2021:
            print('processing..', end="\r")
            batch_x = batch_x.to(device)

            labels = batch_y.to(device)

            feats, outputs = model(batch_x)

            _, score = oc_softmax(feats, labels, is_train=False)

            for j in range(labels.size(0)):
                cm_score_file_2021.write('%s %s\n' % (batch_meta.file_name[j], score[j].item()))

    #     eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join('', './scores/checkpoint_cm_score.txt'),
    #                                             "./scores/")
    #     return eer_cm, min_tDCF
    return


def test(model_path, device, batch_size):
    model_path = os.path.join(model_path)
    print(test_model(model_path, device, batch_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model-path', type=str, help="path to the trained model", default="./models/")
    parser.add_argument('-b', '--batch-size', type=str, help="path to the trained model", default="32")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_path, device, args.batch_size)
