import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import evaluation_metrics as em
import numpy as np

from tools.dataset_loader import ASVDataset


def test_model(model_path, loss_model_path, part, add_loss, device):
    dir_path = './data/'
    model = torch.load(model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None

    test_set = ASVDataset(is_train=False, is_eval=True, is_eval2021=False)
    test_set_2021 = ASVDataset(is_train=False, is_eval=True, is_eval2021=False)
    test_data_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    test_data_loader_2021 = DataLoader(test_set_2021, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    loss_model.eval()

    with open('./scores/checkpoint_cm_score.txt', 'w') as cm_score_file:
        for batch_x, batch_y, batch_meta in test_data_loader:
            batch_x = batch_x.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = batch_y.to(device)

            feats, outputs = model(batch_x)

            _, score = loss_model(feats, labels)

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s %s\n' % (tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    with open('./scores/checkpoint_cm_score_2021.txt', 'w') as cm_score_file_2021:
        for batch_x, batch_y, batch_meta in test_data_loader_2021:
            batch_x = batch_x.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = batch_y.to(device)

            feats, outputs = model(batch_x)

            _, score = loss_model(feats, labels, is_train=False)

            for j in range(labels.size(0)):
                cm_score_file_2021.write(
                    '%s %s %s\n' % (tags[j].data,
                                    "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                    score[j].item()))

    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
                                            "./data/scores/")
    return eer_cm, min_tDCF


def test(model_dir, add_loss, device):
    model_path = os.path.join(model_dir, "model.pt")
    loss_model_path = os.path.join(model_dir, "loss_model.pt")
    test_model(model_path, loss_model_path, "eval", add_loss, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/")
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.loss, device)
