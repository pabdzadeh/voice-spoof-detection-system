import torch
import argparse
import os
import time
import numpy as np
import json

from torch import Tensor
import torchvision
import librosa
from model import Model
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import tools.dataset_loader as dataset_loader
from sklearn.model_selection import KFold

from resnet import setup_seed
from collections import defaultdict
from loss import *
import evaluation_metrics as em


class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def add_parser(parser):
    # parser = argparse.ArgumentParser(description=__doc__)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')

    # Dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=750)
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=4, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")
    parser.add_argument('--epoch', type=int, default=0, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=598)

    parser.add_argument('--add_loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss for one-class training")
    parser.add_argument('--weight_loss', type=float, default=1, help="weight for other loss")
    parser.add_argument('--r_real', type=float, default=0.9, help="r_real for ocsoftmax")
    parser.add_argument('--r_fake', type=float, default=0.2, help="r_fake for ocsoftmax")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax")

    parser.add_argument('--model_path', type=str, help="saved model path")
    parser.add_argument('--loss_model_path', type=str, help="saved loss model path")

    parser.add_argument('--continue_training', action='store_true',
                        help="continue training with previously trained model")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        assert os.path.exists(args.out_fold)
    else:
        # Path for output data
        if not os.path.exists('./log/'):
            os.makedirs('./log/')

        # Save training arguments
        with open(os.path.join('./log/', 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join('./log/', 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join('./log/', 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def k_fold_cross_validation(k_fold, train_set, batch_size):
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(train_set)):
        # Print
        # Sample elements randomly from a given list of ids, no replacement.
        train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sub_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader_part = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, sampler=train_sub_sampler)
        validation_loader_part = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, sampler=test_sub_sampler)

    return train_loader_part, validation_loader_part


def train(parser, device):
    print(f'{Color.OKGREEN}Loading  train dataset...{Color.ENDC}')
    args = parser.parse_args()
    model = Model(input_channels=1, num_classes=256, device=device)
    oc_softmax = OCSoftmax(args.enc_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(device)

    transforms = torchvision.transforms.Compose([
        lambda x: pad(x),
        lambda x: librosa.util.normalize(x),
        lambda x: Tensor(x)
    ])

    k_fold = KFold(n_splits=5, shuffle=True)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        oc_softmax.load_state_dict(torch.load(args.loss_model_path))
        print('Model loaded : {}'.format(args.model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    train_set = dataset_loader.ASVDataset(is_train=True, transform=transforms)

    number_of_epochs = args.num_epochs
    monitor_loss = args.add_loss

    model.train()
    oc_softmax.train()
    oc_softmax_optimizer = torch.optim.SGD(oc_softmax.parameters(), lr=args.lr)

    print(f'{Color.ENDC}Train Start...')

    for epoch in range(args.epoch, number_of_epochs):
        start = time.time()

        print(f'{Color.OKBLUE}Epoch:{epoch}{Color.ENDC}')
        train_loader, validation_loader = k_fold_cross_validation(k_fold, train_set, batch_size=args.batch_size)
        model.train()
        train_loss_dict = defaultdict(list)
        dev_loss_dict = defaultdict(list)

        adjust_learning_rate(args, optimizer, epoch)
        adjust_learning_rate(args, oc_softmax_optimizer, epoch)

        for batch_x, batch_y, batch_meta in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            labels = batch_y.to(device)
            feats, outputs = model(batch_x)

            oc_softmax_loss, _ = oc_softmax(feats, labels)
            loss = oc_softmax_loss * args.weight_loss
            optimizer.zero_grad()
            oc_softmax_optimizer.zero_grad()
            train_loss_dict[args.add_loss].append(oc_softmax_loss.item())
            loss.backward()
            optimizer.step()
            oc_softmax_optimizer.step()

            with open(os.path.join('./log/', 'train_loss.log'), 'a') as log:
                log.write(str(epoch) + "\t" + "\t" +
                          str(np.nanmean(train_loss_dict[monitor_loss])) + "\n")

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        print('start validation phase...')

        # Val the model
        model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (batch_x, batch_y, batch_meta) in enumerate(validation_loader):
                labels = batch_y.to(device)
                feats, outputs = model(batch_x)

                oc_softmax_loss, score = oc_softmax(feats, labels)

                dev_loss_dict[args.add_loss].append(oc_softmax_loss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_val_eer = em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            val_eer = min(val_eer, other_val_eer)

            with open(os.path.join('./log/', "dev_loss.log"), "a") as log:
                log.write(str(epoch) + "\t" + str(np.nanmean(dev_loss_dict[monitor_loss])) + "\t" + str(
                    val_eer) + "\n")
            print("Val EER: {}".format(val_eer))

        torch.save(model.state_dict(), os.path.join('./models/', 'model_%d.pt' % (epoch + 1)))
        loss_model = oc_softmax
        torch.save(loss_model.state_dict(), os.path.join('./models/', 'loss_model_%d.pt' % (epoch + 1)))
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser('ASVSpoof2021')
    add_parser(parser)
    train(parser, device)


if __name__ == '__main__':
    main()

# writer.add_scalar('train_accuracy', train_accuracy, epoch)
# writer.add_scalar('valid_accuracy', validation_accuracy, epoch)
# writer.add_scalar('loss', running_loss, epoch)
