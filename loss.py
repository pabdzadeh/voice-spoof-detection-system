import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class OCAngleLayer(nn.Module):
    """ Output layer to produce activation for one-class softmax

    Usage example:
     batchsize = 64
     input_dim = 10
     class_num = 2

     l_layer = OCAngleLayer(input_dim)
     l_loss = OCSoftmaxWithLoss()

     data = torch.rand(batchsize, input_dim, requires_grad=True)
     target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
     target = target.to(torch.long)

     scores = l_layer(data)
     loss = l_loss(scores, target)

     loss.backward()
    """

    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1

        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        # self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations

        input:
        ------
          input tensor (batchsize, input_dim)

        output:
        -------
          tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        # w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, 1)
        inner_wx = input.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)

        #
        return pos_score, neg_score


class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()

    """

    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()

    def forward(self, inputs, target):
        """
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from OCAngle
                 inputs[0]: positive class score
                 inputs[1]: negative class score
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # Assume target is binary, positive = 1, negaitve = 0
        #
        # Equivalent to select the scores using if-elese
        # if target = 1, use inputs[0]
        # else, use inputs[1]
        output = inputs[0] * target.view(-1, 1) + \
                 inputs[1] * (1 - target.view(-1, 1))
        loss = self.m_loss(output).mean()

        return loss


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels, is_train=True):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            is_train: check if we are in in train mode.
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)


        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        if is_train:
            scores[labels == 0] = self.r_real - scores[labels == 0]
            scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, -output_scores.squeeze(1)


class AMSoftmax(nn.Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits
