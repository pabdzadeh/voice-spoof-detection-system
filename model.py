from resnet import *
from loss import *
import nnAudio.Spectrogram as torch_spec
from torchaudio import transforms


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        super(Model, self).__init__()

        self.device = device
        self.cqt = torch_spec.CQT(output_format='Complex').to(device)
        self.amp_to_db = transforms.AmplitudeToDB()
        self.resnet = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)

        self.mlp_layer1 = nn.Linear(num_classes, 256).to(device)
        self.mlp_layer2 = nn.Linear(256, 256).to(device)
        self.mlp_layer3 = nn.Linear(256, 256).to(device)
        self.drop_out = nn.Dropout(0.5)

        self.oc_softmax_angle = OCAngleLayer(in_planes=256)
        self.oc_softmax_loss = OCSoftmaxWithLoss()

        # self.oc_softmax = OCSoftmax(256).to(device)

    def compute_score(self, feature_vec, angle=False):
        # compute a-softmax output for each feature configuration
        batch_size = feature_vec.shape[0]

        # negaitve class scores
        x_cos_val = torch.zeros(
            [feature_vec.shape[0], 1],
            dtype=feature_vec.dtype, device=feature_vec.device)

        # positive class scores
        x_phi_val = torch.zeros_like(x_cos_val)

        s_idx = 0
        e_idx = batch_size
        tmp1, tmp2 = self.oc_softmax_angle(feature_vec[s_idx:e_idx], angle)
        x_cos_val[s_idx:e_idx] = tmp1
        x_phi_val[s_idx:e_idx] = tmp2

        return [x_cos_val, x_phi_val]

    def forward(self, x, labels, is_train=True):
        x = x.to(self.device)
        x = self.cqt(x)
        x = torch.pow(x[:, :, :, 0], 2) + torch.pow(x[:, :, :, 1], 2)
        x = self.amp_to_db(x)

        x = self.resnet(x.unsqueeze(1).float().to(self.device))

        x = F.relu(self.mlp_layer1(x))
        self.drop_out(x)
        x = F.relu(self.mlp_layer2(x))
        self.drop_out(x)
        x = F.relu(self.mlp_layer3(x))
        feat = x

        scores = self.compute_score(feat, not is_train)
        loss = self.oc_softmax_loss(scores, labels)

        return loss, scores[0]
