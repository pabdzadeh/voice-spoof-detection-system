from resnet import *
from loss import *
import nnAudio.Spectrogram as torch_spec
from torchaudio import transforms


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        super(Model, self).__init__()

        self.device = device
        self.cqt = torch_spec.CQT(output_format='Complex', sr=16000).to(device)
        self.amp_to_db = transforms.AmplitudeToDB()
        self.resnet = ResNet(3, 256, resnet_type='18', nclasses=256).to(device)

        self.mlp_layer1 = nn.Linear(num_classes, 256).to(device)
        self.mlp_layer2 = nn.Linear(256, 256).to(device)
        self.mlp_layer3 = nn.Linear(256, 256).to(device)
        self.drop_out = nn.Dropout(0.5)

        self.oc_softmax = OCSoftmax(256).to(device)

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


        return self.oc_softmax(feat, labels, is_train)
