import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.autograd import Variable


class Encoder (nn.Module):
    def __init__(self, freeze_layers=False):
        super().__init__()

        myresnet = resnet.resnet50(pretrained=True)

        # resnet conv1
        self.conv1 = myresnet.conv1         # 1/2 64ch
        self.bn1 = myresnet.bn1
        self.relu = myresnet.relu

        # resnet conv2_x
        self.maxpool = myresnet.maxpool     # 1/4 64ch
        self.l1 = myresnet.layer1           # 1/4 256ch

        # resnet conv3_x
        self.l2 = myresnet.layer2           # 1/8 512ch

        if freeze_layers:
            for param in self.parameters():
                param.requires_grad = False

        # resnet conv4_x
        self.l3 = myresnet.layer3           # 1/16 1024ch

        # resnet conv5_x
        # self.l4 = myresnet.layer4         # 1/32 2048ch

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # As torchvision demands, images are normalized when using model zoo
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inp):
        # As torchvision demands, images are normalized when using model zoo
        inp = (inp - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # x = self.l4(x)
        return x

    def train(self, mode=True):
        # TODO check if necessary
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            else:
                module.train(mode)
        return self
