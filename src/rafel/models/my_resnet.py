import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch


__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, padding=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, dilation, normalize, freeze_layers, freeze_bn):
        self.normalize = normalize
        self.freeze_bn = freeze_bn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # Freeze all layers but last one
        if freeze_layers:
            # TODO Check if same
            # for m in self.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         for p in m.parameters():
            #             p.requires_grad = False
            for p in self.parameters():
                p.requires_grad = False

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        elif output_stride == 8:
            assert dilation == 1 or dilation == 2, 'Dilation value not permitted'
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=dilation, padding=dilation)
        else:
            raise NotImplementedError('Output_stride not implemented')

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Freeze Batch Normalization
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()  # TODO Needed?
                    for p in m.parameters():
                        p.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation, padding=padding))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = (x - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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
            if isinstance(module, nn.BatchNorm2d) and self.freeze_bn:
                module.eval()
            else:
                module.train(mode)
        return self


def resnet50(pretrained=True, output_stride=16, dilation=1, normalize=True, freeze_layers=False, freeze_bn=True):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output_stride: 8 or 16
        dilation: 1 or 2
        normalize: Normalize input images
        freeze_layers: Freeze all but last layer
        freeze_bn: Freeze batch normalization layers
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, dilation, normalize, freeze_layers, freeze_bn)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=True, output_stride=16, dilation=1, normalize=True, freeze_layers=False, freeze_bn=True):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output_stride: 8 or 16
        dilation: 1 or 2
        normalize: Normalize input images
        freeze_layers: Freeze all but last layer
        freeze_bn: Freeze batch normalization layers
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, dilation, normalize, freeze_layers, freeze_bn)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=True, output_stride=16, dilation=1, normalize=True, freeze_layers=False, freeze_bn=True):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output_stride: 8 or 16
        dilation: 1 or 2
        normalize: Normalize input images
        freeze_layers: Freeze all but last layer
        freeze_bn: Freeze batch normalization layers
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, dilation, normalize, freeze_layers, freeze_bn)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
