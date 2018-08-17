"""
Model class definitions for the pretrain module.
"""

import logging
import torch
import torch.nn as nn
import torchvision as tv

################################################################################
# ResNet definition from pytorch/vision/torchvision/models/resnet.py for
# reference
#class ResNet(nn.Module):
#
#    def __init__(self, block, layers, num_classes=1000):
#        self.inplanes = 64
#        super(ResNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                               bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.relu = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#        self.layer1 = self._make_layer(block, 64, layers[0])
#        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#        self.avgpool = nn.AvgPool2d(7, stride=1)
#        self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#
#    def _make_layer(self, block, planes, blocks, stride=1):
#        downsample = None
#        if stride != 1 or self.inplanes != planes * block.expansion:
#            downsample = nn.Sequential(
#                nn.Conv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(planes * block.expansion),
#            )
#
#        layers = []
#        layers.append(block(self.inplanes, planes, stride, downsample))
#        self.inplanes = planes * block.expansion
#        for i in range(1, blocks):
#            layers.append(block(self.inplanes, planes))
#
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)
#
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#
#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#
#        return x
#
#def resnet18(pretrained=False, **kwargs):
#    """Constructs a ResNet-18 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#    return model
#
#
#def resnet34(pretrained=False, **kwargs):
#    """Constructs a ResNet-34 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#    return model
#
#
#def resnet50(pretrained=False, **kwargs):
#    """Constructs a ResNet-50 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#    return model

################################################################################

class resnet18_pret(tv.models.ResNet):
    def __init__(self, feats_only=False, load_pretrained=True, **kwargs):
        super(resnet18_pret, self).__init__(tv.models.resnet.BasicBlock,
                [2, 2, 2, 2], **kwargs)
        self._feats_only = feats_only
        print("Features only: ", self._feats_only)
        if load_pretrained:
            print("Loading pretrained model")
            self.load_state_dict(torch.utils.model_zoo.load_url(
                tv.models.resnet.model_urls["resnet18"]))

    def get_features(self, x):
        return self._get_avgpool(x)

    def _get_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, x):
        if self._feats_only:
            return self._get_avgpool(x)
        else:
            return super(resnet18_pret,self).forward(x)


class resnet34_pret(tv.models.ResNet):
    def __init__(self, feats_only=False, load_pretrained=True, **kwargs):
        super(resnet34_pret, self).__init__(tv.models.resnet.BasicBlock,
                [3, 4, 6, 3], **kwargs)
        self._feats_only = feats_only
        if load_pretrained:
            self.load_state_dict(torch.utils.model_zoo.load_url(
                tv.models.resnet.model_urls["resnet34"]))

    def get_features(self, x):
        return self._get_avgpool(x)

    def _get_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, x):
        if self._feats_only:
            return self._get_avgpool(x)
        else:
            return super(resnet34_pret,self).forward(x)

class resnet50_pret(tv.models.ResNet):
    def __init__(self, feats_only=False, load_pretrained=True, **kwargs):
        super(resnet50_pret, self).__init__(tv.models.resnet.Bottleneck,
                [3, 4, 6, 3], **kwargs)
        self._feats_only = feats_only
        if load_pretrained:
            self.load_state_dict(torch.utils.model_zoo.load_url(
                tv.models.resnet.model_urls["resnet50"]))

    def get_features(self, x):
        return self._get_avgpool(x)

    def _get_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, x):
        if self._feats_only:
            return self._get_avgpool(x)
        else:
            return super(resnet50_pret,self).forward(x)

