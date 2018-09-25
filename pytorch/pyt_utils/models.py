"""
Various frequently-used model class definitions.
"""

import logging
import torch
import torch.nn as nn
import torchvision as tv


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

