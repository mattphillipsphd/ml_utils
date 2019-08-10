"""
Various frequently-used model class definitions.
"""

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torchvision as tv

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

#### TODO ####
# Obviously should not be two different AlexNet versions! 

class alexnet_wrapper(tv.models.AlexNet):
    def __init__(self, feats_only=False, load_pretrained=True, num_classes=1000,
            **kwargs):
        super().__init__(**kwargs)
        self._feats_only = feats_only
        print("Features only: ", self._feats_only)
        if load_pretrained:
            print("Loading pretrained model")
            self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier_1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            )
        self.classifier_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def get_features(self, x):
        return self._get_features(x)

    def _get_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier_1(x)
        return x

    def forward(self, x):
        x = self._get_features(x)
        if not self.feats_only:
            x = self.classifier_2(x)
        return x

from torchvision.models.alexnet import model_urls as alexnet_model_urls
class alexnet_pret(tv.models.AlexNet):
    def __init__(self, feats_only=False, load_pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self._feats_only = feats_only
        print("Features only: ", self._feats_only)
        if load_pretrained:
            print("Loading pretrained model")
            self.load_state_dict(torch.utils.model_zoo.load_url(\
                    alexnet_model_urls["alexnet"]))

    def forward(self, x):
        if self._feats_only:
            return self._get_linear2(x)
        else:
            return super().forward(x)

    def get_features(self, x):
        return self._get_linear2(x)

    def _get_linear2(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier[:5](x)
        return x


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

