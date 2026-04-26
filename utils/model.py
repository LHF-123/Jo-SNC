# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : model.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 15:54
#   @Description :
#
# ================================================================
import torch
import torch.nn as nn
import torchvision
from utils.utils import init_weights


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(256, n_outputs)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19BN_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = torchvision.models.vgg19_bn(pretrained=pretrained)
        self.feature_encoder = vgg.features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.feat_dim = 512

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.avg_pool(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_scale_factor, projection_size, init_method='He', activation='relu', use_bn=True):
        super().__init__()

        mlp_hidden_size = round(mlp_scale_factor * in_channels)
        if activation == 'relu':
            non_linear_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky relu':
            non_linear_layer = nn.LeakyReLU(inplace=True)
        elif activation == 'tanh':
            non_linear_layer = nn.Tanh()
        else:
            raise AssertionError(f'{activation} is not supported yet.')
        mlp_head_module_list = [nn.Linear(in_channels, mlp_hidden_size)]
        if use_bn:
            mlp_head_module_list.append(nn.BatchNorm1d(mlp_hidden_size))
        mlp_head_module_list.append(non_linear_layer)
        mlp_head_module_list.append(nn.Linear(mlp_hidden_size, projection_size))

        self.mlp_head = nn.Sequential(*mlp_head_module_list)
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)


def build_pre_act_resnet18_encoder():
    # PreAct ResNet is an optional architecture; delay the import so ResNet50
    # evaluation does not require utils/pre_act_resnet.py to exist.
    try:
        from utils.pre_act_resnet import PreActResNet18Encoder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'preact_resnet18 was requested, but utils/pre_act_resnet.py is missing.'
        ) from exc
    return PreActResNet18Encoder()


def build_inception_resnet_v2_encoder():
    # Inception-ResNet-v2 depends on timm; import it only when that backbone is
    # selected so ResNet/VGG/CNN runs do not require the optional dependency.
    try:
        from utils.InceptionResNetV2 import HF_InceptionResNetV2_Encoder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'InceptionResNetV2 was requested, but its optional timm dependency is missing.'
        ) from exc
    return HF_InceptionResNetV2_Encoder(pretrained_weight_index=0)


class Encoder(nn.Module):
    def __init__(self, arch='cnn', num_classes=200, pretrained=True):
        super().__init__()
        if arch.startswith('resnet') and arch in torchvision.models.__dict__.keys():
            resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = resnet.fc.in_features
        elif arch.startswith('preact_resnet18'):
            pre_act_resnet = build_pre_act_resnet18_encoder()
            self.encoder = pre_act_resnet.encoder
            self.feature_dim = 512
        elif arch == 'InceptionResNetV2':
            self.encoder = build_inception_resnet_v2_encoder()
            self.feature_dim = self.encoder.feat_dim
        elif arch == 'vgg19_bn':
            self.encoder = VGG19BN_Encoder(pretrained=pretrained)
            self.feature_dim = self.encoder.feat_dim
        elif arch.startswith('cnn'):
            cnn = CNN(input_channel=3, n_outputs=num_classes)
            self.encoder = nn.Sequential(*list(cnn.children())[:-1])
            self.feature_dim = cnn.classifier.in_features
        else:
            raise AssertionError(f'{arch} is not supported!')

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


class Model(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, mlp_hidden=2, pretrained=True):
        super().__init__()
        self.encoder = Encoder(arch, num_classes, pretrained)
        assert mlp_hidden > 0, f'{mlp_hidden} is negative and invalid mlp_hidden.'
        if (mlp_hidden - 0) < 1e-6:
            self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        else:
            self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DualHeadModel(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, mlp_hidden=2, feature_dim=128, pretrained=True, use_bn=True, simsiam=False):
        super().__init__()
        self.simsiam = simsiam
        self.encoder = Encoder(arch, num_classes, pretrained)
        assert mlp_hidden >= 0, f'{mlp_hidden} is negative and invalid mlp_hidden.'
        if (mlp_hidden - 0) < 1e-6:
            self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        else:
            self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes, use_bn=use_bn)
        if simsiam:
            feature_dim = self.encoder.feature_dim
        self.projector = MLPHead(self.encoder.feature_dim, 1, feature_dim, use_bn=use_bn)

    def forward(self, x):
        fea = self.encoder(x)
        logits = self.classifier(fea)
        feat_c = self.projector(fea)
        if self.simsiam:
            return logits, nn.functional.normalize(feat_c, dim=1), nn.functional.normalize(fea, dim=1)
        else:
            return logits, nn.functional.normalize(feat_c, dim=1)
