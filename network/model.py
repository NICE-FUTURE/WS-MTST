import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .decoders import DecoderSegFormer, DecoderCNN
from . import encoders


AVAILABLE_MODELS = [
    "mit-b1",
    "mit-b3",
    "mit-b5",
    "resnet50",
    "inception_v3",
    "efficientnet_b4",
]


def create_encoder(arch, stride, num_channels, pretrained):
    if arch in ("mit-b1", "mit-b3", "mit-b5"):
        pretrained_name = "{}.pth".format(arch.replace("-", "_"))
        encoder = getattr(encoders, arch.replace("-", "_"))(
            stride=stride, in_chans=num_channels
        )
        feature_dims = encoder.embed_dims
        if pretrained:
            state_dict = torch.load("pretrained/" + pretrained_name, map_location="cpu")
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head")
            }
            if num_channels != 3:  # To be compatible with 4-channel inputs
                weight = state_dict["patch_embed1.proj.weight"]
                padding = torch.rand(
                    (weight.shape[0], 1, weight.shape[2], weight.shape[3])
                )
                weight = torch.cat([weight, padding], dim=1)
                state_dict["patch_embed1.proj.weight"] = weight
            encoder.load_state_dict(state_dict)
        return encoder, feature_dims
    elif arch in ("resnet50", "inception_v3", "efficientnet_b4"):
        encoder = timm.create_model(arch, pretrained=pretrained, in_chans=num_channels)
        out_feature_dim = encoder.num_features
        encoder.forward = encoder.forward_features
        return encoder, out_feature_dim
    else:
        raise NotImplementedError


def create_decoder(
    arch, feature_dims, feature_strides, decoder_dim, num_classes, out_size
):
    if arch in ("mit-b1", "mit-b3", "mit-b5"):
        # output 4 channels, include background
        assert isinstance(feature_dims, list), "feature_dims is not a list"
        decoder = DecoderSegFormer(
            feature_strides=feature_strides,
            in_channels=feature_dims,
            embedding_dim=decoder_dim,
            num_classes=num_classes,
            out_size=out_size,
        )
        return decoder
    elif arch in ("resnet50", "inception_v3", "efficientnet_b4"):
        assert isinstance(feature_dims, int), "feature_dims is not a integer"
        decoder = DecoderCNN(
            feature_dims,
            embedding_dim=decoder_dim,
            num_classes=num_classes,
            out_size=out_size,
        )
        return decoder
    else:
        raise NotImplementedError


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=0,
        decoder_dim=256,
        stride=[4, 2, 2, 1],
        pretrained=None,
        pooling="gmp",
        num_channels=3,
        out_size=None,
    ):
        super().__init__()
        # Attention: `self.num_classes` exclude background, but `num_classes` include background
        self.num_classes = num_classes - 1
        self.decoder_dim = decoder_dim
        self.feature_strides = [4, 8, 16, 32]

        self.encoder, feature_dims = create_encoder(
            backbone, stride, num_channels, pretrained
        )
        self.decoder = create_decoder(
            backbone,
            feature_dims,
            self.feature_strides,
            self.decoder_dim,
            num_classes,
            out_size,
        )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d
        # output 3 channels, exclude background
        out_feature_dim = (
            feature_dims[-1] if isinstance(feature_dims, list) else feature_dims
        )
        self.classifier = nn.Conv2d(
            in_channels=out_feature_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            bias=False,
        )

    def get_param_groups(self):
        param_groups = [
            [],
            [],
            [],
            [],
        ]  # backbone; backbone_norm; cls_head; decode_head;
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        param_groups[2].append(self.classifier.weight)
        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, cam_only=False, with_cam=False, register_hook=False):
        _x = self.encoder(x, register_hook=register_hook)
        if isinstance(_x, list):
            _x1, _x2, _x3, _x4 = _x  # B,D,H,W
        else:
            _x4 = _x

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4

        cls = self.pooling(_x4, (1, 1))
        cls = self.classifier(cls)
        cls = cls.view(-1, self.num_classes)

        if register_hook:
            return cls

        decode = self.decoder(_x)

        if with_cam:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cls, decode, cam_s4
