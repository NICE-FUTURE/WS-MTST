import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activate(self.bn(self.conv(x)))
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B,D,H,W --> B,HW,D
        x = self.proj(x)
        return x


class DecoderSegFormer(nn.Module):
    """Decoder for SegFormer."""

    def __init__(
        self,
        feature_strides=[],
        in_channels=[],
        embedding_dim=256,
        num_classes=0,
        out_size=None,
        **kwargs
    ):
        super(DecoderSegFormer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.out_size = out_size

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        n, _, h, w = c4.shape

        # B,D,H,W --> B,HW,D' --> B,D',H,W --> B,D',H,W
        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = F.interpolate(
            _c4, size=self.out_size, mode="bilinear", align_corners=False
        )

        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = F.interpolate(
            _c3, size=self.out_size, mode="bilinear", align_corners=False
        )

        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = F.interpolate(
            _c2, size=self.out_size, mode="bilinear", align_corners=False
        )

        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )
        _c1 = F.interpolate(
            _c1, size=self.out_size, mode="bilinear", align_corners=False
        )

        # B,D',H,W --> B,4D',H,W--> B,D',H,W
        logit = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(logit)

        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)

        x = self.linear_pred(x)  # B,D',H0,W0 --> B,C,H0,W0

        return x


class DecoderCNN(nn.Module):
    """Decoder for SegFormer."""

    def __init__(
        self,
        input_feature_dim,
        embedding_dim=256,
        num_classes=0,
        out_size=None,
        **kwargs
    ):
        super(DecoderCNN, self).__init__()
        self.num_classes = num_classes
        self.out_size = out_size
        self.inner_size = 64

        self.linear_c1 = MLP(input_feature_dim, embedding_dim)
        self.linear_c2 = MLP(input_feature_dim, embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        B, _, h, w = x.shape

        x1 = self.linear_c1(x).permute(0, 2, 1).reshape(B, -1, h, w)
        x1 = F.interpolate(
            x1, size=self.inner_size, mode="bilinear", align_corners=False
        )

        x2 = self.linear_c2(x).permute(0, 2, 1).reshape(B, -1, h, w)
        x2 = F.interpolate(
            x2, size=self.inner_size, mode="bilinear", align_corners=False
        )

        # B,D',H,W --> B,4D',H,W--> B,D',H,W
        logit = self.linear_fuse(torch.cat([x1, x2], dim=1))
        x = self.dropout(logit)

        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)

        x = self.linear_pred(x)  # B,D',H0,W0 --> B,C,H0,W0

        return x
