from __future__ import annotations
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

import network.torchvision_vit as vision_transformer


class MAEDecoder(vision_transformer.Encoder):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes:
        seq_length:
            Token sequence length, including the class token.
        num_layers:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        embed_input_dim:
            Dimension of the input tokens. Usually be equal to the hidden
            dimension of the MAEEncoder or MAEBackbone.
        hidden_dim:
            Dimension of the decoder tokens.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        out_dim:
            Output dimension of the prediction for a single patch. Usually equal
            to (3 * patch_size ** 2).
        dropout:
            Percentage of elements set to zero after the MLP in the transformer.
        attention_dropout:
            Percentage of elements set to zero after the attention head.

    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns predicted pixel values from encoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim).

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim).

        """
        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: torch.Tensor) -> torch.Tensor:
        """Embeds encoded input tokens into decoder token dimension.

        This is a single linear layer that changes the token dimension from
        embed_input_dim to hidden_dim.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim)
                containing the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the embedded tokens.

        """
        return self.decoder_embed(input)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        return super().forward(input)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Predics pixel values from decoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        return self.prediction_head(input)
