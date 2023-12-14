import torch
from lightly.models.utils import set_at_index


def mask_at_index(
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Copies mask token into the input tensor at the given indices.
    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        mask_token:
            Value tensor with shape (1, 1, dim).
    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.
    """
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)  # type: ignore
    B, L, _ = tokens.shape
    mask_token = mask_token.expand(B, L, -1)
    return (1 - mask) * tokens + mask * mask_token