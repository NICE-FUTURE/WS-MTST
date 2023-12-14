""" Collate Functions """

from typing import List, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from utils import ZeroOneNormalization


class BaseCollateFunction(nn.Module):
    """Base class for other collate implementations.

    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.

    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.

    """

    def __init__(self, transform: torchvision.transforms.Compose):
        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                A batch of tuples of images, labels, and filenames which
                is automatically provided if the dataloader is built from
                a LightlyDataset.

        Returns:
            A tuple of images, labels, and filenames. The images consist of
            two batches corresponding to the two transformations of the
            input images.

        Examples:
            >>> # define a random transformation and the collate function
            >>> transform = ... # some random augmentations
            >>> collate_fn = BaseCollateFunction(transform)
            >>>
            >>> # input is a batch of tuples (here, batch_size = 1)
            >>> input = [(img, 0, 'my-image.png')]
            >>> output = collate_fn(input)
            >>>
            >>> # output consists of two random transforms of the images,
            >>> # the labels, and the filenames in the batch
            >>> (img_t0, img_t1), label, filename = output

        """
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)
        ]
        # list of labels
        labels = torch.LongTensor(np.array([item[1] for item in batch]))
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0),
        )

        return transforms, labels, fnames


class ImageCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for images.

    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.

    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings.

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
    """

    def __init__(
        self,
        input_size: int = 64,
        min_scale: float = 0.15,
        kernel_size: float = 0.1,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
    ):
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        assert 0.0 < kernel_size < 1.0
        kernel_size = int(kernel_size * input_size_)
        if kernel_size % 2 == 0:
            kernel_size += 1

        transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=input_size, scale=(min_scale, 1.0), antialias=True
                ),
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                T.GaussianBlur(kernel_size=kernel_size),
                ZeroOneNormalization(),
            ]
        )

        super(ImageCollateFunction, self).__init__(transform)


class SimCLRCollateFunction(ImageCollateFunction):
    """Implements the transformations for SimCLR.

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
    """

    def __init__(
        self,
        input_size: int = 224,
        min_scale: float = 0.08,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        kernel_size: float = 0.1,
    ):
        super(SimCLRCollateFunction, self).__init__(
            input_size=input_size,
            min_scale=min_scale,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            kernel_size=kernel_size,
        )


class MoCoCollateFunction(ImageCollateFunction):
    """Implements the transformations for MoCo v1.
    For MoCo v2, simply use the SimCLR settings.
    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
    Examples:
        >>> # MoCo v1 for ImageNet
        >>> collate_fn = MoCoCollateFunction()
        >>>
        >>> # MoCo v1 for CIFAR-10
        >>> collate_fn = MoCoCollateFunction(
        >>>     input_size=32,
        >>> )
    """

    def __init__(
        self,
        input_size: int = 224,
        min_scale: float = 0.2,
        kernel_size: float = 0.1,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
    ):
        super(MoCoCollateFunction, self).__init__(
            input_size=input_size,
            min_scale=min_scale,
            kernel_size=kernel_size,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
        )


class MultiViewCollateFunction(nn.Module):
    """Generates multiple views for each image in the batch.
    Attributes:
        transforms:
            List of transformation functions. Each function is used to generate
            one view of the back.
    """

    def __init__(self, transforms: List[torchvision.transforms.Compose]):
        super().__init__()
        self.transforms = transforms

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.
        Args:
            batch:
                The input batch.
        Returns:
            A (views, labels, fnames) tuple where views is a list of tensors
            with each tensor containing one view of the batch.
        """
        views = []
        for transform in self.transforms:
            view = torch.stack([transform(img) for img, _, _ in batch])
            views.append(view)
        # list of labels
        labels = torch.LongTensor(
            np.concatenate([label for _, label, _ in batch], axis=0)
        )
        # list of filenames
        fnames = [fname for _, _, fname in batch]
        return views, labels, fnames


class MAECollateFunction(MultiViewCollateFunction):
    """Implements the view augmentation for MAE [0].
    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
    ):
        transforms = [
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3, antialias=True
            ),  # 3 is bicubic # type: ignore
            T.RandomHorizontalFlip(),
        ]

        super().__init__([T.Compose(transforms)])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
        return views[0], labels, fnames
