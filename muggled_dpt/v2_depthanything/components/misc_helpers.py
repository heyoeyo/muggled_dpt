#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SpatialUpsampleLayer(nn.Module):
    """
    Simpler upscaling layer. Just a wrapper around the interpolation function.
    Note, there are no learnable parameters!
    (purpose is to allow for defining upsample layers within sequential models)
    """

    # .................................................................................................................

    def __init__(self, scale_factor: float = 2, interpolation_mode: str = "bilinear"):

        # Inherit from parent
        super().__init__()

        # Store layer config
        self._scale_factor = scale_factor
        self._mode = interpolation_mode

    # .................................................................................................................

    def forward(self, feature_map_2d: Tensor) -> Tensor:
        return nn.functional.interpolate(
            feature_map_2d, scale_factor=self._scale_factor, mode=self._mode, align_corners=True
        )

    # .................................................................................................................

    def extra_repr(self) -> str:
        return f"scale_factor={self._scale_factor}, interpolation_mode={self._mode}"

    # .................................................................................................................


class Conv3x3Layer(nn.Conv2d):
    """
    Helper class used to create 3x3 2D convolution layers.
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    """

    # .................................................................................................................

    def __init__(self, in_channels: int, out_channels: int | None = None, bias: bool = True):

        # Instantiate parent with some fixed arguments for 3x3 operation
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias, kernel_size=3, stride=1, padding=1)

    # .................................................................................................................


class Conv1x1Layer(nn.Conv2d):
    """
    Helper class used to create 1x1 2D convolution layers (i.e. depthwise convolution).
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    """

    # .................................................................................................................

    def __init__(self, in_channels: int, out_channels: int | None = None, bias: bool = True):

        # Instantiate parent with some fixed arguments for 1x1 operation
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias, kernel_size=1, stride=1, padding=0)

    # .................................................................................................................


class MLP2Layers(nn.Module):
    """
    Simplified implementation of the MLP model from the timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/23e7f177242e7516e8e3fc02ea1071b8cbc41ca8/timm/layers/mlp.py#L13

    This implementation removes most of the flexibility options, so that only the functionality used
    by the Depth-Anything implementation remains. Also removes training-related (i.e. dropout) components.

    This model is a simple feed-forward network, which is intended to be used at the end of each
    transformer block. Note that it defaults to including an 'expansion' type of hidden layer
    (i.e hidden layer has more features than input/output), based on feature_ratio input.
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_feature_ratio: float = 4, bias: bool = True):

        # Inherit from parent
        super().__init__()

        # Calculate number of hidden features
        num_hidden_features = int(round(hidden_feature_ratio * num_features))

        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_features, bias=bias),
            nn.GELU(),
            nn.Linear(num_hidden_features, num_features, bias=bias),
        )

    # .................................................................................................................

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    # .................................................................................................................


class SwiGLU(nn.Module):
    """
    Simplified implementation of the SwiGLUFFNFused model from Depth-Anythingv2
        @ https://github.com/DepthAnything/Depth-Anything-V2/blob/e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1/depth_anything_v2/dinov2_layers/swiglu_ffn.py#L45

    Most of the flexibility (and training-related code) has been removed
    from the original implementation, for the sake of simplicity.

    This model implements a 'SwiGLU-FFN' activation function.
    This is a variation of a Gated-linear unit (GLU), which has the form:
        GLU(x) = GateFunction(x) * Linear(x)
               = Sigmoid(Linear(x)) * Linear(x)  <- The original GLU implementation

    SwiGLU is a variant on the GLU, which uses a 'swish' function as the gate,
    which is basically just the regular GLU multiplied by the input:
        SwiGLU(x) = Swish(x) * Linear(x)
                  = x * Sigmoid(Linear(x)) * Linear(x)

    Finally, the Depth-Anything-v2 version uses an 'FFN' variant of this, which
    is meant to act like the MLP module that it replaces. In the usual MLP model,
    a linear layer is used to produce a higher number of hidden features, then an
    activation function is applied (e.g. GELU), followed by another linear layer to
    reduce the feature count back to match the input features. The FFN variant of
    SwiGLU treats the SwiGLU as the first linear layer + activation, and then applies
    a linear layer to the result of the SwiGLU:
        SwiGLU_FFN(x) = Linear(SwiGLU(x))

    For more information about GLU & SwiGLU, see: https://arxiv.org/abs/2002.05202
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_feature_ratio: float = 4, bias: bool = True):

        # Inherit from parent
        super().__init__()

        # Calculate number of hidden features following original sizing logic
        mlp_hidden_features = int(hidden_feature_ratio * num_features)
        num_hidden_features = 8 * ((int(mlp_hidden_features * 2 / 3) + 7) // 8)

        self.inner_linear_doubled = nn.Linear(num_features, 2 * num_hidden_features, bias=bias)
        self.outer_linear = nn.Linear(num_hidden_features, num_features, bias=bias)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, tokens_bnc: Tensor) -> Tensor:
        """
        Computes:
            SwiGLU_FFN(x) = Linear(SwiGLU(x))
                          = Linear(x * Sigmoid(Linear(x)) * Linear(x))

        This is computed in a clever way (as per the original implementation),
        where the two inner Linear(x) terms are computed together, using a
        single linear layer that uses double the feature count. The result
        is then split back into separate terms to finish the computation.
        Presumably this is more efficient than computing them separately.
        """
        inner_result = self.inner_linear_doubled(tokens_bnc)
        linear_sigmoid_x, linear_x = inner_result.chunk(2, dim=-1)
        swiglu_x = self.silu(linear_sigmoid_x) * linear_x
        return self.outer_linear(swiglu_x)

    # .................................................................................................................


class LayerNormEPS6(nn.LayerNorm):
    """Simple wrapper around the default layer norm module, with eps set to 1e-6"""

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        epsilon = 1e-6
        try:
            # For torch versions >= 2.1
            super().__init__(normalized_shape, epsilon, bias, elementwise_affine, device, dtype)
        except TypeError:
            # For torch versions < 2.1
            super().__init__(normalized_shape, epsilon, elementwise_affine, device, dtype)

        pass
