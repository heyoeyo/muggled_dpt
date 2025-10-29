#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.misc_helpers import Conv1x1Layer, Conv3x3Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Main model


class ReassembleModel(nn.Module):
    """
    Simplified implementation of the 'reassembly' model/component described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413

    This model includes all 4 (hard-coded) reassembly blocks into a single model

    This implementation has been adapted for the Depth-Anything model. The
    only difference from the original implementation is that the readout token
    is simply ignored, rather than using the 'readout projection' described in
    the original DPT paper.
    """

    # .................................................................................................................

    def __init__(
        self,
        num_vit_features: int,
        hidden_channels_list: tuple[int, int, int, int],
        num_output_channels: int,
    ):

        # Inherit from parent
        super().__init__()

        # Make sure we get exactly 4 hidden channel counts
        ok_hidden_counts = len(hidden_channels_list) == 4
        assert ok_hidden_counts, f"Expecting 4 reassembly channel counts, got: {hidden_channels_list}"
        hidden_1, hidden_2, hidden_3, hidden_4 = hidden_channels_list

        # Build reassembly blocks for each transformer output stage
        self.spatial_upx4 = ReassembleBlock(4, num_vit_features, hidden_1, num_output_channels)
        self.spatial_upx2 = ReassembleBlock(2, num_vit_features, hidden_2, num_output_channels)
        self.spatial_noscale = ReassembleBlock(1, num_vit_features, hidden_3, num_output_channels)
        self.spatial_downx2 = ReassembleBlock(0.5, num_vit_features, hidden_4, num_output_channels)

    # .................................................................................................................

    def forward(
        self,
        stage_1_tokens: Tensor,
        stage_2_tokens: Tensor,
        stage_3_tokens: Tensor,
        stage_4_tokens: Tensor,
        patch_grid_hw: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Takes in tokens output from an image encoder, in 'rows of tokens' format (i.e. BxNxF),
        and produces 4 new image-like outputs of varying spatial size.

        The first-most output is upscaled spatially by a factor of 4 compared to the original
        patch grid sizing. Each stage after is downscaled a factor of 2 compared to the prior
        stage (e.g. upscaled x4, upx2, no scaling, down x2).

        Returns:
            upx4, upx2, noscale, downx2

        For example, assuming the input patch grid HW is 36x36, outputs will have shapes:
               upx4 = B,f,144,144
               upx2 = B,f, 72, 72
            noscale = B,f, 36, 36
             downx2 = B,f, 18, 18
        (Where B is batch size of inputs and f is output channel count, from model config)
        """

        # Perform all 4 reassembly stages
        upx4 = self.spatial_upx4(stage_1_tokens, patch_grid_hw)
        upx2 = self.spatial_upx2(stage_2_tokens, patch_grid_hw)
        noscale = self.spatial_noscale(stage_3_tokens, patch_grid_hw)
        downx2 = self.spatial_downx2(stage_4_tokens, patch_grid_hw)

        return upx4, upx2, noscale, downx2

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Model components


class ReassembleBlock(nn.Module):
    """
    According to paper, reassembly consists of 3 steps + 1 undocumented (?) step:
        1. Read (handle readout token - ignore in this case)
        2. Concatenate (into image-like tensor)
        3. Project + Resample
        4. Project channels to match fusion input sizing (not documented in paper!)

    This block represents a single reassembly stage, which takes tokens output from
    a specific stage of a transformer model and 'reassembles' the token back into
    an image-like format, which is eventually given to a fusion model that
    combines all reassembled stages together into a single output (depth) image.
    """

    # .................................................................................................................

    def __init__(
        self,
        spatial_scale_factor: float,
        num_vit_features: int,
        num_hidden_channels: int,
        num_output_channels: int,
    ):

        # Inherit from parent
        super().__init__()

        # Create 'layer' which maps list of tokens into a 2D image-like output (no learnable parameters!)
        self.token_to_2d = TokensTo2DLayer()
        self.resample = self._make_resample_layer(num_vit_features, num_hidden_channels, spatial_scale_factor)

        # Make final projection layer to match expected channel count for fusion blocks
        self.fuse_proj = Conv3x3Layer(num_hidden_channels, num_output_channels, bias=False)

    # .................................................................................................................

    def forward(self, tokens_bnc: Tensor, patch_grid_hw: tuple[int, int]) -> Tensor:

        # Ignore readout (a.k.a. cls) token
        output = tokens_bnc[:, 1:, :]

        # Convert to image-like tensor, along with projection (change channel count) + re-sampling spatially
        output = self.token_to_2d(output, patch_grid_hw)
        output = self.resample(output)
        output = self.fuse_proj(output)

        return output

    # .................................................................................................................

    @staticmethod
    def _make_resample_layer(
        num_vit_features: int,
        num_hidden_channels: int,
        spatial_scale_factor: float,
    ) -> nn.Sequential:

        # Make sure we interpret the scale as an integer to avoid floating point weirdness
        scale_as_int = int(round(spatial_scale_factor))

        is_upscaling = scale_as_int > 1
        if is_upscaling:
            return ProjUpsampleSeq(num_vit_features, num_hidden_channels, spatial_scale_factor)

        is_downscaling = scale_as_int < 1
        if is_downscaling:
            return ProjDownsampleSeq(num_vit_features, num_hidden_channels, spatial_scale_factor)

        return ProjOnlySeq(num_vit_features, num_hidden_channels)

    # .................................................................................................................


class TokensTo2DLayer(nn.Module):
    """
    The purpose of this layer is to convert transformer tokens, into an
    image-like representation. That is, re-joining the image patch tokens (vectors)
    output from a transformer back into a 2D representation (with many channels)
    More formally, this layer reshapes inputs from: BxNxC -> BxCxHxW
    (This layer does not compute/modify values, it just reshapes the tensors!)
    """

    # .................................................................................................................

    def __init__(self):

        # Inherit from parent
        super().__init__()

    # .................................................................................................................

    def forward(self, tokens_bnc: Tensor, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Assume input tokens have shape:
            B x N x C
        -> Where B is batch size
        -> N is number of tokens
        -> C is the feature size ('channels') of the tokens

        Returns a single output tensor of shape:
            BxCxHxW
        -> Where H & W correspond to the number of image patches vertically (H) and horizontally (W)
        """

        # Transpose to get tokens in last dimension: BxDxN
        output = torch.transpose(tokens_bnc, 1, 2)

        # Expand last (token) dimension into HxW to get image-like (patch-grid) shape: BxDxN -> BxDxHxW
        output = torch.unflatten(output, 2, patch_grid_hw)

        return output

    # .................................................................................................................


class ProjOnlySeq(nn.Sequential):
    """
    Class used to represent the projection + resampling blocks used in reassembly at certain stages
    This version of the block does not perform up-/down-sampling, so the output is the same size
    (spatially) as the input tensor (though the number of channels may change due to projection).

    Technically this is just a 1x1 convolution, but it is represented
    as a sequence in order to mirror the other up-/down-sampling blocks!

    Expects input shape: BxCxHxW
    Returns output shape: Bx(C')xHxW
    -> Where C' is num_output_channels
    """

    # .................................................................................................................

    def __init__(self, num_vit_features: int, num_output_channels: int):

        # Inherit from parent
        super().__init__(
            Conv1x1Layer(num_vit_features, num_output_channels),
        )

    # .................................................................................................................


class ProjUpsampleSeq(nn.Sequential):
    """
    Class used to represent the projection + resampling blocks used in reassembly at certain stages
    This version of the block performs upsampling, so that the output is (spatially) larger
    than the input tensor.
    """

    # .................................................................................................................

    def __init__(self, num_vit_features: int, num_output_channels: int, spatial_scale_factor: float):

        # Make sure scale factor is a factor of 2
        scale_factor = int(torch.log2(torch.tensor(spatial_scale_factor)))
        stride = int(2 ** abs(scale_factor))

        # Inherit from parent
        super().__init__(
            Conv1x1Layer(num_vit_features, num_output_channels),
            nn.ConvTranspose2d(
                num_output_channels,
                num_output_channels,
                stride=stride,
                kernel_size=stride,
                padding=0,
                bias=True,
            ),
        )

    # .................................................................................................................


class ProjDownsampleSeq(nn.Sequential):
    """
    Class used to represent the projection + resampling blocks used in reassembly at certain stages
    This version of the block performs downsampling, so that the output is (spatially) smaller
    than the input tensor.

    Expects input shape: BxCxHxW
    Returns output shape: Bx(C')x(H')x(W')
    -> Where C' is num_output_channels
    -> H' and W' are downsampled height/width (from spatial_scale_factor)
    """

    # .................................................................................................................

    def __init__(self, num_vit_features: int, num_output_channels: int, spatial_scale_factor: float):

        # Make sure scale factor is a factor of 2
        scale_factor = int(torch.log2(torch.tensor(spatial_scale_factor)))
        stride = int(2 ** abs(scale_factor))

        # Make a guess at how kernel size should change with scale factor
        # (this is hard-coded in the original implementation)
        kernel_size = stride + 1

        # Inherit from parent
        super().__init__(
            Conv1x1Layer(num_vit_features, num_output_channels),
            nn.Conv2d(
                num_output_channels,
                num_output_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=1,
                bias=True,
            ),
        )

    # .................................................................................................................
