#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Main module

class ReadoutProjectLayer(nn.Sequential):

    '''
    The purpose of this layer is to re-distribute the information contained in the
    extra 'readout' (also called 'class' or 'cls') token that comes from the output of
    a transformer model. This extra token does not correspond to an image patch, and
    is instead included as a kind of 'global representation' token. From testing in the
    paper, the authors conclude that merging the information from this token into
    the image patch tokens provided some benefit to the model performance
    (compared to ignoring it or simply adding it, for example).
    
    Takes input of shape: BxNxF
    Returns output of shape: Bx(N-1)xF
    -> B is batch dimension
    -> N-1 is number of tokens after removing the readout token
    -> F is the number of features per token
    '''    
    
    # .................................................................................................................
    
    def __init__(self, num_vit_features):
        
        # Create 'mlp' component used to project concatenated tokens back down to original token size
        size_of_concatenated_token = (num_vit_features + num_vit_features)
        super().__init__(
            ReadoutConcatLayer(),
            nn.Linear(size_of_concatenated_token, num_vit_features),
            nn.GELU(),
        )
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Components

class ReadoutConcatLayer(nn.Module):
    
    '''
    The purpose of this layer is to extract the readout (or 'cls') token from the
    output of a transformer, and concatenate the token onto all remaining tokens.
    
    Note that this is not a learnable module! It exists as a module only so that it can
    be used within sequential module.
    
    Takes input with shape: BxNxF
    Outputs shape: Bx(N-1)x(2F)
    -> N-1 is the number of tokens left after removing the readout token
    -> 2F is the number of features of each token after concatenating the readout token
    '''
    
    # .................................................................................................................
    
    def forward(self, transformer_tokens):
        
        # Extract readout token from transformer output, assuming readout token is at index 0
        readout_token = transformer_tokens[:, 0]
        image_tokens = transformer_tokens[:, 1:]
        
        # Concatenate readout token to all other tokens (note: readout is repeated for every image token)
        readout_token = readout_token.unsqueeze(1).expand_as(image_tokens)
        output = torch.cat((image_tokens, readout_token), -1)
        
        return output
    
    # .................................................................................................................


