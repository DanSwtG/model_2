#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from .transformer_model import TimeSeriesTransformer
from .vae_model import VAE

class EnsembleModel(nn.Module):
    
    def __init__(self, transformer, vae):
        super(EnsembleModel, self).__init__()
        self.transformer = transformer
        self.vae = vae
    
    def forward(self, x):
        transformer_out = self.transformer(x)
        vae_out, mu, logvar = self.vae(x)
        return transformer_out, vae_out, mu, logvar

