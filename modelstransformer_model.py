#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from .base_model import BaseModel

class TimeSeriesTransformer(BaseModel):
    """
    Трансформер для временных рядов.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.transformer(x)

