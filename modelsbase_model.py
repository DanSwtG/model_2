#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn

class BaseModel(nn.Module):
    """
    Базовый класс моделей.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *inputs):
        raise NotImplementedError("Метод forward для реализации в дочернем классе.")

    def save(self, path):
        """Сохраняет модель."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Загружает модель."""
        self.load_state_dict(torch.load(path))

