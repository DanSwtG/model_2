#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from config import Config
from models.transformer_model import TimeSeriesTransformer
from models.vae_model import VAE
from models.ensemble_model import EnsembleModel
from utils.logger import setup_logger

logger = setup_logger(Config.LOG_DIR)

# Создание моделей
transformer = TimeSeriesTransformer(
    input_dim=Config.EMBEDDING_DIM * 2 + 1,
    hidden_dim=Config.HIDDEN_DIM,
    num_heads=Config.NUM_HEADS,
    num_layers=Config.NUM_LAYERS
)
vae = VAE(input_dim=Config.EMBEDDING_DIM * 2 + 1, hidden_dim=Config.HIDDEN_DIM, latent_dim=64)
model = EnsembleModel(transformer, vae)

# Обучение
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
criterion = torch.nn.MSELoss()

for epoch in range(Config.EPOCHS):
    for batch in DataLoader(dataset, batch_size=Config.BATCH_SIZE):
        optimizer.zero_grad()
        output, vae_out, mu, logvar = model(batch)
        loss = criterion(output, batch.target) + 0.1 * criterion(vae_out, batch.target)
        loss.backward()
        optimizer.step()
    
    logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

