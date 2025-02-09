#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config:
    # Гиперпараметры модели
    NUM_SKUS = 1000
    NUM_REGIONS = 50
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    NUM_HEADS = 8
    NUM_LAYERS = 6
    SEASONALITY_PERIOD = 12

    # Параметры обучения
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 500
    DROPOUT = 0.2
    WEIGHT_DECAY = 0.01

    # Логирование
    LOG_DIR = "logs/"

