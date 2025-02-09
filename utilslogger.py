#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
from datetime import datetime

def setup_logger(log_dir):
    """
    Настройка логгера.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

