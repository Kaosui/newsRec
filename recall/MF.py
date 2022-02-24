from logging.config import stopListening
import sys
sys.path.append(".")
sys.path.append("..")
import RecallConfig
from  model_base import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from utils import *

class MF(BaseModel):
    '''
    params:
        Features -> List: name of features
        default_vector_dim -> int||dict[str:int]: embedding dim of features
    '''
    def __init__(self, Features, default_vector_dim):
        super(MF, self).__init__()
        if len(Features)>2:
            Features = ["click_article_id", 'user_last_click_7t']
        self.features = sorted(Features)
        self.embs = Normal_Embedding(self.features, default_vector_dim)
        self.ce = nn.MSELoss()
    def get_input(self, df):
        res = []
        for feature in self.features:
            res.append(self.embs.get_slot_vector(df, feature))  #[batch, emb_dim]
        return res
    def forward(self, df):
        In = self.get_input(df)
        # 计算余弦相似度
        res = torch.cosine_similarity(In[0], In[1], dim=-1)
        return res  #[batch, ]

