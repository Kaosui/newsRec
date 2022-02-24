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

class DSSM(BaseModel):
    '''
        params:
            uFeatures -> List: features of user
            iFeatures -> List: features of item
            unit_hidden -> List: dim of hidden states
        return:
            (user_embedding, item_embedding), sim_scores
    '''
    def __init__(self, uFeatures, iFeatures, default_vector_dim, unit_hidden):
        super(DSSM, self).__init__()
        self.ufeatures = sorted(uFeatures)
        self.ifeatures = sorted(iFeatures)
        self.hidden = unit_hidden #+ [2] if unit_hidden[-1]!=2 else unit_hidden
        self.embs = Normal_Embedding(uFeatures + iFeatures, default_vector_dim)
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))
        self.ufc = nn.ModuleList([])  # 全连接层
        self.ifc = nn.ModuleList([])
    def get_input(self, df):
        user_emb = self.embs.get_feature_matrix(df, self.ufeatures, type='concat')
        item_emb = self.embs.get_feature_matrix(df, self.ifeatures, type='concat')
        # initialize fc layer
        if len(self.ufc)==0:
            self.ufc = nn.ModuleList([nn.Linear(user_emb.shape[1], self.hidden[0])] + [nn.Linear(self.hidden[i], self.hidden[i+1]) for i in range(len(self.hidden)-1)])
        if len(self.ifc)==0:
            self.ifc = nn.ModuleList([nn.Linear(item_emb.shape[1], self.hidden[0])] + [nn.Linear(self.hidden[i], self.hidden[i+1]) for i in range(len(self.hidden)-1)])
        return (user_emb, item_emb)        
    def forward(self, df):
        user_emb, item_emb = self.get_input(df)
        # 计算tower输出
        for i in range(len(self.ufc)):
            user_emb = F.relu(self.ufc[i](user_emb))
            item_emb = F.relu(self.ifc[i](item_emb))
        scores = torch.cosine_similarity(user_emb, item_emb, dim=-1)
        return (user_emb, item_emb), scores
    def get_y_pre(self, df):
        self.eval()
        return self.forward(df)[1]
    def get_embedding(self, df):
        embs, _ = self.forward(df)
        return embs
