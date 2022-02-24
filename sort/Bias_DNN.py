# 对于原始特征和统计特征, 输入DNN
# 对于时间差特征, 作为bias单独训练

import sys
sys.path.append(".")
sys.path.append("..")
from model_base import *
import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from utils import *

class DNN(BaseModel):
    def __init__(self, Features, hidden_unit, Bias_Features=[], bias_unit=[]):
        super(DNN, self).__init__()
        self.features = Features
        self.bias = Bias_Features
        self.hidden = hidden_unit + [1] # 最后输出为logit
        self.bias_hidden = bias_unit + [1]
        self.embs = Normal_Embedding(Features + Bias_Features, Feature_Embedding_Dim)
        self.fc = nn.ModuleList([])
        self.bias_fc = nn.ModuleList([])
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))
    def get_input(self, df):
        input = self.embs.get_feature_matrix(df, self.features, "concat")
        if len(self.bias)!=0:
            bias_input = self.embs.get_feature_matrix(df, self.bias, "concat")
        else:
            bias_input = torch.Tensor([])
        return (input, bias_input)
    def forward(self, df):
        input, bias_input = self.get_input(df)
        # initialize
        if len(self.fc)==0:
            self.fc = nn.ModuleList([nn.Linear(input.shape[1], self.hidden[0])] + [nn.Linear(self.hidden[i], self.hidden[i+1]) for i in range(len(self.hidden)-1)])
        if len(self.bias)>0 and len(self.bias_fc)==0:
            self.bias_fc = nn.ModuleList([nn.Linear(bias_input.shape[1], self.bias_hidden[0])] + [nn.Linear(self.bias_hidden[i], self.bias_hidden[i+1]) for i in range(len(self.bias_hidden)-1)])
        # dnn output
        for i in range(len(self.fc)-1):
            input = F.relu(self.fc[i](input))
        dnn_output = self.fc[-1](input)
        # bias output 
        if len(self.bias)>0:
            for i in range(len(self.bias_fc)-1):
                bias_input = F.relu(self.bias_fc[i](bias_input))
            bias_output = self.bias_fc[-1](bias_input)
            dnn_output + bias_output
        return dnn_output.reshape(-1)
        

if __name__=="__main__":
    model = DNN(
        Features = Config.user_feature + Config.item_feature + Config.count_feature, 
        hidden_unit = [32, 32], 
        Bias_Features = ['time_diff'],
        bias_unit = []
    )

    test = pd.read_csv('./data/test_data.csv')
    test_y = torch.Tensor(list(test['label']))

    for fea in Config.count_feature:
        test[fea] = list(map(eval, list(test[fea])))

    print(model.get_input(test))

    #print(model.fc)


    model.fit(test, test_y, batch_size = 512, lr=1e-1)