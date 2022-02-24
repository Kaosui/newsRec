'''
所有模型的模板, 包含的模块:
训练、打印状态、获得预测值、计算loss、计算auc
子类继承时只需要复写: 计算输入模块get_input(); 前向传播forward()

df: pd.DataFrame, 特征集必须是df列的子集
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import sys
sys.path.append(".")
sys.path.append("..")
from utils import *

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.ce = "use loss function which you need in torch.nn"

    def get_input(self, df):
        pass

    def forward(self, df):
        pass

    def get_y_pre(self, df):
        self.eval()
        return self.forward(df)

    def get_loss(self, df, y):
        y_pre = self.get_y_pre(df)
        return self.ce(y_pre, y)

    def fit(self, train_df, train_y, test_df=None, test_y=None, epoch = 3, batch_size = 128, lr = 1e-3):
        self.train()
        optimizer = optim.Adagrad(self.parameters(), lr=lr)
        timelogger("start fit model")
        for i in range(epoch):
            for j in range(0, len(train_df), batch_size):
                optimizer.zero_grad()
                x,y = train_df[j:j+batch_size], train_y[j:j+batch_size]
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
            #timelogger(f"[epoch:{i+1}||loss:{self.get_loss(df, label)}||auc:{self.get_auc(df, label)}]]")
            self.training_log(i, train_df, train_y, test_df, test_y)

    def get_auc(self, df, label):
        from sklearn.metrics import roc_auc_score
        try:
            self.eval()
            with torch.no_grad():
                y_pre = self.get_y_pre(df).data
                return roc_auc_score(label.numpy() ,y_pre.numpy())
        except:
            return -1

    def training_log(self, i, train_df, train_y, test_df=None, test_y=None):
        if test_df is not  None and test_y is not  None:
            timelogger("[epoch:{}||train_loss/test_loss:{:.4f}/{:.4f}||train_auc/test_auc:{:.3f}/{:.3f}]".format(
                i+1, self.get_loss(train_df, train_y), self.get_loss(test_df, test_y), self.get_auc(train_df, train_y), self.get_auc(test_df, test_y)
                ))
        else:
            timelogger(f"[epoch:{i+1}||loss:{self.get_loss(train_df, train_y)}||auc:{self.get_auc(train_df, train_y)}]]")