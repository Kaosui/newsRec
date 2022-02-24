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

# Features = user_feature + item_feature + match_feature + count_feature

class FM(BaseModel):
    '''
    params:
        Features -> List: name of features
        default_vector_dim -> int||dict[str:int]: embedding dim of features
    '''
    def __init__(self, Features, default_vector_dim):
        super(FM, self).__init__()
        self.features = Features
        self.embs = Normal_Embedding(Features, default_vector_dim)
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))
    def get_input(self, df):
        res = []
        for feature in self.features:
            res.append(self.embs.get_slot_vector(df, feature).unsqueeze(1))  #[batch, 1, emb_dim]
        res = torch.cat(res, dim=1) #[batch, feature_num, emb_dim]
        return res
    def forward(self, df):
        X = self.get_input(df)
        # FM sum
        FM_sum = torch.sum(torch.square(torch.sum(X, dim=1)), dim=1) - torch.sum(torch.sum(torch.square(X), dim=2), dim=1)
        return FM_sum  #[batch, ]


        



if __name__=="__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(114514)

    Features = RecallConfig.user_feature + RecallConfig.item_feature + RecallConfig.match_feature + RecallConfig.count_feature
    print(Features)

    # 读取数据
    train = pd.read_csv('./data/train_data.csv')
    test = pd.read_csv('./data/test_data.csv')

    for fea in count_feature:
        train[fea] = list(map(eval, list(train[fea])))
        test[fea] = list(map(eval, list(test[fea])))

    train_y, test_y = torch.Tensor(list(train['label'])), torch.Tensor(list(test['label']))

    fm = FM(Features, 8)

    tmp = fm.get_FM_input(test)
    print(len(tmp))
    print(tmp[0].shape)
    # 训练
    #fm.fit(train, train_y, test, test_y, epoch=5, batch_size=1024, lr=1e-2)
