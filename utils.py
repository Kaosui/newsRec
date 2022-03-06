import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./')
from Config import *

# 在这里完成所有特征的Embedding
class Normal_Embedding(nn.Module):
    def __init__(self, Features, default_vector_dim):
        super(Normal_Embedding, self).__init__()
        self.features = Features
        if type(default_vector_dim)==int:
            default_vector_dim = {feature_name:default_vector_dim for feature_name in Features}
        self.embs = nn.ModuleDict({})
        for feature_name in Features:
            if feature_name in count_feature:
                if "count_feature" not in self.embs:
                    self.embs["count_feature"] = nn.Embedding(Feature_Slot['click_article_id']+1, default_vector_dim['click_article_id'], padding_idx=Feature_Slot['click_article_id'])
            else:
                self.embs[feature_name] = nn.Embedding(Feature_Slot[feature_name], default_vector_dim[feature_name])
    def get_slot_vector(self, df, feature_name):
        assert(feature_name in self.features)
        if feature_name in count_feature:
            res = self.embs["count_feature"](torch.LongTensor(list(df[feature_name])))
            res = torch.sum(res, dim=1)
            return F.normalize(res, p=2, dim=-1)
        elif feature_name == 'click_article_id':
            return F.normalize(self.embs[feature_name](torch.LongTensor(list(df[feature_name]))), p=2, dim=-1)
        else:
            return self.embs[feature_name](torch.LongTensor(list(df[feature_name])))
    def get_feature_matrix(self, df, features, type="concat"):
        ''' type in [concat, sum, mean]'''
        if type not in ['concat', 'sum', 'mean']:
            raise ValueError("type must be one of: [\'concat\', \'sum\', \'mean\']")
        res = []
        for feature in features:
            res.append(self.get_slot_vector(df ,feature))
        if type == 'concat':
            return torch.cat(res, dim=-1)
        elif type == 'sum':
            return sum(res)
        elif type == 'mean':
            return sum(res)/len(res)


# 打印状态
def timelogger(message=None):
    print(f"[{datetime.now()}]") if message==None else print(f"[{datetime.now()}]" + str(message))


# 构造点击时差特征
def get_time_diff(last_clk_time, created_ts):
    time_diff = last_clk_time-created_ts
    if time_diff < 3*60*60*1000: return 0
    elif time_diff < 12*60*60*1000: return 1
    elif time_diff < 24*60*60*1000 : return 2
    else: return 3


# 特征拼接
def merge_feature(user_id, user_df, item_df, use_match_feature=False):
    #item_df['user_id'] = [user_id for _ in range(len(item_df))]
    item_df['user_id'] = user_id
    res = user_df[user_df['user_id'].isin([user_id])]
    res = res.merge(item_df, how='left', on='user_id')
    if use_match_feature==True:
        time_diff = []
        for i,row in res.iterrows():
            time_diff.append(get_time_diff(row['last_click_time'], row['created_at_ts']))
        res['time_diff'] = time_diff
    return res


if __name__ == '__main__':
    # 测试代码
    Features = user_feature + item_feature + match_feature + count_feature

    data = pd.read_csv("./data/test_data.csv").drop(['label'],axis=1)
    for fea in count_feature:
        data[fea] = list(map(eval, list(data[fea])))

    emb1 = Normal_Embedding(Features = Features,default_vector_dim = Feature_Embedding_Dim)
    emb2 = Normal_Embedding(Features = Features ,default_vector_dim = 8)

    for feature in Features:
        print(f'feature: {feature}')
        print(emb1.get_slot_vector(data, feature))

    X = emb1.get_feature_matrix(data, Features, 'concat')
    print(X)
    print(X.shape)

    print("===============================================================================")

    X = emb2.get_feature_matrix(data, Features, 'sum')
    print(X)
    print(X.shape)
    # output = emb(data)
    # print(output)
    # print(output.shape)