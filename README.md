# newsRec

新闻推荐的小项目, 原始数据取自天池大数据竞赛"零基础入门--新闻推荐", 忽略冷启动, 这里仅考虑有行为的用户和文章, 原始数据存在一定的**规律**, 即存在大量点击差为固定值30s的行为, 推测该比赛是一个**以模型逼近模型**的比赛, 为了数据集分布的合理性, 这里仅采用**最后一次点击和倒数第二次点击差为30s**的数据进行训练,并作了少许加工：

1. 筛选符合条件的数据
2. 取所有用户的最后一次点击作为正样本, 剔除点击次数≤1的用户(筛选发现所有满足1.的数据都至少有2次点击)
3. 按照规则负采样生成负样本(正/负 = 1/4)
4. 重新随机划分训练集&验证集(150000/50000)

文件构成：
1. recall  
    包含采用的召回模型, 召回的数据, 以及模型评估;  
    评估规则：topk命中率

2. sort  
    包含采用的精排模型, 以及模型评估;  
    评估规则：auc

3. data  
    原始数据预处理后的数据

4. feature  
   生成的user&item特征

5. data_preprocess  
    原始数据预处理
 
流程: 预处理 -> 特征工程 -> 负采样生成训练集 -> 召回 -> 精排  
运行方式(每一步请进入文件所在的文件夹下执行, 否则会报路径错误), 下载原始数据到 "./data_preprocess/data/"
1. 运行data_preprocess文件夹下的run.ipynb, 完成数据的预处理, 产出在当前目录的data文件夹下
2. 依次运行 feature_gen.ipynb, feature_joiner.ipynb 完成特征生成和拼接的任务, 保存在feature文件夹下
3. 运行recall文件夹下的run.ipynb完成召回, 结果保存在 recall/recall_data/下
4. 运行feature_gen_sort.ipynb完成精排阶段的负采样&特征拼接
5. 运行sort文件夹下的run.ipynb完成精排.  
[原始数据下载链接](https://tianchi.aliyun.com/competition/entrance/531842/information)

#
# 特征工程
## 统一保存为csv格式
0. 预处理  
    离散特征重新label encoder;  
    数据集是基于用户的推荐, 在时间维度上不统一, 因此这里的时间特征以每一个用户最后一次点击时间为基准T, 以T-3h, T-12h, T-24h分桶.

1. 统计特征为用户的最近1,3,5,7次点击

#
# 召回结果展示, 召回样本300所达到的点击覆盖率
## MF
[2022-02-19 15:57:08.301364]k = 1, hit rate: 3602/50000 = 7.20%  
[2022-02-19 15:57:16.073106]k = 5, hit rate: 10322/50000 = 20.64%  
[2022-02-19 15:57:24.120386]k = 10, hit rate: 13704/50000 = 27.41%  
[2022-02-19 15:57:32.206683]k = 20, hit rate: 17326/50000 = 34.65%  
[2022-02-19 15:57:40.669179]k = 30, hit rate: 19502/50000 = 39.00%  
[2022-02-19 15:57:49.614035]k = 50, hit rate: 22123/50000 = 44.25%  
[2022-02-19 15:57:59.503784]k = 100, hit rate: 25765/50000 = 51.53%  
[2022-02-19 15:58:10.892880]k = 200, hit rate: 29120/50000 = 58.24%  
[2022-02-19 15:58:23.919842]k = 300, hit rate: 31040/50000 = 62.08% 

## FM
[2022-02-19 23:10:41.408552]k = 1, hit rate: 186/50000 = 0.37%  
[2022-02-19 23:10:49.377339]k = 5, hit rate: 809/50000 = 1.62%  
[2022-02-19 23:10:57.212678]k = 10, hit rate: 1525/50000 = 3.05%  
[2022-02-19 23:11:04.273601]k = 20, hit rate: 2952/50000 = 5.90%  
[2022-02-19 23:11:12.604879]k = 30, hit rate: 4363/50000 = 8.73%  
[2022-02-19 23:11:21.777417]k = 50, hit rate: 6873/50000 = 13.75%  
[2022-02-19 23:11:31.667860]k = 100, hit rate: 11656/50000 = 23.31%  
[2022-02-19 23:11:44.759606]k = 200, hit rate: 18124/50000 = 36.25%  
[2022-02-19 23:11:58.175781]k = 300, hit rate: 22469/50000 = 44.94%  

## DSSM
[2022-02-20 12:58:41.057589]k = 1, hit rate: 486/50000 = 0.97%  
[2022-02-20 12:58:48.857425]k = 5, hit rate: 2071/50000 = 4.14%  
[2022-02-20 12:58:56.647488]k = 10, hit rate: 3878/50000 = 7.76%  
[2022-02-20 12:59:04.263203]k = 20, hit rate: 6904/50000 = 13.81%  
[2022-02-20 12:59:10.377717]k = 30, hit rate: 8960/50000 = 17.92%  
[2022-02-20 12:59:19.155544]k = 50, hit rate: 12069/50000 = 24.14%  
[2022-02-20 12:59:32.554252]k = 100, hit rate: 18070/50000 = 36.14%  
[2022-02-20 12:59:46.257330]k = 200, hit rate: 25285/50000 = 50.57%  
[2022-02-20 13:00:00.220067]k = 300, hit rate: 28840/50000 = 57.68%    
    
#
# 精排结果展示，采用MF单路召回，候选数量300，top10的命中率
## DNN  
