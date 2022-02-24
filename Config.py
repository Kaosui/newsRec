# created_at_ts 和 last_click_time 用来生成 time_diff

user_feature = [
    # "user_id",
    "click_environment",
    "click_deviceGroup",
    "click_os",
    "click_country",
    "click_region",
    "click_referrer_type",
]

item_feature = [
    "click_article_id",
    "category_id",
]

match_feature = [
    "time_diff",
]

count_feature = [
    'user_last_click_1t',
    'user_last_click_3t',
    'user_last_click_5t',
    'user_last_click_7t',
]

# 特征的个数
Feature_Slot = {
    # user
    "user_id": 200000,
    "click_environment": 3,
    "click_deviceGroup": 5,
    "click_os": 8,
    "click_country": 11,
    "click_region": 28,
    "click_referrer_type": 7,
    # item
    "click_article_id": 31116,
    "category_id": 290,
    # match
    "time_diff": 4,
}

# 设置embedding的维度
Feature_Embedding_Dim = {
    "user_id": 32,
    "click_environment": 1,
    "click_deviceGroup": 1,
    "click_os": 1,
    "click_country": 1,
    "click_region": 2,
    "click_referrer_type": 1,

    "time_diff": 1,

    "click_article_id": 32,
    "category_id": 4,
}

