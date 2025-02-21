# -*- coding: utf-8 -*-
# @Time    : 2020/4/4 8:18
# @Author  : Hui Wang

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import matplotlib.pyplot as plt
import tqdm


def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_flie = 'reviews_' + dataset_name + '.json.gz'
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas



def get_interaction(datas):
    user_seq = defaultdict(list)
    user_max_time1 = defaultdict(lambda: float('-inf'))
    user_min_time1 = defaultdict(lambda: float('inf'))
    user_max_time2 = defaultdict(lambda: float('-inf'))
    user_min_time2 = defaultdict(lambda: float('inf'))
    user_max_time3 = defaultdict(lambda: float('-inf'))
    user_min_time3 = defaultdict(lambda: float('inf'))
    def update_max_times(user, time):
        # 更新前三大值
        if time > user_max_time1[user]:
            user_max_time3[user] = user_max_time2[user]
            user_max_time2[user] = user_max_time1[user]
            user_max_time1[user] = time
        elif time > user_max_time2[user]:
            user_max_time3[user] = user_max_time2[user]
            user_max_time2[user] = time
        elif time > user_max_time3[user]:
            user_max_time3[user] = time

    def update_min_times(user, time):
        # 更新前三小值
        if time < user_min_time1[user]:
            user_min_time3[user] = user_min_time2[user]
            user_min_time2[user] = user_min_time1[user]
            user_min_time1[user] = time
        elif time < user_min_time2[user]:
            user_min_time3[user] = user_min_time2[user]
            user_min_time2[user] = time
        elif time < user_min_time3[user]:
            user_min_time3[user] = time

    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
            update_max_times(user, time)
            update_min_times(user, time)
            
        else:
            user_max_time1[user] = time
            user_min_time1[user] = time
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    sorted_users = sorted(user_max_time3, key=user_max_time3.get, reverse=False)
    
    sorted_user_seq = {user: user_seq[user] for user in sorted_users}


    return sorted_user_seq



# def plot_user_time_lines(user_min_time, user_max_time):
#     users = list(user_min_time.keys())
#     min_times = list(user_min_time.values())
#     max_times = list(user_max_time.values())

#     plt.figure(figsize=(10, 6))
#     t = 0
#     for i, user in enumerate(users):
#         plt.plot([i, i], [min_times[i], max_times[i]], marker='o', label=f'User {user}')
#         print(max_times[i], max_times[i])
#         t += 1
#         if t > 10:
#             break
#     tt = t
#     t = 0
#     for i, user in enumerate(users[-10:]):
#         plt.plot([i + tt, i + tt], [min_times[-10 + t], max_times[-10 + t]], marker='o', label=f'User {user}')
#         t += 1
    


#     plt.xlabel('Users')
#     plt.ylabel('Time')
#     plt.title('User Interaction Time Ranges')
#     plt.xticks(range(t))
#     # plt.legend()
#     plt.savefig("a.png")
#     plt.show()

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def main(data_name, data_type='Amazon'):
    assert data_type in {'Amazon'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id


    # -------------- Save Data ---------------
    data_file = data_name + '.txt'
    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')


# amazon_datas = ['Beauty', 'Video_Games', 'Toys_and_Games', 'Sports_and_Outdoors', 'Health_and_Personal_Care']
amazon_datas = ['Beauty']

for name in amazon_datas:
    main(name, data_type='Amazon')
