import json
import math
import os
import random


class BaseUserCF(object):
    """
    功能描述： 基于 用户的协调过滤 电影推荐系统
    """
    def __init__(self):
        self.ratio_file_path = f"../dataset/ml-1m/ratings.dat"

        self.data = self.read_data()

        self.train_data, self.test_data = self.split_data(k=1, seed=42)

        self.user_sim = self.user_similarity()

    # 加载评分数据到data
    def read_data(self):
        print(f"加载数据...")
        data_list = []

        with open(self.ratio_file_path, mode="r", encoding="utf-8") as f_r:
            data_info_list = f_r.readlines()
            for data_info in data_info_list :
                user_id, movie_id, ratio, _ = data_info.strip().split("::")
                data_list.append((user_id, movie_id, int(ratio)))

        return data_list

    # 拆分数据集为训练集和测试集
    def split_data(self, k, seed, M=8):
        print("训练数据集与测试数据集切分...")
        train_data, test_data = {}, {}
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                test_data.setdefault(user, {})
                test_data[user][item] = record
            else:
                train_data.setdefault(user, {})
                train_data[user][item] = record

        return train_data, test_data

    def user_similarity(self):
        print("开始计算用户之间的相似度 ...")
        user_sim_file_path = f"./processed_data/user_sim_json"
        if os.path.exists(user_sim_file_path):
            print("用户相似度从文件加载 ...")
            with open(user_sim_file_path, mode="r", encoding="utf-8") as f_r:
                user_sim = json.load(f_r)
            return user_sim

        else:
            # 得到每个item被哪些user评价过
            item_users = dict()
            for u, items in self.train_data.items():
                for i in items.keys():
                    item_users.setdefault(i, set())
                    if self.train_data[u][i] > 0:
                        item_users[i].add(u)
            # 构建倒排表
            count = dict()
            user_item_count = dict()
            for i, users in item_users.items():
                for u in users:
                    user_item_count.setdefault(u, 0)
                    user_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / math.log(1 + len(users))
            # 构建相似度矩阵
            user_sim = dict()
            for u, related_users in count.items():
                user_sim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    user_sim[u].setdefault(v, 0.0)
                    user_sim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])

            with open(user_sim_file_path, mode="w", encoding="utf-8") as f_w:
                json.dump(user_sim, f_w, indent=4)

            return user_sim

    # 为用户user进行物品推荐
    def recommend(self, user, k=4, n_items=10):
        result = dict()
        have_score_items = self.train_data.get(user, {})
        for v, wuv in sorted(self.user_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.train_data[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        result_sorted = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:n_items])
        print(f"user: {user}  --> recommend result : {result_sorted}")
        return result_sorted

    # 计算算法准确性
    def precision(self, k=4, n_items=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        for user in self.train_data.keys():
            tu = self.test_data.get(user, {})
            rank = self.recommend(user, k=k, n_items=n_items)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += n_items
        pre = hit / (precision * 1.0)

        print(f"pre : {pre}")
        return pre


if __name__ == '__main__':
    base_userCF = BaseUserCF()
    base_userCF.recommend("3")
    base_userCF.precision()
