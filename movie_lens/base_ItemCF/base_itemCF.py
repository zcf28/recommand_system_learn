import json
import math
import os
import random


class BaseItemCF(object):
    """
    功能描述： 基于 物品的协同过滤算法 电影推荐系统
    """

    def __init__(self):
        self.ratio_file_path = f"../dataset/ml-1m/ratings.dat"

        self.data = self.read_data()

        self.train_data, self.test_data = self.split_data(k=1, seed=42)

        self.item_sim = self.item_similarity()

        # 加载评分数据到data

    def read_data(self):
        print(f"加载数据...")
        data_list = []

        with open(self.ratio_file_path, mode="r", encoding="utf-8") as f_r:
            data_info_list = f_r.readlines()
            for data_info in data_info_list:
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

    # 计算物品之间的相似度
    def item_similarity(self):
        print("开始计算物品之间的相似度")
        item_sim_file_path = f"./precessed_data/item_sim.json"
        if os.path.exists(item_sim_file_path):
            print("物品相似度从文件加载 ...")
            with open(item_sim_file_path, mode="r", encoding="utf-8") as f_r:
                item_sim = json.load(f_r)
            return item_sim

        else:
            item_sim = dict()
            item_user_count = dict()  # 得到每个物品有多少用户产生过行为
            count = dict()  # 共现矩阵
            for user, item in self.train_data.items():
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.train_data[str(user)][i] > 0.0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.train_data[str(user)][i] > 0.0 and \
                                self.train_data[str(user)][j] > 0.0 and i != j:
                            count[i][j] += 1

            # 共现矩阵 -> 相似度矩阵
            for i, related_items in count.items():
                item_sim.setdefault(i, dict())
                for j, cuv in related_items.items():
                    item_sim[i].setdefault(j, 0)
                    item_sim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])

        with open(item_sim_file_path, mode="w", encoding="utf-8") as f_w:
            json.dump(item_sim, f_w, indent=4)

        return item_sim

    # 为用户进行推荐
    def recommend(self, item, k=4, n_items=4):
        result = dict()
        u_items = self.train_data.get(item, {})
        for i, pi in u_items.items():
            for j, wj in sorted(self.train_data[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                result[j] += pi * wj

        result_sorted = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:n_items])

        print(f"user: {item}  --> recommend result : {result_sorted}")

        return result_sorted

    #  计算准确率
    def precision(self, k=8, n_items=4):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        for user in self.test_data.keys():
            u_items = self.test_data.get(user, {})
            result = self.recommend(user, k=k, n_items=n_items)
            for item, rate in result.items():
                if item in u_items:
                    hit += 1
            precision += n_items

        pre = hit / (precision * 1.0)

        print(f"pre : {pre}")
        return pre


if __name__ == '__main__':
    base_itemCF = BaseItemCF()
    base_itemCF.recommend("2")
    base_itemCF.precision()

