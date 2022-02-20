import json
import random
import pickle
import pandas as pd
import numpy as np
from math import exp
import time

from tqdm import tqdm


class LFM(object):
    """
    功能描述： 基于隐语义模型的电影推荐系统
    """

    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lam = 0.01
        self._init_model()

    def _init_model(self):
        """
            初始化参数
                randn: 从标准正态分布中返回n个值
                pd.DataFrame: columns 指定列顺序，index 指定索引
        """
        random.seed(42)

        file_path = './processed_data/ratings.csv'
        pos_neg_path = './processed_data/lfm_items.json'

        self.ui_scores = pd.read_csv(file_path)
        self.user_ids = set(self.ui_scores['UserID'].values)  # 6040
        self.item_ids = set(self.ui_scores['MovieID'].values)  # 3706

        with open(pos_neg_path, mode="r", encoding="utf-8") as f_r:
            self.items_dict = json.load(f_r)

        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def _predict(self, user_id, item_id):
        """
            计算用户 user_id 对 item_id的兴趣度
                p: 用户对每个类别的兴趣度
                q: 物品属于每个类别的概率
        """
        p = np.mat(self.p.loc[int(user_id)].values)
        q = np.mat(self.q.loc[int(item_id)].values).T
        r = (p * q).sum()
        # 借助sigmoid函数，转化为是否感兴趣
        logit = 1.0 / (1 + exp(-r))
        return logit

    # 使用MSE作为损失函数
    def _loss(self, user_id, item_id, y, step):
        e = y - self._predict(user_id, item_id)
        # print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        """
            使用随机梯度下降算法求解参数，同时使用L2正则化防止过拟合
            eg:
                E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
                derivation(E, p) = -matrix_q*(y - predict), derivation(E, q) = -matrix_p*(y - predict),
                derivation（l2_square，p) = lam * p, derivation（l2_square, q) = lam * q
                delta_p = lr * (derivation(E, p) + derivation（l2_square，p))
                delta_q = lr * (derivation(E, q) + derivation（l2_square, q))
        """
        gradient_p = -e * self.q.loc[int(item_id)].values
        l2_p = self.lam * self.p.loc[int(user_id)].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[int(user_id)].values
        l2_q = self.lam * self.q.loc[int(item_id)].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[int(user_id)] -= delta_p
        self.q.loc[int(item_id)] -= delta_q

    # 训练模型，每次迭代都要降低学习率，刚开始由于离最优值相差较远，因此下降较快，当到达一定程度后，就要减小学习率
    def train(self):

        loop = tqdm(range(0, self.iter_count))
        for step in loop:
            for user_id, item_dict in self.items_dict.items():
                e = 0.0
                item_ids = []

                for item_id_str in list(item_dict.values()):
                    item_ids.extend(item_id_str.split(","))

                random.shuffle(item_ids)
                for item_id in item_ids:
                    target = 0 if item_id in item_dict.get("0") else 1
                    e = self._loss(user_id, item_id, target, step)
                    self._optimize(user_id, item_id, e)

                loop.set_description(f'Epoch [{step}/{self.iter_count}]')
                loop.set_postfix(loss=e)
            self.lr *= 0.9

        self.save()

    # 计算用户未评分过的电影，并取top N返回给用户
    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user_id]['MovieID'])
        other_item_ids = self.item_ids ^ user_item_ids  # 交集与并集的差集
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    # 保存模型
    def save(self):
        with open("./processed_data/lfm.model", mode="wb") as f_w:
            pickle.dump((self.p, self.q), f_w)

    # 加载模型
    def load(self):
        with open("./processed_data/lfm.model", mode="rb") as f_r:
            self.p, self.q = pickle.load(f_r)

    # 模型效果评估，从所有user中随机选取10个用户进行评估,评估方法为：MSE
    def evaluate(self):
        self.load()
        users = random.sample(self.user_ids, 10)
        user_dict = {}
        for user in users:
            user_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user]['MovieID'])
            _sum = 0.0
            for item_id in user_item_ids:
                p = np.mat(self.p.loc[int(user)].values)
                q = np.mat(self.q.loc[int(item_id)].values).T
                _r = (p * q).sum()
                r = self.ui_scores[(self.ui_scores['UserID'] == user)
                                   & (self.ui_scores['MovieID'] == item_id)]["Rating"].values[0]
                _sum += abs(r - _r)
            user_dict[user] = _sum / len(user_item_ids)
            print("userID：{},MSE：{}".format(user, user_dict[user]))

        return sum(user_dict.values()) / len(user_dict.keys())


if __name__ == "__main__":
    lfm = LFM()
    # lfm.train()
    # print(lfm.predict(23, 10))
    print(lfm.evaluate())
