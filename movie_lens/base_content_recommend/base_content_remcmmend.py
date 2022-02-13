import json
import math

import numpy as np
import pandas as pd


class BaseContentRecommend(object):
    """
    功能描述： 基于内容的电影推荐系统
            利用 电影类型特征向量  用户类型特征向量 计算 用户对未看过电影的喜爱程度
    """
    def __init__(self, K):
        self.K = K

        self.item_profile = self.read_json_file(f"./processed_dataset/item_profile.json")
        self.user_profile = self.read_json_file(f"./processed_dataset/user_profile.json")

    def read_json_file(self, json_file_path):
        with open(json_file_path, mode="r", encoding="utf-8") as f_r:
            json_data = json.load(f_r)
        return json_data

    # 获取用户未进行评分的item列表
    def get_none_score_item(self, user):
        items = pd.read_csv("./processed_dataset/movies.csv")["MovieID"].values
        data = pd.read_csv("./processed_dataset/ratio.csv")
        have_score_items = data[data["UserID"] == user]["MovieID"].values
        none_score_items = set(items) - set(have_score_items)
        return none_score_items

    # 获取用户对item的喜好程度
    def cosUI(self, user, item):
        Uia = sum(
            np.array(self.user_profile[str(user)])
            *
            np.array(self.item_profile[str(item)])
        )
        Ua = math.sqrt(sum([math.pow(one, 2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one, 2) for one in self.item_profile[str(item)]]))
        return Uia / (Ua * Ia)

    # 为用户进行电影推荐
    def recommend(self, user):
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user, item)
        if self.K is None:
            result = sorted(
                user_result.items(), key=lambda k: k[1], reverse=True
            )
        else:
            result = sorted(
                user_result.items(), key=lambda k: k[1], reverse=True
            )[:self.K]
        print(result)


if __name__ == '__main__':
    base_content_recommend = BaseContentRecommend(K=5)
    base_content_recommend.recommend(12)
