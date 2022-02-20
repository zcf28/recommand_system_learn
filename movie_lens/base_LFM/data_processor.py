import json
import os
import pickle

import pandas as pd


class DataProcessor(object):
    """
    功能描述 数据处理
    """

    def __init__(self):
        self.origin_ratio_path = f'../dataset/ml-1m/ratings.dat'
        self.ratio_path = f"./processed_data/ratings.csv"
        self.lfm_items_path = f"./processed_data/lfm_items.json"

        self.processor()

        self.ui_scores = pd.read_csv(self.ratio_path)
        self.user_ids = set(self.ui_scores["UserID"].values)
        self.item_ids = set(self.ui_scores["MovieID"].values)

        self.get_pos_neg_item()

    def processor(self):
        print('开始转化用户对电影评分数据（ratings.dat）...')
        self.process_rating_data()
        print('Over!')

    def process_rating_data(self):
        if not os.path.exists(self.ratio_path):
            fp = pd.read_table(self.origin_ratio_path, sep='::', engine='python',
                               names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            fp.to_csv(self.ratio_path, index=False)

    # 对用户进行有行为电影和无行为电影数据标记
    def get_pos_neg_item(self):

        if not os.path.exists(self.lfm_items_path):
            ui_scores = pd.read_csv(self.ratio_path)
            user_ids = set(ui_scores["UserID"].values)

            items_dict = {str(user_id): self.get_one(user_id) for user_id in list(user_ids)}
            with open(self.lfm_items_path, mode="w", encoding="utf-8") as f_w:
                json.dump(items_dict, f_w, indent=4)

    # 定义单个用户的正向和负向数据
    # 正向：用户有过评分的电影；负向：用户无评分的电影
    def get_one(self, user_id):
        print('为用户%s准备正向和负向数据...' % user_id)

        pos_item_ids = set(self.ui_scores[self.ui_scores['UserID'] == user_id]['MovieID'])
        # 对称差：x和y的并集减去交集
        neg_item_ids = self.item_ids ^ pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        item_dict.update({
            1: ",".join([str(i) for i in pos_item_ids])
        })
        item_dict.update({
            0: ",".join([str(i) for i in neg_item_ids])
        })

        return item_dict


if __name__ == '__main__':
    data_processor = DataProcessor()

    data_processor.get_pos_neg_item()
