import json
import os

import pandas as pd


class DataProcessor(object):
    """
    功能描述 分别获取 电影类型特征向量  用户类型特征向量
    """
    def __init__(self):
        self.user_file_path = f"../dataset/ml-1m/users.dat"
        self.movie_file_path = f"../dataset/ml-1m/movies.dat"
        self.ratio_file_path = f"../dataset/ml-1m/ratings.dat"

    def process(self):
        print(f"开始转换用户数据(users.dat) ...")
        self.process_user_data()
        print('开始转化电影数据（movies.dat）...')
        self.process_movie_data()
        print('开始转化用户对电影评分数据（ratings.dat）...')
        self.process_ratio_data()
        print('Over!')

    def process_user_data(self):
        user_save_file_path = f"./processed_dataset/users.csv"
        if os.path.exists(user_save_file_path):
            print("user.csv已经存在, return")
            return
        df = pd.read_table(self.user_file_path, sep="::", encoding="unicode_escape",
                           names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        df.to_csv(user_save_file_path, index=False)

    def process_movie_data(self):
        movie_save_file_path = f"./processed_dataset/movies.csv"
        if os.path.exists(movie_save_file_path):
            print("movies.csv已经存在, return")
            return
        df = pd.read_table(self.movie_file_path, sep="::", encoding="unicode_escape",
                           names=["MovieID", "Title", "Genres"])
        df.to_csv(movie_save_file_path, index=False)

    def process_ratio_data(self):
        ratio_save_file_path = f"./processed_dataset/ratio.csv"
        if os.path.exists(ratio_save_file_path):
            print("ratio.csv已经存在, return")
            return
        df = pd.read_table(self.ratio_file_path, sep="::", encoding="unicode_escape",
                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        df.to_csv(ratio_save_file_path, index=False)

    # 获取item的特征信息矩阵
    def prepare_item_profile(self, file='./processed_dataset/movies.csv'):

        items = pd.read_csv(file)
        item_ids = set(items["MovieID"].values)
        self.item_dict = {}
        genres_all = list()
        # 将每个电影的类型放在item_dict中
        for item in item_ids:
            genres = items[items["MovieID"] == item]["Genres"].values[0].split("|")
            self.item_dict.setdefault(item, []).extend(genres)
            genres_all.extend(genres)
        self.genres_all = set(genres_all)
        # 将每个电影的特征信息矩阵存放在 self.item_matrix中
        # 保存dict时，key只能为str，所以这里对item id做str()转换
        self.item_matrix = {}
        for item in self.item_dict.keys():
            self.item_matrix[str(item)] = [0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index = list(set(genres_all)).index(genre)
                self.item_matrix[str(item)][index] = 1
        json.dump(self.item_matrix,
                  open('./processed_dataset/item_profile.json', 'w'), indent=4)
        print("item 信息计算完成，保存路径为：{}"
              .format('./processed_dataset/item_profile.json'))

    # 计算用户的偏好矩阵
    def prepare_user_profile(self, file='./processed_dataset/ratio.csv'):
        users = pd.read_csv(file)
        user_ids = set(users["UserID"].values)
        # 将users信息转化成dict
        users_rating_dict = {}
        for user in user_ids:
            users_rating_dict.setdefault(str(user), {})
        with open(file, "r") as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user, item, rate) = line.split(",")[:3]
                    users_rating_dict[user][item] = int(rate)

        # 获取用户对每个类型下都有哪些电影进行了评分
        self.user_matrix = {}
        # 遍历每个用户
        for user in users_rating_dict.keys():

            score_list = users_rating_dict[user].values()
            # 用户的平均打分
            avg = sum(score_list) / len(score_list)
            self.user_matrix[user] = []
            # 遍历每个类型（保证item_profile和user_profile信息矩阵中每列表示的类型一致）
            for genre in self.genres_all:
                score_all = 0.0
                score_len = 0
                # 遍历每个item
                for item in users_rating_dict[user].keys():
                    # 判断类型是否在用户评分过的电影里
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item] - avg)
                        score_len += 1
                if score_len == 0:
                    self.user_matrix[user].append(0.0)
                else:
                    self.user_matrix[user].append(score_all / score_len)
        json.dump(self.user_matrix,
                  open('./processed_dataset/user_profile.json', 'w'), indent=4)
        print("user 信息计算完成，保存路径为：{}"
              .format('./processed_dataset/user_profile.json'))


if __name__ == '__main__':
    process_data = DataProcessor()
    process_data.process()
    process_data.prepare_item_profile()
    process_data.prepare_user_profile()
