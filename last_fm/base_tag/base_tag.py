import pandas as pd
import math


class RecBasedTag:
    """
    功能描述：  利用标签推荐算法实现艺术家的推荐
    """

    # 由于从文件读取为字符串，统一格式为整数，方便后续计算
    def __init__(self):
        # 用户听过艺术次数文件
        self.user_rate_file = "../dataset/lastfm-2k/user_artists.dat"
        # 用户打标签信息
        self.user_tag_file = "../dataset/lastfm-2k/user_taggedartists.dat"

        # 获取所有的艺术家ID
        self.artists_all = list(
            pd.read_table("../dataset/lastfm-2k/artists.dat", delimiter="\t")["id"].values
        )
        # 用户对艺术家的评分
        self.user_rate_dict = self.get_user_rate()
        # 艺术家与标签的相关度
        self.artists_tags_dict = self.get_artists_tags()
        # 用户对每个标签打标的次数统计和每个标签被所有用户打标的次数统计
        self.user_tag_count_dict, self.tag_count_dict = self.get_user_tag_num()
        # 用户最终对每个标签的喜好程度
        self.user_tag_pre = self.get_user_tag_pre()

    # 获取用户对艺术家的评分信息
    def get_user_rate(self):
        user_rate_dict = dict()

        with open(self.user_rate_file, mode="r", encoding="utf-8") as f_r:
            data = f_r.readlines()

        for line in data:
            if not line.startswith("userID"):
                user_id, artist_id, weight = line.split("\t")
                user_rate_dict.setdefault(int(user_id), {})
                # 对听歌次数进行适当比例的缩放，避免计算结果过大
                user_rate_dict[int(user_id)][int(artist_id)] = float(weight) / 10000
        return user_rate_dict

    # 获取艺术家对应的标签基因,这里的相关度全部为1
    # 由于艺术家和tag过多，存储到一个矩阵中维度太大，这里优化存储结构
    # 如果艺术家有对应的标签则记录，相关度为1，否则不为1
    def get_artists_tags(self):
        artists_tags_dict = dict()
        with open(self.user_tag_file, mode="r", encoding="utf-8") as f_r:
            data = f_r.readlines()
        for line in data:
            if not line.startswith("userID"):
                artists_id, tag_id = line.split("\t")[1:3]
                artists_tags_dict.setdefault(int(artists_id), {})
                artists_tags_dict[int(artists_id)][int(tag_id)] = 1
        return artists_tags_dict

    # 获取每个用户打标的标签和每个标签被所有用户打标的次数
    def get_user_tag_num(self):
        user_tag_count_dict = dict()
        tag_count_dict = dict()
        with open(self.user_tag_file, mode="r", encoding="utf-8") as f_r:
            data = f_r.readlines()

        for line in data:
            if not line.startswith("userID"):
                user_id, artist_id, tag_id = line.strip().split("\t")[:3]
                # 统计每个标签被打标的次数
                tag_count_dict[int(tag_id)] = tag_count_dict.get(int(tag_id), 0) + 1

                # 统计每个用户对每个标签的打标次数
                user_tag_count_dict.setdefault(int(user_id), {})
                user_tag_count_dict[int(user_id)][int(tag_id)] = \
                    user_tag_count_dict[int(user_id)].get(int(tag_id), 0) + 1

        return user_tag_count_dict, tag_count_dict

    # 获取用户对标签的最终兴趣度
    def get_user_tag_pre(self):
        # 用户对标签的喜欢程度
        user_tag_pre = dict()
        # 用户打过该标签的次数
        user_tag_count = dict()
        # Num 为用户打标总条数
        with open(self.user_tag_file, mode="r", encoding="utf-8") as f_r:
            data = f_r.readlines()

        num_lines = len(data)
        for line in data:
            if not line.startswith("userID"):
                user_id, artist_id, tag_id = line.split("\t")[:3]
                user_tag_pre.setdefault(int(user_id), {})
                user_tag_count.setdefault(int(user_id), {})
                # 用户对艺术的评分
                rate_ui = (
                    self.user_rate_dict[int(user_id)][int(artist_id)]
                    if int(artist_id) in self.user_rate_dict[int(user_id)].keys()
                    else 0
                )
                if int(tag_id) not in user_tag_pre[int(user_id)].keys():
                    user_tag_pre[int(user_id)][int(tag_id)] = (
                            rate_ui * self.artists_tags_dict[int(artist_id)][int(tag_id)]
                    )
                    user_tag_count[int(user_id)][int(tag_id)] = 1
                else:
                    user_tag_pre[int(user_id)][int(tag_id)] += (
                            rate_ui * self.artists_tags_dict[int(artist_id)][int(tag_id)]
                    )
                    user_tag_count[int(user_id)][int(tag_id)] += 1

        for user_id in user_tag_pre.keys():
            for tag_id in user_tag_pre[user_id].keys():
                # 用户打该标签的次数/用户打所有标签的次数
                tf_ut = self.user_tag_count_dict[int(user_id)][int(tag_id)] / \
                        sum(self.user_tag_count_dict[int(user_id)].values())
                idf_ut = math.log(num_lines * 1.0 / (self.tag_count_dict[int(tag_id)] + 1))
                user_tag_pre[user_id][tag_id] = \
                    user_tag_pre[user_id][tag_id] / user_tag_count[user_id][tag_id] * tf_ut * idf_ut

        return user_tag_pre

    # 对用户进行艺术家推荐
    def recommend_for_user(self, user, K, flag=True):
        user_artist_pre_dict = dict()
        # 得到用户没有打标过的艺术家
        for artist in self.artists_all:
            if int(artist) in self.artists_tags_dict.keys():
                # 计算用户对艺术的喜好程度
                for tag in self.user_tag_pre[int(user)].keys():
                    rate_ut = self.user_tag_pre[int(user)][int(tag)]
                    rel_it = (
                        0
                        if tag not in self.artists_tags_dict[int(artist)].keys()
                        else self.artists_tags_dict[int(artist)][tag]
                    )
                    if artist in user_artist_pre_dict.keys():
                        user_artist_pre_dict[int(artist)] += rate_ut * rel_it
                    else:
                        user_artist_pre_dict[int(artist)] = rate_ut * rel_it
        new_user_artist_pre_dict = dict()
        if flag:
            # 对推荐结果进行过滤，过滤掉用户已经听过的艺术家
            for artist in user_artist_pre_dict.keys():
                if artist not in self.user_rate_dict[int(user)].keys():
                    new_user_artist_pre_dict[artist] = user_artist_pre_dict[int(artist)]
            return sorted(
                new_user_artist_pre_dict.items(), key=lambda k: k[1], reverse=True
            )[:K]
        else:
            # 表示是用来进行效果评估
            return sorted(
                user_artist_pre_dict.items(), key=lambda k: k[1], reverse=True
            )[:K]

    # 效果评估 重合度
    def evaluate(self, user):
        K = len(self.user_rate_dict[int(user)])
        rec_result = self.recommend_for_user(user, K=K, flag=False)
        count = 0
        for (artist, pre) in rec_result:
            if artist in self.user_rate_dict[int(user)]:
                count += 1
        return count * 1.0 / K


if __name__ == "__main__":
    rbt = RecBasedTag()
    print(rbt.recommend_for_user("4", K=20))
    print(rbt.evaluate("4"))
