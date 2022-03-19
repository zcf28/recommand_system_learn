import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):

    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.raw_file_path = args.raw_file_path
        self.train_file_path = args.train_file_path

        self.total_data, self.train_data, self.test_data = self.process_data()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        data = self.train_data[item]
        feature = torch.tensor(data["feature"], dtype=torch.float32)
        label = torch.tensor([data["label"]], dtype=torch.float32)

        return feature, label

    @staticmethod
    def split_data(total_data):

        random.shuffle(total_data)
        lens = len(total_data)
        train_data = total_data[(lens // 7) + 1:]
        test_data = total_data[:lens // 7]

        return train_data, test_data

    @staticmethod
    def standard_scale(data):  # 自定义标准差标准化函数
        data = (data - data.mean()) / data.std()
        return data

    def process_data(self):
        total_data = []
        if os.path.exists(self.train_file_path):
            df = pd.read_csv(self.train_file_path, index_col="customerID")

            for index, row in df.iterrows():
                feature = row.iloc[:-1]
                label = row.iloc[-1]
                total_data.append({"feature": np.array(feature.values, dtype=np.float32), "label": label})

        else:
            df = pd.read_csv(self.raw_file_path, index_col="customerID")

            columns_list = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                            "MultipleLines",
                            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                            "StreamingTV",
                            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
                            "TotalCharges", "Churn"
                            ]

            for column in columns_list:
                status_dict = df[column].unique().tolist()
                df[column] = df[column].apply(lambda x: status_dict.index(x))
                if column != "Churn":
                    df[column] = self.standard_scale(df[column])

            for index, row in df.iterrows():
                feature = row.iloc[:-1]
                label = row.iloc[-1]
                total_data.append(
                    {"feature": np.array(feature.values, dtype=np.float32), "label": np.array(label, dtype=np.int32)})

            df.to_csv(self.train_file_path, index="customerID")

        train_data, test_data = self.split_data(total_data)

        return total_data, train_data, test_data


class TestDataset(Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        train_dataset = TrainDataset(args)
        self.total_data, self.train_data, self.test_data = train_dataset.process_data()

    def __getitem__(self, item):
        data = self.test_data[item]
        feature = torch.tensor(data["feature"], dtype=torch.float32)
        label = torch.tensor([data["label"]], dtype=torch.float32)

        return feature, label

    def __len__(self):
        return len(self.test_data)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()

    parse.add_argument("--raw_file_path", type=str, default="../dataset/telecom-churn-prediction-data.csv")
    parse.add_argument("--train_file_path", type=str, default="./dataset/telecom-churn-prediction-train-data.csv")

    args = parse.parse_args()

    train_dataset = TrainDataset(args)

    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for feature, label in train_data_loader:
        print(feature, label)
        break

    test_dataset = TestDataset(args)
    test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    for feature, label in test_data_loader:
        print(feature, label)
        break
