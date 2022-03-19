import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fm_model import FM

from data_loader import TrainDataset, TestDataset


def train(args):
    logging.basicConfig(filemode="w", filename=f"./run.log", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    model = FM(args).to(device)
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    train_dataset = TrainDataset(args)
    test_dataset = TestDataset(args)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for epoch in range(1, args.epochs + 1):

        train_avg_loss = 0.0
        test_avg_loss = 0.0

        for train_feature, train_label in train_data_loader:
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)

            out = model(train_feature)

            # out [[0.1],[0.9],[0.5]] train_label [[0], [1], [0]]
            loss = criterion(out, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_avg_loss += loss.item()

        scheduler.step()

        logging.info(f"train loss: {train_avg_loss / len(train_data_loader):0.6f}")

        if epoch % 5 == 0:
            model.eval()
            for test_feature, test_label in test_data_loader:
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)

                test_out = model(test_feature)
                test_loss = criterion(test_out, test_label)
                test_avg_loss += test_loss.item()
            logging.info(f"*" * 10)
            logging.info(f"test loss : {test_avg_loss / len(test_data_loader):0.6f}")
            logging.info(f"*" * 10)

            model.train()

    torch.save(model.state_dict(), args.save_model_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--epochs", type=int, default=200)
    parse.add_argument("--num_features", type=int, default=19)
    parse.add_argument("--num_latent_dim", type=int, default=8)
    parse.add_argument("--seed", type=int, default=42)

    parse.add_argument("--raw_file_path", type=str, default="../dataset/telecom-churn-prediction-data.csv")
    parse.add_argument("--train_file_path", type=str, default="./dataset/telecom-churn-prediction-train-data.csv")
    parse.add_argument("--save_model_path", type=str, default="./dataset/model.pth")

    args = parse.parse_args()

    train(args)
