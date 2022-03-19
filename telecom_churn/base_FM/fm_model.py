import torch
import torch.nn as nn
import argparse


class FM(nn.Module):
    def __init__(self, args):
        super(FM, self).__init__()
        # 特征的个数
        self.p = args.num_features  # config['num_features']
        # 隐向量的维度
        self.k = args.num_latent_dim
        # FM的线性部分，即∑WiXi
        self.linear = nn.Linear(self.p, 1, bias=True)
        # 隐向量的大小为nxk,即为每个特征学习一个维度为k的隐向量
        self.v = nn.Parameter(torch.randn(self.k, self.p))
        # nn.init.uniform_(self.v, -0.1, 0.1)

        self.ac = nn.Sigmoid()

    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return self.ac(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_features", type=int, default=23)
    parser.add_argument("--num_latent_dim", type=int, default=8)

    args = parser.parse_args()

    x = torch.randn(2, 23)

    model = FM(args)
    out = model(x)

    print(out.shape)
