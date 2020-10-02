from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class RNNdiscriminator(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(RNNdiscriminator, self).__init__()

        self.main = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.main(x)


class discCrit(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(RNNdiscriminator, self).__init__()

        self.main = nn.BCECriterion()

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
