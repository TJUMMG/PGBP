import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def cw2se(cw, fix_out_of_bound=False):
    # 创建一个与输入张量 cw 相同形状的全零张量 se
    se = torch.zeros_like(cw)
    
    # 计算起始坐标
    se[..., 0] = cw[..., 0] - cw[..., 1] / 2
    
    # 计算结束坐标
    se[..., 1] = cw[..., 0] + cw[..., 1] / 2
    
    # 如果开启修复越界的选项
    if fix_out_of_bound:
        # 将小于 0.0 的起始坐标修正为 0.0
        se[..., 0][se[..., 0] < 0.0] = 0.0
        # 将大于 1.0 的结束坐标修正为 1.0
        se[..., 1][se[..., 1] > 1.0] = 1.0
    return se

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)