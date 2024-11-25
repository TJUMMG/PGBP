import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import numpy as np
import math
from .attention import TemporalMaxer,Cross_Attention,MultiHeadAttention
from .operation import Conv1D, mask_logits


class CQFusion(nn.Module):
    def __init__(self, configs, drop_rate=0.0):
        dim = configs.dim
        super(CQFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(
            context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = torch.softmax(mask_logits(score, q_mask.unsqueeze(1)),
                               dim=2)  # (batch_size, c_seq_len, q_seq_len)
        score_t = torch.softmax(mask_logits(score, c_mask.unsqueeze(2)),
                                dim=1)  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t),
                           context)  # (batch_size, c_seq_len, dim)
        output = torch.cat(
            [context, c2q,
             torch.mul(context, c2q),
             torch.mul(context, q2c)],
            dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output * c_mask.unsqueeze(2)

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res

class multiscale_Fusion(nn.Module):
    def __init__(self, configs):
        super(multiscale_Fusion, self).__init__()
        self.branch = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.fusion.append(Cross_Attention(configs))
        self.MULTI_SCALE = configs.MULTI_SCALE
        if configs.MULTI_SCALE == True:
            for idx in range(configs.MULTI_SCALE_LEN):
                self.branch.append(TemporalMaxer(kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                n_embd=configs.dim))
                self.fusion.append(Cross_Attention(configs))  
        self.attention = MultiHeadAttention(configs)
        
    def forward(self, context, query, c_mask, q_mask):
        b,l,d = context.shape
        fusion = self.fusion[0](context,query,c_mask,q_mask)
        if self.MULTI_SCALE == True:
            for i in range(len(self.branch)):
                if i == 0:
                    multi_feature,multi_feature_mask = self.branch[i](context,c_mask)
                else:
                    multi_feature,multi_feature_mask = self.branch[i](multi_feature,multi_feature_mask)
                multi_fusion = self.fusion[i+1](multi_feature,query,multi_feature_mask,q_mask)
                fusion = torch.cat((fusion,multi_fusion),dim = 1)
                c_mask = torch.cat((c_mask,multi_feature_mask),dim = 1)
            fusion = self.attention(fusion,c_mask)
            fusion = fusion[:,:l,:]
            c_mask = c_mask[:,:l]
        return fusion
    
    
class multiscale_CQFusion(nn.Module):
    def __init__(self, configs):
        super(multiscale_CQFusion, self).__init__()
        self.branch = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.fusion.append(CQFusion(configs))
        self.MULTI_SCALE = configs.MULTI_SCALE
        if configs.MULTI_SCALE == True:
            for idx in range(configs.MULTI_SCALE_LEN):
                self.branch.append(TemporalMaxer(kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                n_embd=configs.dim))
                self.fusion.append(CQFusion(configs))  
        self.attention = MultiHeadAttention(configs)
        
    def forward(self, context, query, c_mask, q_mask):
        b,l,d = context.shape
        fusion = self.fusion[0](context,query,c_mask,q_mask)
        if self.MULTI_SCALE == True:
            for i in range(len(self.branch)):
                if i == 0:
                    multi_feature,multi_feature_mask = self.branch[i](context,c_mask)
                else:
                    multi_feature,multi_feature_mask = self.branch[i](multi_feature,multi_feature_mask)
                multi_fusion = self.fusion[i+1](multi_feature,query,multi_feature_mask,q_mask)
            #修改
                # fusion = self.muti_fuse[i](fusion,multi_fusion,multi_feature_mask)
                fusion = torch.cat((fusion,multi_fusion),dim = 1)
                c_mask = torch.cat((c_mask,multi_feature_mask),dim = 1)
        fusion = self.attention(fusion,c_mask)
        fusion = fusion[:,:l,:]
        c_mask = c_mask[:,:l]
        fusion = fusion * c_mask.unsqueeze(2)
        return fusion
    
        
class multiscale_CQFusion1(nn.Module):
    def __init__(self, configs):
        super(multiscale_CQFusion1, self).__init__()
        self.branch = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.muti_fuse = nn.ModuleList()
        self.fusion.append(CQFusion(configs))
        self.MULTI_SCALE = configs.MULTI_SCALE
        self.fusion_attention = configs.fusion_attention
        if configs.MULTI_SCALE == True:
            for idx in range(configs.MULTI_SCALE_LEN):
                self.branch.append(TemporalMaxer(kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                n_embd=configs.dim))
                self.fusion.append(CQFusion(configs))  
                self.muti_fuse.append(MutiFuse(configs))
            self.attention = MultiHeadAttention(configs)
        
    def forward(self, context, query, c_mask, q_mask):
        b,l,d = context.shape
        fusion = self.fusion[0](context,query,c_mask,q_mask)
        if self.fusion_attention is True:
            fusion = self.attention(fusion,c_mask)
        if self.MULTI_SCALE == True:
            for i in range(len(self.branch)):
                if i == 0:
                    multi_feature,multi_feature_mask = self.branch[i](context,c_mask)
                else:
                    multi_feature,multi_feature_mask = self.branch[i](multi_feature,multi_feature_mask)
                multi_fusion = self.fusion[i+1](multi_feature,query,multi_feature_mask,q_mask)
            #修改
                fusion = self.muti_fuse[i](fusion,multi_fusion,multi_feature_mask)
            fusion = fusion * c_mask.unsqueeze(2)
        return fusion

class MutiFuse(nn.Module):
    def __init__(self, cfg):
        super(MutiFuse, self).__init__()
        self.txt_softmax = nn.Softmax(1)
        self.txt_linear1 = nn.Linear(cfg.dim, 1)
        self.layernorm = nn.LayerNorm(cfg.dim, eps=1e-6)
        
    def forward(self, vis_encoded, txt_encoded,txt_mask):
        # vis_encoded: B, C, T
        # txt_encoded: B, L, C
        vis_encoded = vis_encoded.permute(0,2,1)
        txt_attn = self.txt_softmax(self.txt_linear1(txt_encoded))  # B, L, 1
        txt_attn = txt_attn * txt_mask.unsqueeze(2)
        txt_pool = torch.sum(txt_attn * txt_encoded, dim=1)[:,:,None]  # B, C, 1
        # 先计算注意力权重，并在词维度进行sum，最后得到的是一个二维，方便计算所以增加一个第三维度
        vis_fused =self.layernorm((txt_pool * vis_encoded).permute(0,2,1)) + vis_encoded.permute(0,2,1) # B, C, T
        return vis_fused