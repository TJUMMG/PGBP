import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .operation import Conv1D, mask_logits
from .encoder import MultiStepLSTMEncoder, TemporalContextModule
from .phraseEncoder import PhraseEncodeNet

class TemporalMaxer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride,
            padding,
            n_embd):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(
            kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask):

        # out, out_mask = self.channel_att(x, mask)
        x = x.permute(0,2,1)
        mask  = mask.unsqueeze(1)
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=(x.size(-1)+self.stride-1)//self.stride, mode='nearest')
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.to(x.dtype)
        out = out.permute(0,2,1)
        return out, out_mask.squeeze(1)
    
class DETR_Decoder(nn.Module):
    def __init__(self, configs):
        super(DETR_Decoder, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.attention = Cross_Attention(configs)
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, memory, x,mask = None):
        output = self.layer_norm1(memory)
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + memory
        residual = self.layer_norm2(residual)
        output = self.attention(residual,x,mask)
        return output
        
class Cross_Attention(nn.Module):
    def __init__(self, configs):
        super(Cross_Attention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, memory,x,mask = None):
        output = self.layer_norm1(memory)
        x = self.layer_norm3(x)
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(x))
        value = self.transpose_for_scores(self.value(x))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + memory
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(MultiHeadAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiLSTMAttention(nn.Module):
    def __init__(self, configs):
        super(MultiLSTMAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        num_layers = configs.num_layers
        num_step = configs.num_step
        bi_direction = configs.bi_direction

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        self.key = MultiStepLSTMEncoder(in_dim=dim,
                                        out_dim=dim,
                                        num_layers=num_layers,
                                        num_step=num_step,
                                        bi_direction=bi_direction,
                                        drop_rate=drop_rate)
        self.value = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiConvAttention(nn.Module):
    def __init__(self, configs):
        super(MultiConvAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        kernels = configs.kernels

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        self.key = TemporalContextModule(in_dim=dim,
                                         out_dim=dim,
                                         kernels=kernels,
                                         drop_rate=drop_rate)
        self.value = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output
    
class ConvMultiAttention(nn.Module):
    def __init__(self, configs):
        super(ConvMultiAttention, self).__init__()
        self.attention = MultiHeadAttention(configs)
        self.multi_grain = PhraseEncodeNet(configs.dim)
        
    def forward(self, x, mask=None):
        x = self.attention(x,mask)
        x = self.multi_grain(x)
        return x * mask.unsqueeze(2)
    
class ContrastBlock(nn.Module):
    def __init__(self, dim, beta):
        super(ContrastBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=dim,
                                out_channels=dim//beta,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        self.conv2 = nn.Conv1d(in_channels=dim//beta,
                                out_channels=dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self,v_1,v_2):
        v_1 = v_1.transpose(1, 2) 
        v_2 = v_2.transpose(1, 2) 
        v = v_1 * v_2
        v = self.conv1(v)
        v = self.activation(v)
        v = torch.sigmoid(self.layer_norm1(self.conv2(v).transpose(1, 2)))
        v = v * v_1.transpose(1, 2) 
        return v
        
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
