import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from .operation import Conv1D, mask_logits
from torchvision.ops import RoIAlign
from .layers import Prediction

class MultiheadAttention(nn.Module):
    def __init__(self, dim,num_heads,dropout,dim_v):
        super(MultiheadAttention, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.head_size_v = int(dim_v/num_heads)
        self.dim_v = dim_v
        self.dropout = nn.Dropout(p=dropout)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)
        
    def transpose_for_scores_v(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size_v)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)
    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, q,k,v, mask=None):
        query = self.transpose_for_scores(
            q.permute(1, 0, 2))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(k.permute(1, 0, 2))
        value = self.transpose_for_scores_v(v.permute(1, 0, 2))
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
        return value.permute(1, 0, 2)
   
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, configs, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, configs.detr_layers)
        self.detr_layers = configs.detr_layers
        self.norm = norm
        self.return_intermediate = configs.return_intermediate
        assert configs.return_intermediate
        self.query_dim = configs.query_dim
        self.dim = configs.dim
        self.norm1 = nn.LayerNorm(configs.dim)
        self.norm2 = nn.LayerNorm(configs.dim)
        assert configs.query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = configs.query_scale_type
        if configs.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(configs.dim, configs.dim, configs.dim, 2)
        elif configs.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(configs.dim, configs.dim, 1, 2)
        elif configs.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(configs.detr_layers, configs.dim)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(configs.query_scale_type))

        self.ref_point_head = MLP(configs.dim, configs.dim, configs.dim, 2)

        # self.bbox_embed = None
        # for DAB-deter
        if configs.bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(configs.dim, configs.dim, 2, 3) for i in range(configs.detr_layers)])
        else:
            self.bbox_embed = MLP(configs.dim, configs.dim, 2, 3)
        # init bbox_embed
        if configs.bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model =configs.dim
        self.modulate_t_attn = configs.modulate_t_attn
        self.bbox_embed_diff_each_layer = configs.bbox_embed_diff_each_layer

        if configs.modulate_t_attn:
            self.ref_anchor_head = MLP(configs.dim, configs.dim, 1, 2)

        if not configs.keep_query_pos:
            for layer_id in range(configs.detr_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
 
    def forward(self,pos_feature,scale,tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output =self.norm1(tgt)  #torch.Size([10, 32, 256])
        memory = self.norm2(memory)
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        # import ipdb; ipdb.set_trace()

        for layer_id, layer in enumerate(self.layers): #rence_points torch.Size([10, 32, 2])
            obj_center = reference_points[..., :self.query_dim]#torch.Size([10, 32, 2])
            # get sine embedding for the query vector 
            query_sine_embed = gen_sineembed_for_position(obj_center,self.dim//2)
            # print('line230', query_sine_embed.shape)
            query_pos = self.ref_point_head(query_sine_embed) #torch.Size([10, 32, 256])
            # print('line232',query_sine_embed.shape)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            # print(query_sine_embed.shape) # 10 32 512
            query_sine_embed = query_sine_embed * pos_transformation

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 1
                # print(reft_cond.shape, reft_cond[..., 0].shape) # 10 32 1, 10 32
                # print(obj_center.shape, obj_center[..., 1].shape) # 10 32 2, 10 32
                # print(query_sine_embed.shape) # 10 32 256

                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

            output = layer(pos_feature,scale,reference_points,output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0)) #torch.Size([10, 32, 256])

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.detr_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach() #torch.Size([10, 32, 2])

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)   
     
class TransformerDecoderLayer(nn.Module):

    def __init__(self, configs):
        super().__init__()
        # Decoder Self-Attention
        d_model = configs.dim
        nhead =configs.num_heads
        rm_self_attn_decoder = configs.rm_self_attn_decoder
        dropout = configs.dropout
        dim_feedforward = configs.feedforward
        beta = configs.beta
        self.sementic_fu = configs.sementic_fu
        self.aligned_len = configs.aligned_len
        
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead,dropout,dim_v=d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead,dropout, dim_v=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(configs.activation)
        self.normalize_before = configs.normalize_before
        self.keep_query_pos = configs.keep_query_pos
        if self.sementic_fu is True:
            self.sementic_fusion = semantic_align(d_model,dropout,beta,self.aligned_len)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, pos_feature,scale,ref_points,tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False,
                ):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, v, mask=tgt_key_padding_mask)
            # ========== End of Self-Attention =============
            box = ref_points.transpose(0,1) * scale.unsqueeze(1)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            if self.sementic_fu is True:
                tgt3 = self.sementic_fusion(memory.transpose(0,1),box,tgt.transpose(0,1),pos_feature)
                tgt3 =tgt + self.dropout4(tgt3)
                tgt = self.norm4(tgt3)
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(q,k,v,mask=memory_key_padding_mask)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, detr_layers):
        super().__init__()
        self.detr_layers = detr_layers
        h = [hidden_dim] * (detr_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.detr_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor,dim):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    # dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / dim)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class semantic_align(nn.Module):
    def __init__(self, dim, dropout,beta,aligned_len):
        super().__init__()
        self.aligned_len = aligned_len
        self.gate = Prediction(in_dim= 2*dim, hidden_dim= dim, out_dim=2,drop_rate=dropout)
        self.softmax = nn.Softmax(2)
        self.contrast1 = ContrastBlock(dim,beta)
        self.contrast2 = ContrastBlock(dim,beta)
    def forward(self,features,quires_box,quires_features,pos_feature):
        B, L1, _ = quires_box.shape
        _,L,C = features.shape
        batch_feature = []
        roi_start = torch.round(((quires_box[..., 0] - quires_box[..., 1] / 2)*L).clamp(0, L-1)).long()
        roi_end = torch.round(((quires_box[..., 0] + quires_box[..., 1] / 2)*L).clamp(0, L-1)).long()
        start_features = torch.gather(features, dim=1, index=roi_start.unsqueeze(-1).expand(-1, -1, C))
        start_features = self.contrast1(start_features,pos_feature).unsqueeze(-2)
        end_features = torch.gather(features, dim=1, index=roi_end.unsqueeze(-1).expand(-1, -1, C))
        end_features = self.contrast2(end_features,pos_feature).unsqueeze(-2)
        boundary_features = torch.cat((start_features,end_features),dim = -2)
        if self.aligned_len:
            pool_boundary_features = torch.mean(boundary_features, dim=2, keepdim=False)
        else:
            pool_boundary_features,_ = torch.max(boundary_features, dim=2, keepdim=False)
        x = torch.cat([pool_boundary_features ,quires_features],dim = -1)
        gate =self.softmax(self.gate(x))
        x = pool_boundary_features*gate[...,0:1] + quires_features*gate[...,1:2]
        return x.transpose(0,1)
        

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
# class semantic_align(nn.Module):
#     def __init__(self, dim,beta,aligned_len):
#         super().__init__()
#         self.aligned_len = aligned_len
#         self.gate = nn.Linear(in_features=2*dim,out_features=2)
#         self.softmax = nn.Softmax(2)
#         self.contrast = ContrastBlock(dim,beta)
#     def forward(self, features,quires_box,quires_features,pos_feature):
#         B, L1, _ = quires_box.shape
#         _,L,C = features.shape
#         batch_feature = []
#         roi_start = torch.round(((quires_box[..., 0] - quires_box[..., 1] / 2)*L).clamp(0, L-1)).long()
#         roi_end = torch.round(((quires_box[..., 0] + quires_box[..., 1] / 2)*L).clamp(0, L-1)).long()
#         start_features = torch.gather(features, dim=1, index=roi_start.unsqueeze(-1).expand(-1, -1, C)).unsqueeze(-2)
#         end_features = torch.gather(features, dim=1, index=roi_end.unsqueeze(-1).expand(-1, -1, C)).unsqueeze(-2)
#         boundary_features = torch.cat((start_features,end_features),dim = -2)
#         boundary_features = self.contrast(boundary_features,pos_feature)
#         if self.aligned_len:
#             pool_boundary_features = torch.mean(boundary_features, dim=2, keepdim=False)
#         else:
#             pool_boundary_features,_ = torch.max(boundary_features, dim=2, keepdim=False)
#         x = torch.cat([pool_boundary_features ,quires_features],dim = -1)
#         gate =self.softmax(self.gate(x))
#         x = pool_boundary_features*gate[...,0:1] + quires_features*gate[...,1:2]
#         return x.transpose(0,1)
        

# class ContrastBlock(nn.Module):
#     def __init__(self, dim, beta):
#         super(ContrastBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=dim,
#                                 out_channels=dim//beta,
#                                 kernel_size=1,
#                                 stride=1,
#                                 padding=0,
#                                 bias=True)
#         self.conv2 = nn.Conv1d(in_channels=dim//beta,
#                                 out_channels=dim,
#                                 kernel_size=1,
#                                 stride=1,
#                                 padding=0,
#                                 bias=True)
#         self.activation = nn.ReLU()
#         self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        
#     def forward(self,v_1,v_2):
#         v_1 = v_1.transpose(1, 2) 
#         v_2 = v_2.transpose(1, 2) 
#         v = v_1 * v_2
#         v = self.conv1(v)
#         v = self.activation(v)
#         v = torch.sigmoid(self.layer_norm1(self.conv2(v).transpose(1, 2)))
#         v = v * v_1.transpose(1, 2) 
#         return v