import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
from core.config import config
from core.runner_utils import index_to_time2, calculate_iou, calculate_iou_accuracy, cal_statistics
from . import attention
from .encoder import LSTMEncoder, MultiStepLSTMEncoder, TemporalContextModule
from . import fusion
from .layers import Projection, Prediction, PositionalEmbedding, PositionEmbeddingSine
from .operation import Conv1D, mask_logits,cw2se
from .triplet_loss import batch_all_triplet_loss, pairwise_distances
import random
from .slidewindow import find_most_relevant_frame
from .gauss import generate_gaussian_tensor
from einops import repeat,rearrange
from .decoder import TransformerDecoder,TransformerDecoderLayer
from torchvision.ops import sigmoid_focal_loss
from .matcher import HungarianMatcher
# torch.set_printoptions(profile="full", linewidth=1000, precision=2)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class PGBP(nn.Module):
    def __init__(self):
        super(PGBP, self).__init__()
        configs = config.MODEL.PARAMS 
        self.use_keyword = configs.use_keyword
        self.windowsize = configs.windowsize
        self.debug_print = configs.DEBUG
        self.top_k = configs.top_k
        self.top_k0=configs.top_k0
        self.neg = configs.neg
        self.pos =configs.pos
        self.detr_layers = configs.detr_layers
        self.content_prior = configs.content_prior
        self.match = HungarianMatcher(configs.cost_class,configs.cost_span,configs.cost_giou)
        empty_weight = torch.ones(2)
        empty_weight[-1] = configs.eos_coef # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        self.video_affine = Projection(in_dim=configs.video_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)

        self.query_affine = Projection(in_dim=configs.query_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)
        self.query_position = configs.query_position
        self.video_position = configs.video_position
        if self.query_position:
            self.q_pos_embedding = PositionalEmbedding(configs.dim, 30)
        if self.video_position:
            self.v_pos_embedding = PositionEmbeddingSine(configs.dim, normalize=True)
        if configs.content_prior == "learned":
            self.pattern = nn.Embedding(configs.num_queries, configs.dim)    
            # self.pos_embedding = TransformerPositionalEmbedding(configs.dim, 500,drop_rate=configs.drop_rate)
        self.query_embeddings = nn.Embedding(configs.num_queries, 2)
        query_attention_layer = getattr(attention,
                                        configs.query_attention)(configs)
        video_attention_layer = getattr(attention,
                                        configs.video_attention)(configs)
        decoder_layer = TransformerDecoderLayer(configs)
        decoder_norm = nn.LayerNorm(configs.dim)
        self.detr_decoder = TransformerDecoder(decoder_layer,configs,decoder_norm)
        self.query_encoder = nn.Sequential(*[
            copy.deepcopy(query_attention_layer)
            for _ in range(configs.query_attention_layers)
        ])
        self.video_encoder = nn.Sequential(*[
            copy.deepcopy(video_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])
        early_attention_layer = getattr(attention,
                                        configs.early_attention)(configs)
        self.early_encoder = nn.Sequential(*[
            copy.deepcopy(early_attention_layer)
            for _ in range(configs.early_attention_layers)
        ])
        self.contrastlayer = copy.deepcopy(video_attention_layer)
        self.fg_prediction_layer = Prediction(in_dim=configs.dim,
                                              hidden_dim=configs.dim // 2,
                                              out_dim=1,
                                              drop_rate=configs.drop_rate)
        self.early_fusion_layer = getattr(fusion,
                                          configs.early_fusion_module)(configs)
        
        self.fusion_layer = getattr(fusion, configs.fusion_module)(configs)

        post_attention_layer = getattr(attention,
                                       configs.post_attention)(configs)
        self.post_attention_layer = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.post_attention_layers)
        ])
        self.video_encoder2 = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])
        self.linear = nn.Linear(in_features=2*configs.dim,out_features=configs.dim,bias= True)
        cw_pred = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=2,
                                   drop_rate=configs.drop_rate)
        self.cw_pred = nn.Sequential(*[
            copy.deepcopy(cw_pred)
            for _ in range(configs.detr_layers)
        ])
        pred_results = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=2,
                                   drop_rate=configs.drop_rate)
        self.pred_results = nn.Sequential(*[
            copy.deepcopy(pred_results)
            for _ in range(configs.detr_layers)
        ])
        self.intering = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=1,
                                   drop_rate=configs.drop_rate)
        self.pos_fused_layer =attention.ContrastBlock(configs.dim,configs.beta)
        self.neg_fused_layer =attention.ContrastBlock(configs.dim,configs.beta)
        self.pn_fused_layer =attention.ContrastBlock(configs.dim,configs.beta)
        
    def forward(self, batch_visual_scale,batch_word_vectors, batch_keyword_mask, batch_txt_mask,
                batch_vis_feats, batch_vis_mask):
        batch_vis_feats = self.video_affine(batch_vis_feats)
        batch_vis_feats = batch_vis_feats * batch_vis_mask.unsqueeze(2)
        for i, module in enumerate(self.video_encoder):
            if i == 0:
                video_features = module(batch_vis_feats, batch_vis_mask)
            else:
                video_features = module(video_features, batch_vis_mask)
        for i, module in enumerate(self.video_encoder2):
            if i == 0:
                video_features2 = module(batch_vis_feats, batch_vis_mask)
            else:
                video_features2 = module(video_features2, batch_vis_mask)

        batch_word_vectors = self.query_affine(batch_word_vectors)
        if self.query_position:
            batch_word_vectors = batch_word_vectors + self.q_pos_embedding(
                batch_word_vectors)
        batch_word_vectors = batch_word_vectors * batch_txt_mask.unsqueeze(2)
        for i, module in enumerate(self.query_encoder):
            if i == 0:
                query_features = module(batch_word_vectors, batch_txt_mask)
            else:
                query_features = module(query_features, batch_txt_mask)
        if self.use_keyword:
            entity_features = batch_word_vectors * batch_keyword_mask.unsqueeze(2)
            entity_features = query_features + entity_features
        else:
            entity_features = query_features        
        # First stage
        entity_video_fused = self.early_fusion_layer(video_features,
                                                     entity_features,
                                                     batch_vis_mask,
                                                     batch_txt_mask)
        for i, module in enumerate(self.early_encoder):
            entity_video_fused = module(entity_video_fused, batch_vis_mask)
        fg_prob = self.fg_prediction_layer(entity_video_fused)

        fg_prob1 =torch.sigmoid(fg_prob.squeeze(2)) 

        pos_values, pos_indices = torch.topk(fg_prob1.masked_fill(~batch_vis_mask.bool(), float('0.0')), k=self.top_k0, dim=1, largest=True)
        neg_values, neg_indices = torch.topk(fg_prob1.masked_fill(~batch_vis_mask.bool(), float('1.0')), k=self.top_k, dim=1, largest=False)
        B,l,c = entity_video_fused.shape
        if self.top_k0>1:
            pos=torch.gather(entity_video_fused, dim=1, index=pos_indices.unsqueeze(-1).expand(-1, -1, c))
            pos=F.max_pool1d(pos.transpose(1,2),kernel_size=self.top_k0).transpose(1,2)
        else:
            pos = torch.gather(entity_video_fused, 1, pos_indices.view(-1, 1).expand(-1, c).unsqueeze(1))
        neg = torch.gather(entity_video_fused, dim=1, index=neg_indices.unsqueeze(-1).expand(-1, -1, c))
        if not self.training and self.debug_print:
            print('fg_prob', torch.sigmoid(fg_prob))
        fg_vis_feature = (video_features2 +
                            video_features) * torch.sigmoid(fg_prob) 
        fused_pos_feature = self.pos_fused_layer(fg_vis_feature,pos)
        contrast_feature = self.contrastlayer(fg_vis_feature,batch_vis_mask) 
        if self.pos is True:
            contrast_feature = contrast_feature + fused_pos_feature
        if self.neg is True:
            fused_neg_feature = torch.mean(self.neg_fused_layer(neg,pos),dim= 1).unsqueeze(1)
            fused_pn_feature = fg_vis_feature - self.pn_fused_layer(fg_vis_feature,fused_neg_feature)
            contrast_feature =contrast_feature + fused_pn_feature
        fg_vis_feature = torch.cat((fg_vis_feature,contrast_feature),dim=2)
        fg_vis_feature = self.linear(fg_vis_feature)
        fused_action_feature = self.fusion_layer(fg_vis_feature,
                                                 entity_features,
                                                 batch_vis_mask,
                                                 batch_txt_mask)
        for i, module in enumerate(self.post_attention_layer):
            fused_action_feature = module(fused_action_feature, batch_vis_mask)
        query_embeddings = self.query_embeddings.weight
        refpoint_embed = repeat(query_embeddings, "nq d -> b nq d", b=B).transpose(0,1)
        if self.content_prior == "learned":
            pattern = self.pattern.weight
            tgt = repeat(pattern, "nq d -> b nq d", b=B).transpose(0,1)
        else:
            tgt = torch.zeros(refpoint_embed.shape[0],B,c).cuda()
        pred_start = []
        pred_end = []
        results = []
        memory_local = fused_action_feature.permute(1, 0, 2)
        pos_embed_local = self.v_pos_embedding(fused_action_feature,batch_vis_mask).permute(1, 0, 2)
        hs, references = self.detr_decoder(pos,batch_visual_scale,tgt, memory_local, memory_key_padding_mask=batch_vis_mask,
                    pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed) 
        reference_before_sigmoid = inverse_sigmoid(references)
        for i in range(self.detr_layers):
            results.append(self.pred_results[i](hs[i,...]).squeeze(2))
            d_cw = self.cw_pred[i](hs[i,...])
            cw = (reference_before_sigmoid[i,...] + d_cw)
            se = cw2se(torch.sigmoid(cw))
            pred_start.append(se[...,0]) 
            pred_end.append(se[...,1]) 
        pred_inter = self.intering(fused_action_feature).squeeze(2)


        return pred_start,pred_end,pred_inter, query_features, video_features2, fg_prob.squeeze(
            2), video_features, batch_word_vectors, batch_vis_feats,results,pos_indices,neg_indices,\
            contrast_feature
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length
    
    def contrast_loss(self,mask,key_frame,inter_label,contrast_feature,hy_sigma=1,weight = True):
        b,l,c = contrast_feature.shape
        gauss_weights = generate_gaussian_tensor(inter_label, key_frame, hy_sigma)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=2)
        key_frame_feature = torch.gather(contrast_feature, 1, key_frame.view(-1, 1).expand(-1, c).unsqueeze(1))
        score = torch.bmm(contrast_feature,key_frame_feature.transpose(1,2)).squeeze(2)
        loss = nn.BCEWithLogitsLoss(reduction='none')(score,inter_label)
        if weight is True:
            loss = loss * gauss_weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss * mask,
                         dim=1) / (torch.sum(mask, dim=1) + 1e-13)
        return loss.mean()
    
    def PNcontrast_loss(self,mask,pos_frame,neg_frame,inter_label,contrast_feature,hy_sigma=1,weight = True):
        if self.top_k0>1:
            pos_loss = self.contrast_loss(mask,pos_frame[:,0],inter_label,contrast_feature,hy_sigma,False)
        else:
            pos_loss = self.contrast_loss(mask,pos_frame,inter_label,contrast_feature,hy_sigma,False)
        B,l = neg_frame.shape
        neg_loss = 0.
        if self.neg is True:
            for i in range(l):  
                neg_loss = neg_loss + self.contrast_loss(mask,neg_frame[:,i],(1.0-inter_label),contrast_feature,hy_sigma,weight)
        return pos_loss + neg_loss/l

    def compute_loss(self, pred_start,pred_end, pred_inter, start_labels,
                     end_labels, inter_label, mask,duration,pred_pro):
        bce_loss,iou_loss,L1_loss = 0,0,0
        for i in range(len(pred_start)):
            pred_times = torch.cat([pred_start[i].unsqueeze(2),pred_end[i].unsqueeze(2)],dim=2)
            b,l,_ = pred_times.shape
            times = torch.cat([(start_labels/duration).unsqueeze(1),(end_labels/duration).unsqueeze(1)],dim=1)
            indices = self.match(pred_pro[i],pred_times,times)
            idx = self._get_src_permutation_idx(indices)
            src_spans = pred_times[idx]
            L1_loss =L1_loss + F.l1_loss(src_spans, times, reduction='none').mean()
            iou_loss = iou_loss + (1- self.calculate_giou(src_spans, times)[1]).mean()
            target_classes = torch.full(pred_pro[i].shape[:2], 1,
                                    dtype=torch.int64, device=pred_pro[i].device)
            target_classes[idx] = 0
            bce_loss = bce_loss + self.bce_rescale_loss(pred_pro[i],target_classes)
            
        inter_loss = self.compute_location_loss(pred_inter, inter_label, mask)
        return L1_loss, inter_loss, iou_loss,bce_loss
    
    def bce_rescale_loss(self,scores, targets):
        loss_value = F.cross_entropy(scores.transpose(1, 2),targets,self.empty_weight, reduction="none")
        loss_value = loss_value.mean()
        return loss_value
    
    def calculate_giou(self,box1, box2):
        iou,union = self.calculate_iou(box1,box2)
        box1_left, box1_right = box1[..., 0], box1[..., 1]
        box2_left, box2_right = box2[..., 0], box2[..., 1]
        right = torch.maximum(box2_right, box1_right)
        left = torch.minimum(box2_left, box1_left)
        enclosing_area = (right - left).clamp(min=0)
        giou = iou - (enclosing_area - union) / enclosing_area
        return iou,giou
    
    def calculate_iou(self,box1, box2):
        box1_left, box1_right = box1[..., 0], box1[..., 1]
        box2_left, box2_right = box2[..., 0], box2[..., 1]
        areas1 = box1_right-box1_left
        areas2 = box2_right-box2_left
        inter_left = torch.maximum(box1_left, box2_left)
        inter_right = torch.minimum(box1_right, box2_right)
        inter = (inter_right - inter_left).clamp(min=0)
        union = areas1 + areas2 - inter
        iou = inter/ union
        return iou,union    
        
    def compute_boundary_loss(self, pred, targets):
        return F.cross_entropy(pred, targets.long())

    def compute_location_loss(self, pred, targets, mask):
        weights_per_location = torch.where(targets == 0.0, targets + 1.0,
                                           1.0 * targets)
        loss_per_location = nn.BCEWithLogitsLoss(reduction='none')(pred,
                                                                   targets)
        loss_per_location = loss_per_location * weights_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask,
                         dim=1) / (torch.sum(mask, dim=1) + 1e-13)
        return loss.mean()
    

    def compute_sim_loss(self, pred, pos, neg, saliency_margin = 0.2):
        b, l = pred.shape
        _, num_indices = pos.shape
        pos_indices = pos + (torch.arange(0, b).reshape(-1, 1) * l).cuda()
        neg_indices = neg + (torch.arange(0, b).reshape(-1, 1) * l).cuda()
        pred_score = torch.sigmoid(pred)
        pos_scores = pred_score.view(-1)[pos_indices.view(-1)].view(b, num_indices)
        neg_scores = pred_score.view(-1)[neg_indices.view(-1)].view(b, num_indices)
        loss_sim = torch.clamp(saliency_margin + neg_scores - pos_scores, min=0).sum() \
                / (b * num_indices) * 2  # * 2 to keep the loss the same scale
        return loss_sim

    
    def early_pred_loss(self, video_features, pred, targets, mask):
        return self.compute_location_loss(pred, targets, mask)
        
    def aligment_score(self,
                       query_features,
                       video_features,
                       query_mask,
                       video_mask,
                       inner_label,
                       GT_inner=True):
        B, T, channels = video_features.shape

        query_features = query_features.sum(1) / query_mask.sum(1).unsqueeze(1)
        query_features = F.normalize(query_features, p=2, dim=1)  # B, channels

        if GT_inner:
            frame_weights = inner_label / video_mask.sum(1, keepdim=True)
        else:
            norm_video = F.normalize(video_features, p=2, dim=-1)
            frame_weights = torch.bmm(query_features.unsqueeze(1),
                                      norm_video.transpose(1, 2))  # B,1,T
            frame_weights = mask_logits(frame_weights.squeeze(1),
                                        video_mask)  # B,T
            frame_weights = torch.softmax(frame_weights, dim=-1)

        video_features = video_features * frame_weights.unsqueeze(2)
        video_features = video_features.sum(1)
        video_features = F.normalize(video_features, p=2, dim=1)
        video_sim = torch.matmul(video_features, video_features.T)
        video_sim = torch.softmax(video_sim, dim=-1)
        query_sim = torch.matmul(query_features, query_features.T)
        query_sim = torch.softmax(query_sim, dim=-1)
        kl_loss = (F.kl_div(query_sim.log(), video_sim, reduction='sum') +
                   F.kl_div(video_sim.log(), query_sim, reduction='sum')) / 2

        return kl_loss

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2),
                             end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0],
                                   dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0],
                                 dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def eval_test(model,
                  data_loader,
                  device,
                  mode='test',
                  epoch=None,
                  shuffle_video_frame=False):
        ious = []
        pos_labels = []
        pseudo=[]
        preds, durations,names,times = [], [],[],[]
        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(data_loader),
                                        total=len(data_loader),
                                        desc='evaluate {}'.format(mode)):
                data, annos = batch_data
                batch_word_vectors = data['batch_word_vectors'].to(device)
                batch_keyword_mask = data['batch_keyword_mask'].to(device)
                batch_txt_mask = data['batch_txt_mask'].squeeze(2).to(device)
                batch_vis_feats = data['batch_vis_feats'].to(device)
                batch_vis_mask = data['batch_vis_mask'].squeeze(2).to(device)
                batch_extend_pre = data['batch_extend_pre'].to(device)
                batch_extend_suf = data['batch_extend_suf'].to(device)
                batch_visual_scale = data["visual_scale"].unsqueeze(-1).to(device)
                if shuffle_video_frame:
                    B = batch_vis_feats.shape[0]
                    for i in range(B):
                        T = batch_vis_mask[i].sum().int().item()
                        pre = batch_extend_pre[i].item()
                        new_T = torch.randperm(T)
                        batch_vis_feats[i, torch.arange(T) +
                                        pre] = batch_vis_feats[i, new_T + pre]
                # compute predicted results
                with torch.cuda.amp.autocast():
                    output = model(batch_visual_scale,batch_word_vectors, batch_keyword_mask,
                                   batch_txt_mask, batch_vis_feats,
                                   batch_vis_mask)
                pseudo_pros=output[5]
                probalities_class = torch.softmax(output[9][-1],dim = -1)
                probalities = probalities_class[...,0]
                pred_p = torch.argmax(probalities,dim = 1)
                start_logits, end_logits = output[0][-1], output[1][-1]
                start_logits = torch.gather(start_logits, 1, pred_p.view(-1, 1)).clamp(0,1).squeeze(1)
                start_logits[torch.isnan(start_logits)] = 0.
                end_logits = torch.gather(end_logits, 1, pred_p.view(-1, 1)).clamp(0,1).squeeze(1)
                end_logits[torch.isnan(end_logits)] = 1.
                pos_frames = output[-3]

                start_indices = start_logits.cpu().numpy()
                end_indices = end_logits.cpu().numpy()
                batch_vis_mask = batch_vis_mask.cpu().numpy()
                batch_extend_pre = batch_extend_pre.cpu().numpy()
                batch_extend_suf = batch_extend_suf.cpu().numpy()
                pos_frames = pos_frames.cpu().numpy()


                for vis_mask, start_index, end_index, extend_pre, extend_suf, anno,pos_frame,pseudo_pro in zip(
                        batch_vis_mask, start_indices, end_indices,
                        batch_extend_pre, batch_extend_suf, annos,pos_frames,pseudo_pros):

                    start_time, end_time = index_to_time2(
                        start_index, end_index, vis_mask.sum(), extend_pre,
                        extend_suf, anno["duration"])

                    iou = calculate_iou(i0=[start_time, end_time],
                                        i1=anno['times'])
                    ious.append(iou)
                    preds.append((start_time, end_time))
                    durations.append(anno["duration"])
                    times.append(anno["times"])
                    names.append(anno["video"])
                    pseudo.append(pseudo_pro)
        import pandas as pd
        df = pd.DataFrame({
            'Column1': names,
            'Column2': times,
            'Column3': preds,
            "Column4":pseudo
        })    
        df.to_excel('output.xlsx', index=False, engine='openpyxl')        

        statistics_str = cal_statistics(preds, durations)
        r1i1 = calculate_iou_accuracy(ious, threshold=0.1)
        r1i2 = calculate_iou_accuracy(ious, threshold=0.2)
        r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i4 = calculate_iou_accuracy(ious, threshold=0.4)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i6 = calculate_iou_accuracy(ious, threshold=0.6)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        r1i8 = calculate_iou_accuracy(ious, threshold=0.8)
        r1i9 = calculate_iou_accuracy(ious, threshold=0.9)

        mi = np.mean(ious) * 100.0
        # write the scores
        score_str = "Epoch {}\n".format(epoch)
        score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
        score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
        score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
        score_str += "mean IoU: {:.2f}\n".format(mi)
        return r1i3, r1i5, r1i7, mi, score_str, statistics_str
