import torch
import torch.nn as nn
from torch.nn.utils import rnn
from core.config import config
import numpy as np


def collate_fn(batch):
    batch_word_vectors = [b[0]['word_vectors'] for b in batch]
    # batch_pos_tags = [b[0]['pos_tags'] for b in batch]
    batch_txt_mask = [b[0]['txt_mask'] for b in batch]
    batch_vis_feats = [b[0]['visual_input'] for b in batch]
    batch_vis_mask = [b[0]['vis_mask'] for b in batch]
    batch_start_label = [b[0]['start_label'] for b in batch]
    batch_end_label = [b[0]['end_label'] for b in batch]
    batch_start_frame = [b[0]['start_frame'] for b in batch]
    batch_end_frame = [b[0]['end_frame'] for b in batch]
    batch_internel_label = [b[0]['internel_label'] for b in batch]
    batch_extend_pre = [b[0]['extend_pre'] for b in batch]
    batch_extend_suf = [b[0]['extend_suf'] for b in batch]
    batch_keyword_mask = [b[0]['keyword_mask'] for b in batch]
    batch_negative_indices =np.array([b[0]['selected_negative_indices'] for b in batch]) 
    batch_positive_indices = np.array([b[0]['selected_positive_indices'] for b in batch])
    batch_time = [b[1]["times"] for b in batch]
    batch_duration = [b[1]["duration"] for b in batch]
    annotations = [b[1] for b in batch]
    visual_len = torch.from_numpy(np.array([b[0]['visual_len'] for b in batch])).float()
    visual_scale = visual_len / torch.max(visual_len)
    batch_data = {
        'batch_word_vectors':
        nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        # 'batch_pos_tags':
        # rnn.pad_sequence(batch_pos_tags, batch_first=True),
        'batch_txt_mask':
        nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_vis_feats':
        nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_vis_mask':
        nn.utils.rnn.pad_sequence(batch_vis_mask, batch_first=True).float(),
        'batch_start_label':
        nn.utils.rnn.pad_sequence(batch_start_label, batch_first=True).float(),
        'batch_end_label':
        nn.utils.rnn.pad_sequence(batch_end_label, batch_first=True).float(),
        'batch_internel_label':
        nn.utils.rnn.pad_sequence(batch_internel_label,
                                  batch_first=True).float(),
        'batch_start_frame':
        torch.tensor(batch_start_frame).long(),
        'batch_end_frame':
        torch.tensor(batch_end_frame).long(),
        'batch_extend_pre':
        torch.tensor(batch_extend_pre).long(),
        'batch_extend_suf':
        torch.tensor(batch_extend_suf).long(),
        "batch_keyword_mask":
        nn.utils.rnn.pad_sequence(batch_keyword_mask,
                                  batch_first=True).float(),
        "batch_negative_indices":
        torch.from_numpy(batch_negative_indices).long(),
        "batch_positive_indices":
        torch.from_numpy(batch_positive_indices).long(),
        "batch_start_time":
            torch.tensor(batch_time).float()[:,0],
        "batch_end_time":
            torch.tensor(batch_time).float()[:,1],
        "batch_duration":
            torch.tensor(batch_duration).float(),
        "visual_scale":
            visual_scale
            
    }

    return batch_data, annotations


def average_to_fixed_length(visual_input, num_sample_clips=0):
    if num_sample_clips == 0:
        num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1,
                        1.0) / num_sample_clips * num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(
                torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


from datasets.activitynet import ActivityNet
from datasets.charades import Charades
from datasets.charades_len import Charades_len
from datasets.charades_mom import Charades_mom
from datasets.tacos import TACoS
