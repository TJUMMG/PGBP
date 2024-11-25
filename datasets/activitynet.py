""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from .BaseDataset import BaseDataset
from . import average_to_fixed_length
from core.config import config


class ActivityNet(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:2 medium: max:1415  mean: 204, std:120
        # max sentence length：train-->73, test-->73
        super(ActivityNet, self).__init__(split)

    def __len__(self):
        return len(self.annotations)


    def get_annotation(self):

        with open(
                os.path.join(self.anno_dirs['ActivityNet'],
                             '{}_data.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for video_anno in annotations:

            vid = video_anno[0]
            duration = video_anno[1]
            timestamp = video_anno[2]
            sentence = video_anno[3]

            if timestamp[0] < timestamp[1]:
                anno_pairs.append({
                        'video':
                        vid,
                        'duration':
                        duration,
                        'times':
                        [max(timestamp[0], 0),
                         min(timestamp[1], duration)],
                        'description':
                        sentence,
                        'dataset':
                        'ActivityNet'
                    })
        # print("activitynet max sentence length: ", max_sentence_length)
        return anno_pairs