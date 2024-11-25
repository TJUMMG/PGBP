""" Dataset loader for the TACoS dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from .BaseDataset import BaseDataset
from core.config import config


class TACoS(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:90 max:2578 mean: 528, std:436
        # max sentence length：train-->46, test-->50
        super(TACoS, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):
        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(
                os.path.join(self.anno_dirs['TACoS'],
                             '{}.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        # max_sentence_length = 0
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames'] / video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'],
                                           video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append({
                        'video':
                        vid,
                        # vid[:-4],
                        'duration':
                        duration,
                        'times': [
                            max(timestamp[0] / video_anno['fps'], 0),
                            min(timestamp[1] / video_anno['fps'], duration)
                        ],
                        'description':
                        sentence,
                        'dataset':
                        'TACoS'
                    })
        # print("tacos max sentence length: ", max_sentence_length)
        return anno_pairs