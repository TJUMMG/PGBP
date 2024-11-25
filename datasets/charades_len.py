""" Dataset loader for the Charades-STA dataset """
import os
import csv

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import json
from . import average_to_fixed_length
from .BaseDataset import BaseDataset
from core.config import config


class Charades_len(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:12 max:390 mean: 62, std:18
        # max sentence length：train->10, test->10
        super(Charades_len, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):

        anno_file = open(
            os.path.join(self.anno_dirs['Charades'],
                         "{}_len_80.jsonl".format(self.split)), 'r')
        annotations = []
        # max_sentence_length = 0
        for line in anno_file:
            line_obj = json.loads(line.strip())
            sent = line_obj["query"]
            vid = line_obj["vid"]
            times = line_obj["relevant_windows"][0]
            duration = line_obj["duration"]
            annotations.append({
                    'video': vid,
                    'times': times,
                    'description': sent,
                    'duration': duration,
                    'dataset': 'Charades_len'
                })
        anno_file.close()
        # print("charade max sentence length: ", max_sentence_length)
        return annotations