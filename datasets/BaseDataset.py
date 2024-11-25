""" Dataset loader for the ActivityNet Captions dataset """
import os
import json
import h5py
from nltk.tag import pos_tag
import torch
from torch import nn
from torch._C import _resolve_type_from_object
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import numpy as np

from . import average_to_fixed_length
from core.config import config
import nltk

if nltk.data.find('taggers/averaged_perceptron_tagger') is not None:
    print("averged_perceptron_tagger has been downloaded.")
else:
    nltk.download("averaged_perceptron_tagger")  # download data for the first time run


class BaseDataset(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
    vocab.itos.extend(["<unk>"])
    vocab.stoi["<unk>"] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    # CC  并列连词  0        NNS 名词复数  1      UH 感叹词 2
    # CD  基数词  3           NNP 专有名词  1     VB 动词原型 4
    # DT  限定符  5          NNP 专有名词复数 1   VBD 动词过去式 4
    # EX  存在词  6          PDT 前置限定词  5    VBG 动名词或现在分词 4
    # FW  外来词   7         POS 所有格结尾  8    VBN 动词过去分词 4
    # IN  介词或从属连词 9    PRP 人称代词   10     VBP 非第三人称单数的现在时 4
    # JJ  形容词     11       PRP$ 所有格代词 17    VBZ 第三人称单数的现在时 4
    # JJR 比较级的形容词 11    RB  副词     12       WDT 以wh开头的限定词 18
    # JJS 最高级的形容词  11  RBR 副词比较级  12    WP 以wh开头的代词 19
    # LS  列表项标记   13      RBS 副词最高级  12    WP$ 以wh开头的所有格代词 20
    # MD  情态动词     4      RP  小品词  14        WRB 以wh开头的副词 21
    # NN  名词单数    1       SYM 符号     15       TO  to 16
    # ',': 22, '.': 23,

    # 三个数据集的词性占比，charades, activitynet, TACoS
    # 1:  38.6, 25.8, 27.8 || 4:  20,   18,  16  || 5:  22,  16,  20  || 9:  11.4, 15,6, 11
    # 10: 2,    3.5,  5.7  || 12: 0.6,  2.8, 1.5 || 14: 2.1, 1.1, 3.3 || 16: 0.9,  2,    1.1
    # 17: 1.3,  1.3,  0.35 || 22: 0.03, 1  , 1,2 || 23: 0,   7.1, 8.3

    pos_tags = {
        "NNS": 0,
        "NNP": 0,
        "NN": 0,
        "VB": 1,
        "VBD": 1,
        "VBN": 1,
        "VBP": 1,
        "VBG": 1,
        "VBZ": 1,
        "MD": 1,
        "IN": 2,
        "JJ": 0,
        "PRP": 0,
        "JJR": 7,
        "JJS": 7,
        "RB": 1,
        "RBR": 1,
        "RBS": 1,
        "LS": 7,
        "RP": 0,
        "SYM": 7,
        "TO": 5,
        "PRP$": 0,
        "WDT": 5,
        "WP": 3,
        "WP$": 3,
        "WRB": 1,
    }

    def __init__(self, split):
        super(BaseDataset, self).__init__()

        self.anno_dirs = {}
        self.anno_dirs["Charades"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.anno_dirs["Charades_len"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.anno_dirs["Charades_mom"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.anno_dirs["ActivityNet"] = "/media/HardDisk_A/users/zzb/dataset/ActivityNet"
        self.anno_dirs["TACoS"] = "/media/HardDisk_A/users/zzb/dataset/TACoS"
        self.feature_dirs = {}
        self.feature_dirs["Charades"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.feature_dirs["Charades_mom"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.feature_dirs["ActivityNet"] = "/media/HardDisk_A/users/zzb/dataset/ActivityNet"
        self.feature_dirs["TACoS"] = "/media/HardDisk_A/users/zzb/dataset/TACoS"
        self.feature_dirs["Charades_len"] = "/media/HardDisk_A/users/zzb/dataset/Charades-STA"
        self.input_type = {}
        self.input_type["Charades"] = "i3d_adam_epoch16_25fps"
        self.input_type["Charades_len"] = "vgg_rgb_features"
        self.input_type["Charades_mom"] = "vgg_rgb_features"
        self.input_type["ActivityNet"] = "cmcs_features"
        self.input_type["TACoS"] = "tall_c3d_features"
        self.split = split
        self.num_pairs = config.DATASET.num_pairs
        self.annotations = self.get_annotation()
        self.num_clips = config.DATASET.num_clips

        self.epsilon = 1e-10

    def __getitem__(self, index):
        video_id = self.annotations[index]["video"]
        gt_s_time, gt_e_time = self.annotations[index]["times"]
        sentence = self.annotations[index]["description"]
        duration = self.annotations[index]["duration"]
        dataset = self.annotations[index]["dataset"]
        # words = sentence.split()
        # 分词
        words = nltk.word_tokenize(sentence)
        if len(words) >= 30:
            words = words[:30]
        words_tags = nltk.pos_tag(words)
        word_idxs, pos_tags,keyword_mask,keyword_idxs = [], [],[],[]
        # print(sentence)
        for keyword, tag in words_tags:
            if tag in self.pos_tags.keys():
                keyword_idxs.append(self.vocab.stoi.get(keyword.lower(), 400000))
                pos_tags.append(self.pos_tags[tag] + 1)
                # print(word, self.pos_tags[tag] + 1)
        keyword_idxs = torch.tensor(keyword_idxs, dtype=torch.long)
        # print(sentence)
        for word in words:
            word_idxs.append(self.vocab.stoi.get(word.lower(), 400000))
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)
        keyword_mask = [1 if v in keyword_idxs else 0 for v in word_idxs]

        (
            visual_input,
            visual_mask,
            extend_pre,
            extend_suf,
            flip_in_time_direction,
        ) = self.get_video_features(video_id, dataset)

        feat_length = visual_input.shape[0]
        ori_feat_length = feat_length - extend_pre - extend_suf
        fps = ori_feat_length / duration
        start_frame = int(fps * gt_s_time)
        end_frame = int(fps * gt_e_time)
        if end_frame >= ori_feat_length:
            end_frame = ori_feat_length - 1
        if start_frame > end_frame:
            start_frame = end_frame

        if flip_in_time_direction:
            start_frame, end_frame = (
                ori_feat_length - 1 - end_frame,
                ori_feat_length - 1 - start_frame,
            )
        assert start_frame <= end_frame
        assert 0 <= start_frame < ori_feat_length
        assert 0 <= end_frame < ori_feat_length
        start_frame += extend_pre
        end_frame += extend_pre

        start_label = np.ones(feat_length, dtype=np.float32) * self.epsilon
        end_label = np.ones(feat_length, dtype=np.float32) * self.epsilon

        y = (1 - (ori_feat_length - 3) * self.epsilon - 0.5) / 2

        if start_frame > 0:
            start_label[start_frame - 1] = y
        if start_frame < feat_length - 1:
            start_label[start_frame + 1] = y
        start_label[start_frame] = 0.5

        if end_frame > 0:
            end_label[end_frame - 1] = y
        if end_frame < feat_length - 1:
            end_label[end_frame + 1] = y
        end_label[end_frame] = 0.5
        # ---- above part is for ACRM use only------

        internel_label = np.zeros(feat_length, dtype=np.float32)
        extend_inner_len = round(
            config.DATASET.EXTEND_INNRE * float(end_frame - start_frame + 1)
        )
        if extend_inner_len > 0:
            st_ = max(0, start_frame - extend_inner_len)
            et_ = min(end_frame + extend_inner_len, feat_length - 1)
            internel_label[st_ : (et_ + 1)] = 1.0
        else:
            internel_label[start_frame:(end_frame+1)] = 1.0
        
        if np.all(internel_label==1.0):
            choice = np.random.choice([0, -1])
            internel_label[choice] = 0.0
        neg_label = 1.0 - internel_label
        if len(internel_label) ==1:
            internel_label[0] = neg_label[0] = 1.0
        positive_indices = np.nonzero(internel_label)[0] # 获取正样本的索引
        if len(positive_indices) == 0:
            print("wrong")
        positive_indices = positive_indices.tolist()
        np.random.shuffle(positive_indices)
        if len(positive_indices) >= self.num_pairs:
            selected_positive_indices = positive_indices[:self.num_pairs]  # 随机选择 num_pairs 个正样本的索引
        else:
            selected_positive_indices = positive_indices
            while len(selected_positive_indices) < self.num_pairs:
                random_positive_indices  = np.random.choice(positive_indices)
                selected_positive_indices = np.hstack((selected_positive_indices, random_positive_indices))

        # 随机选择相应的负样本的索引
        negative_indices = np.nonzero(neg_label)[0]  # 获取正样本的索引
        if len(negative_indices) == 0:
            print("wrong")
        negative_indices = negative_indices.tolist()
        np.random.shuffle(negative_indices)
        if len(negative_indices) >=self.num_pairs:
            selected_negative_indices = negative_indices[:self.num_pairs]  # 随机选择 num_pairs 个正样本的索引
        else:
            selected_negative_indices = negative_indices
            while len(selected_negative_indices) < self.num_pairs:
                random_negative_indices  = np.random.choice(negative_indices)
                selected_negative_indices = np.hstack((selected_negative_indices, random_negative_indices))

        start_frame = np.array(start_frame)
        end_frame = np.array(end_frame)
        extend_pre = np.array(extend_pre)
        extend_suf = np.array(extend_suf)
        item = {
            "visual_input": visual_input,
            "vis_mask": visual_mask,
            "word_vectors": word_vectors,
            # "pos_tags": pos_tags,
            "txt_mask": torch.ones(word_vectors.shape[0], 1),
            "start_label": torch.from_numpy(start_label),
            "end_label": torch.from_numpy(end_label),
            "internel_label": torch.from_numpy(internel_label),
            "start_frame": torch.from_numpy(start_frame),
            "end_frame": torch.from_numpy(end_frame),
            "extend_pre": torch.from_numpy(extend_pre),
            "extend_suf": torch.from_numpy(extend_suf),
            "keyword_mask":torch.tensor(keyword_mask),
            "selected_positive_indices":np.array(selected_positive_indices),
            "selected_negative_indices":np.array(selected_negative_indices),
            "visual_len": len(visual_input)
            
        }
        return item, self.annotations[index]

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid, dataset):
        with h5py.File(os.path.join(self.feature_dirs[dataset], '{}.hdf5'.format(self.input_type[dataset])), 'r') as f:
            if dataset == "ActivityNet" and self.input_type["ActivityNet"]=="sub_activitynet_v1-3.c3d":
                features = torch.from_numpy(f[vid]['c3d_features'][:])
            else:
                features = torch.from_numpy(f[vid][:])
        if dataset != "Charades":
            if features.shape[0] > self.num_clips:
                features = average_to_fixed_length(features, num_sample_clips=self.num_clips)
        frame_rate = 1
        features = features[list(range(0, features.shape[0], frame_rate))]
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
            
        # flip the input in time direction
        flip_in_time_direction = False  # use for start/end label flip
        if (
            self.split == "train"
            and config.DATASET.FLIP_TIME
            and np.random.random() < 0.5
        ):
            features = torch.flip(features, dims=[0])
            flip_in_time_direction = True

        length = features.shape[0]
        prefix, suffix = 0, 0
        # add a mean_feature in front of and end of the video to double the time length
        if (
            self.split == "train"
            and config.DATASET.EXTEND_TIME
            and np.random.random() < 0.7
        ):
            # mean_feature = torch.mean(features, dim=0)
            # extend_feature = mean_feature.unsqueeze(0).repeat((prefix, 1))  # add mean feature
            # extend_feature = torch.zeros((prefix, features.shape[1]))      # add zeros feature
            #  --->add another_features start<---
            index = np.random.randint(len(self.annotations))  # another_video
            video_id = self.annotations[index]["video"]
            while video_id == vid:
                index = np.random.randint(len(self.annotations))  # another_video
                video_id = self.annotations[index]["video"]
            featurePath = os.path.join(self.feature_dirs[dataset], video_id + ".npy")
            another_features = np.load(featurePath)
            another_features = np.squeeze(another_features)
            another_features = torch.from_numpy(another_features).float()
            # 特征长度最长为1500lenth
            if another_features.shape[0] > 1500:
                another_features = average_to_fixed_length(
                    another_features, num_sample_clips=1500
                )
            another_features = another_features[
                list(range(0, another_features.shape[0], frame_rate))
            ]
            prefix = round(np.random.random() * another_features.shape[0])
            extend_feature = another_features[:prefix]
            assert extend_feature.shape[0] == prefix
            #  --->add another_features end<---
            features = torch.cat([extend_feature, features], dim=0)
        vis_mask = torch.ones((features.shape[0], 1))

        return features, vis_mask, prefix, suffix, flip_in_time_direction

    def get_annotation(self, dataset):
        raise NotImplementedError
