import os
import copy
import json
import numpy as np
import pandas as pd
from einops import repeat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .categories import AVVP_CATEGORIES, UNAV_CATEGORIES


class LLP_dataset(Dataset):
    def __init__(self, mode, weak_data_csv, gt_a_data_csv, gt_v_data_csv,
                 audio_dir, res152_dir, r2plus1d_18_dir,
                 clap_feat_dir=None, clip_feat_dir=None,
                 v_pseudo_data_dir=None, a_pseudo_data_dir=None,
                 v_logit_dir=None, a_logit_dir=None):
        
        self.mode = mode
        self.video_len = 10

        self.weak_data_df = pd.read_csv(weak_data_csv, header=0, sep='\t')
        self.gt_a_df = pd.read_csv(gt_a_data_csv, header=0, sep='\t')
        self.gt_v_df = pd.read_csv(gt_v_data_csv, header=0, sep='\t')

        self.audio_dir = audio_dir
        self.video_dir = res152_dir
        self.st_dir = r2plus1d_18_dir
        self.clap_feat_dir = clap_feat_dir
        self.clip_feat_dir = clip_feat_dir
        self.a_pseudo_data_dir = a_pseudo_data_dir
        self.v_pseudo_data_dir = v_pseudo_data_dir
        self.a_logit_dir = a_logit_dir
        self.v_logit_dir = v_logit_dir

        self.categories = AVVP_CATEGORIES
        self.id_to_idx = {id: index for index, id in enumerate(AVVP_CATEGORIES)}


        if self.a_pseudo_data_dir is None or len(os.listdir(self.a_pseudo_data_dir)) == 0:
            print("No valid audio class frequencies.")
            self.a_class_freq = np.zeros(len(self.categories))
        else:
            self.a_class_freq = self.get_class_freq(self.a_pseudo_data_dir)

        if self.v_pseudo_data_dir is None or len(os.listdir(self.v_pseudo_data_dir)) == 0:
            print("No valid visual class frequencies.")
            self.v_class_freq = np.zeros(len(self.categories))
        else:
            self.v_class_freq = self.get_class_freq(self.v_pseudo_data_dir)


    def __len__(self):
        return len(self.weak_data_df["filename"])

    def __getitem__(self, idx):
        row = self.weak_data_df.loc[idx, :]
        video_name = row[0][:11]

        # load standard features
        audio = self.load_numpy_file(os.path.join(self.audio_dir, video_name + '.npy'))
        video_s = self.load_numpy_file(os.path.join(self.video_dir, video_name + '.npy'))
        video_st = self.load_numpy_file(os.path.join(self.st_dir, video_name + '.npy'))

        # load clip, clap features
        if (self.clap_feat_dir is not None) and (self.clip_feat_dir is not None):
            clap_feat = self.load_numpy_file(os.path.join(self.clap_feat_dir, video_name + '.npy'))
            clip_feat = self.load_numpy_file(os.path.join(self.clip_feat_dir, video_name + '.npy'))
        else:
            clap_feat = np.zeros((self.video_len, 512))
            clip_feat = np.zeros((self.video_len, 768))

        # construct labels
        weak_label = self.get_weak_label(row[-1].split(','))

        gt_a_label = self.get_gt_label(row[0], self.gt_a_df)
        gt_v_label = self.get_gt_label(row[0], self.gt_v_df)

        if (self.a_pseudo_data_dir is not None) and (self.v_pseudo_data_dir is not None):
            a_pseudo_label = self.load_numpy_file(os.path.join(self.a_pseudo_data_dir, video_name + '.npy'))
            v_pseudo_label = self.load_numpy_file(os.path.join(self.v_pseudo_data_dir, video_name + '.npy'))
        else:
            a_pseudo_label = np.zeros((self.video_len, len(self.categories)))
            v_pseudo_label = np.zeros((self.video_len, len(self.categories)))

        if (self.a_logit_dir is not None) and (self.v_logit_dir is not None):
            a_logit = self.load_numpy_file(os.path.join(self.a_logit_dir, video_name + '.npy'))
            v_logit = self.load_numpy_file(os.path.join(self.v_logit_dir, video_name + '.npy'))
        else:
            a_logit = np.zeros((self.video_len, len(self.categories)))
            v_logit = np.zeros((self.video_len, len(self.categories)))

        # construct masks
        valid_mask = np.ones(self.video_len, dtype=np.float32)
        attn_mask = np.ones((self.video_len, self.video_len), dtype=bool)

        sample = {
            'audio': audio,
            'video_s': video_s,
            'video_st': video_st,
            'weak_label': weak_label,
            'video_name': video_name,
            'duration': self.video_len,
            'attn_mask': attn_mask,
            'valid_mask': valid_mask,
            'gt_a_label': gt_a_label,
            'gt_v_label': gt_v_label,
            'a_pseudo_label': a_pseudo_label,
            'v_pseudo_label': v_pseudo_label,
            'a_logit': a_logit,
            'v_logit': v_logit,
            'clap_feat': clap_feat,
            'clip_feat': clip_feat
        }

        return sample
    
    def load_numpy_file(self, np_path):
        with open(np_path, 'rb') as f:
            data = np.load(f)
        return data
    
    def get_weak_label(self, ids):
        weak_label = np.zeros(len(self.categories))
        for id in ids:
            index = self.id_to_idx[id]
            weak_label[index] = 1
        return weak_label
    
    def get_gt_label(self, video_name, gt_df):

        gt_label = np.zeros((len(self.categories), self.video_len))

        df_video = gt_df.loc[gt_df['filename'] == video_name]
        filenames = df_video["filename"]
        events = df_video["event_labels"]
        onsets = df_video["onset"]
        offsets = df_video["offset"]
        num = len(filenames)
        if num > 0:
            for i in range(num):
                x1 = int(onsets[df_video.index[i]])
                x2 = int(offsets[df_video.index[i]])
                event = events[df_video.index[i]]
                idx = self.id_to_idx[event]
                gt_label[idx, x1:x2] = 1

        return gt_label.T
    
    def get_class_freq(self, seg_label_dir):

        total_seg_counts = np.zeros(len(self.categories))
        pos_seg_counts = np.zeros(len(self.categories))

        for i in range(len(self.weak_data_df.index)):
            video_name = self.weak_data_df.loc[i, :][0][:11]
            seg_label_path = os.path.join(seg_label_dir, video_name+'.npy')
            seg_label = np.load(seg_label_path) # (T, C)

            class_seg_counts = seg_label.sum(axis=0)

            pos_seg_counts += class_seg_counts
            total_seg_counts += np.ones(len(self.categories)) * len(seg_label)

        class_freq = pos_seg_counts / total_seg_counts

        return class_freq


class UnAV_dataset(Dataset):
    def __init__(self, mode, data_json,
                 audio_dir, vis_rgb_dir, vis_flow_dir,
                 clap_feat_dir, clip_feat_dir):
        
        self.mode = mode if mode != 'val' else 'validation'
        self.max_quantized_duration = 0     # max quantized duration of a video

        self.audio_dir = audio_dir
        self.vis_rgb_dir = vis_rgb_dir
        self.vis_flow_dir = vis_flow_dir
        self.clap_feat_dir = clap_feat_dir
        self.clip_feat_dir = clip_feat_dir
        
        self.json_file = data_json
        self.data_list, _ = self._load_json_db(data_json)

        self.categories = UNAV_CATEGORIES
        self.num_classes = len(self.categories)
        self.id_to_idx = {id: index for index, id in enumerate(UNAV_CATEGORIES)}

        attn_mask_dict, mask_dict, gt_label_dict = self.get_all_labels()
        self.label_manager = LabelManager(attn_mask_dict, mask_dict, gt_label_dict)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        video_item = self.data_list[idx]
        quant_duration = video_item['quantized duration']

        # load features
        filename_audio = os.path.join(self.audio_dir, video_item['id'] + '_vggish.npy')
        feats_audio = self.load_numpy_file(filename_audio).astype(np.float32)
        filename_rgb = os.path.join(self.vis_rgb_dir, video_item['id'] + '_rgb.npy')
        feats_rgb = self.load_numpy_file(filename_rgb).astype(np.float32)
        filename_flow = os.path.join(self.vis_flow_dir, video_item['id'] + '_flow.npy')                                                       
        feats_flow = self.load_numpy_file(filename_flow).astype(np.float32)
        filename_clap = os.path.join(self.clap_feat_dir, video_item['id'] + '.npy')
        feats_clap = self.load_numpy_file(filename_clap).astype(np.float32)
        filename_clip = os.path.join(self.clip_feat_dir, video_item['id'] + '.npy')
        feats_clip = self.load_numpy_file(filename_clip).astype(np.float32)

        feats_audio = torch.from_numpy(feats_audio)
        feats_rgb = torch.from_numpy(feats_rgb)
        feats_flow = torch.from_numpy(feats_flow)
        feats_clap = torch.from_numpy(feats_clap)
        feats_clip = torch.from_numpy(feats_clip)

        # avoid audio and visual features have different lengths
        # min_len = min(feats_rgb.shape[0], feats_audio.shape[0])
        # feats_rgb = feats_rgb[:min_len]
        # feats_flow = feats_flow[:min_len]
        # feats_audio = feats_audio[:min_len]

        # pool feat along time dim to have one feat for each second
        feats_audio = self.avgpool_feature(feats_audio, quant_duration)
        feats_rgb = self.avgpool_feature(feats_rgb, quant_duration)
        feats_flow = self.avgpool_feature(feats_flow, quant_duration)
        feats_clap = self.avgpool_feature(feats_clap, quant_duration)
        feats_clip = self.avgpool_feature(feats_clip, quant_duration)

        # pad feat sequence to have the same len
        feats_audio = self.pad_feature(feats_audio, target_len=self.max_quantized_duration)
        feats_rgb = self.pad_feature(feats_rgb, target_len=self.max_quantized_duration)
        feats_flow = self.pad_feature(feats_flow, target_len=self.max_quantized_duration)
        feats_clap = self.pad_feature(feats_clap, target_len=self.max_quantized_duration)
        feats_clip = self.pad_feature(feats_clip, target_len=self.max_quantized_duration)

        # get labels
        attn_mask, valid_mask, gt_label = self.label_manager.get_labels(video_item['id'])

        # construct weak label
        weak_label = torch.where(gt_label.sum(dim=0) > 0, 1, 0).float()

        sample = {
            'audio': feats_audio,
            'video_s': feats_rgb,
            'video_st': feats_flow,
            'weak_label': weak_label,
            'video_name': video_item['id'],
            'duration': quant_duration,
            'attn_mask': attn_mask,
            'valid_mask': valid_mask,
            'gt_label': gt_label,
            'clap_feat': feats_clap,
            'clip_feat': feats_clip,
        }

        return sample
    
    def load_numpy_file(self, np_path):
        with open(np_path, 'rb') as f:
            data = np.load(f)
        return data
    
    def _load_json_db(self, json_file):

        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # get label_dict
        label_dict = {}
        for key, value in json_db.items():
            for act in value['annotations']:
                label_dict[act['label']] = act['label_id']

        # get max quantized video duration
        for video_id, value in json_db.items():
            if value['subset'].lower() not in self.mode:
                continue

        dict_db = tuple()
        for video_id, value in json_db.items():
            if value['subset'].lower() not in self.mode:
                continue

            # get quantized video duration if available
            if int(value['duration']) == value['duration']:
                duration = int(value['duration'])
            else:
                duration = int(value['duration']) + 1
            self.max_quantized_duration = max(self.max_quantized_duration, duration)

            # get quantized annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                annot = value['annotations']
            else:
                annot = None

            dict_db += ({'id': video_id,
                         'quantized duration' : duration,
                         'annot' : annot
                        }, )

        return dict_db, label_dict

    def get_all_labels(self):
        attn_mask_dict = {}
        mask_dict = {}
        gt_label_dict = {}
        
        for i in range(len(self.data_list)):
            video_item = self.data_list[i]

            valid_mask = torch.zeros(self.max_quantized_duration, dtype=torch.float32)
            valid_mask[:video_item['quantized duration']] = 1

            attn_mask = torch.zeros(self.max_quantized_duration, self.max_quantized_duration, dtype=torch.bool)
            attn_mask[:video_item['quantized duration'], :video_item['quantized duration']] = True

            gt_label = self.get_gt_label(video_item['annot'])

            attn_mask_dict[video_item['id']] = attn_mask
            mask_dict[video_item['id']] = valid_mask
            gt_label_dict[video_item['id']] = gt_label
            
        return attn_mask_dict, mask_dict, gt_label_dict

    def get_gt_label(self, gt_annots):

        GT = np.zeros((len(self.categories), self.max_quantized_duration))

        for idx, act in enumerate(gt_annots):
            start, end = act['segment']
            
            # find the shortest integer span that includes the original time span
            new_start = int(start)
            new_end = int(end) if (int(end) == end) else int(end) + 1
            assert new_start >= 0, f'start time = {new_start} is less than 0'
            assert new_end <= self.max_quantized_duration, f'end time = {new_end} is larger than {self.max_quantized_duration} (max duration)'

            label_id = act['label_id']
            GT[label_id, new_start:new_end] = 1

        return torch.from_numpy(GT).long().permute(1, 0)

    
    def avgpool_feature(self, feat, target_len):
        # feat size: (L, D)
        L, D = feat.size()
        feat = feat.permute(1, 0)
        feat = F.adaptive_avg_pool1d(feat, target_len)
        feat = feat.permute(1, 0).contiguous()
        return feat
    
    def pad_feature(self, feat, target_len):
        # feat size: (L, D)
        L, D = feat.size()
        pad_feat = torch.zeros(target_len - L, D, dtype=torch.float32)
        feat = torch.cat((feat, pad_feat), dim=0)
        return feat
    

class LabelManager:
    def __init__(self, attn_mask_dict, mask_dict, gt_label_dict):
        self.attn_mask_dict = self.__set_tensor_data(attn_mask_dict)
        self.mask_dict = self.__set_tensor_data(mask_dict)
        self.gt_label_dict = self.__set_tensor_data(gt_label_dict)

    def __set_tensor_data(self, dict):
        for key, val in dict.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            dict[key] = val
        return dict
    
    def get_labels(self, video_id):
        assert video_id in self.gt_label_dict, f"video_id: {video_id} not in LabelManager label_dict"

        attn_mask = self.attn_mask_dict[video_id]
        valid_mask = self.mask_dict[video_id]
        gt_label = self.gt_label_dict[video_id]

        return attn_mask, valid_mask, gt_label


def build_dataset(args, mode):

    if args.dataset == 'LLP':
        WEAK_DATA_CSV_DICT = {'train': args.label_train, 'val': args.label_val, 'test': args.label_test}
        dataset = LLP_dataset(
            mode=mode,
            weak_data_csv=WEAK_DATA_CSV_DICT[mode],
            gt_a_data_csv=args.gt_audio_csv,
            gt_v_data_csv=args.gt_visual_csv,
            audio_dir=args.audio_dir,
            res152_dir=args.video_dir,
            r2plus1d_18_dir=args.st_dir,
            clap_feat_dir=args.clap_feat_dir,
            clip_feat_dir=args.clip_feat_dir,
            a_pseudo_data_dir=args.a_pseudo_data_dir,
            v_pseudo_data_dir=args.v_pseudo_data_dir,
            a_logit_dir=args.a_logit_dir,
            v_logit_dir=args.v_logit_dir,
        )

    elif args.dataset == 'UnAV':
        dataset = UnAV_dataset(
            mode=mode,
            data_json=args.label_all,
            audio_dir=args.audio_dir,
            vis_rgb_dir=args.video_dir,
            vis_flow_dir=args.st_dir,
            clap_feat_dir=args.clap_feat_dir,
            clip_feat_dir=args.clip_feat_dir
        )
    else:
        raise ValueError(f'The {args.dataset} dataset is not supported.')

    return dataset


def build_dataloader(args, dataset, mode):
    BATCH_SIZE = args.batch_size if mode == 'train' else 1
    SHUFFLE = True if mode == 'train' else False

    dataloader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
    return dataloader
