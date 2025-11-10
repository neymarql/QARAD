#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Filename: aras_json
  Author: Long Qian
  Date: 2025-07-04
  Email: neymarql0614@gmail.com
"""

import os, json, torch, random, cv2, torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from losses.clip_pc import clip_similarity

def _norm_path(p): return p.replace('\\', '/')

class ARASJson(Dataset):
    def __init__(self, json_root, cat, clip_w='linear', sample_mode='paired'):
        self.sample_mode = sample_mode
        self.clip_w_mode = clip_w
        self.items = []
        jdir  = os.path.join(json_root, cat)
        for jp in sorted(os.listdir(jdir)):
            with open(os.path.join(jdir, jp)) as f:
                data = json.load(f)
            norm_p = _norm_path(data['image'])
            for k, v in enumerate(data['variants']):
                self.items.append(dict(
                    norm=norm_p,
                    anom=_norm_path(v['anomaly_image_path']),
                    mask=_norm_path(v['anomaly_mask_path']),
                    prompt=v['description']
                ))

        if sample_mode == 'mix':
            random.shuffle(self.items)
        else:
            self.group = {}
            for it in self.items:
                self.group.setdefault(it['norm'], []).append(it)

    def __len__(self):
        return len(self.items) if self.sample_mode=='mix' else len(self.group)

    def _load_img(self, p, to_tensor=True):
        img = Image.open(p).convert('RGB')
        return TF.to_tensor(img) if to_tensor else img

    def __getitem__(self, idx):
        if self.sample_mode == 'mix':
            it = self.items[idx]
            norm = self._load_img(it['norm'])
            anom = self._load_img(it['anom'])
            mask = cv2.imread(it['mask'], 0) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            w = self._clip_weight(anom, it['prompt'])
            return norm, anom, mask, w
        else:
            norm_path, variants = list(self.group.items())[idx]
            norm = self._load_img(norm_path)
            outs = []
            for it in variants:
                anom = self._load_img(it['anom'])
                mask = torch.from_numpy(cv2.imread(it['mask'],0)/255.).float().unsqueeze(0)
                w = self._clip_weight(anom, it['prompt'])
                outs.append((anom, mask, w))
            return norm, outs

    def _clip_weight(self, img_tensor, prompt):
        if self.clip_w_mode == 'none':
            return torch.tensor(1.)
        s = clip_similarity(img_tensor.unsqueeze(0), [prompt]).item()
        if self.clip_w_mode == 'linear':
            return torch.tensor(max(0., (s-0.2)/0.8))
        if self.clip_w_mode == 'softmax':
            return torch.tensor(torch.exp(s/0.07).item())
