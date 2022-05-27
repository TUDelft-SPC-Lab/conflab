import os
from pathlib import Path

import torch
import pandas as pd

from conflab.data_loaders.base import Extractor
from conflab.data_loaders.utils import time_to_seg, vid_seg_to_segment

class ConflabMultimodalDataset(torch.utils.data.Dataset):
    '''
    Takes a list of (person_id, ini_time, len, camera)
    Each element represents one data point.
    It will return a combination of modalities depending on the single-modality Extractors passed to it. 
    '''

    def __init__(self,
                 examples,
                 extractors,
                 transform=None):
        self.examples = examples 
        self.extractors = extractors
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = {}
        for k, ds in self.extractors.items():
            item[k] = ds.extract(self.examples[idx])

        if self.transform:
            item = self.transform(item)
        return item


class ConflabLabelExtractor(Extractor):
    def __init__(self, labels_path):
        self.labels = []
        for p in Path(labels_path).glob('*.csv'):
            self.labels.append(pd.read_csv(p, index_col=False))

    def extract(self, example):
        pid, ini_time, len, _ = example

        seg, offset = time_to_seg(ini_time)
        num_frames = round(len*59.97)

        try:
            labels = self.labels[seg][str(pid)].iloc[offset: offset + num_frames]
        except:
            raise Exception(f'Error for pid={pid}, ini_time={ini_time}, seg={seg}, offset={offset}')

        return (labels.mean() > 0.5)
