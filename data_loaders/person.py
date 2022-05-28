import os
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import numpy as np
import pandas as pd
from scipy.special import expit

from conflab.data_loaders.base import Extractor
from conflab.data_loaders.utils import time_to_seg, vid_seg_to_segment


class ConflabSubset(torch.utils.data.Subset):
    def random_split(self, size=0.1):
        num_samples = int(size*len(self))
        shuffled_indices = list(self.indices)
        random.shuffle(self.indices)
        return (
            ConflabSubset(self.dataset, shuffled_indices[:num_samples]),
            ConflabSubset(self.dataset, shuffled_indices[num_samples:])
        )  

    def auc(self, idxs, proba):
        return self.dataset.auc(idxs, proba)

    def accuracy(self, idxs, proba):
        return self.dataset.accuracy(idxs, proba)

class ConflabDataset(torch.utils.data.Dataset):
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
        item['index'] = idx

        if self.transform:
            item = self.transform(item)
        
        return item

    def get_groups(self):
        return [e[0] for e in self.examples]

    def get_all_labels(self) -> np.array:
        if 'label' not in self.extractors:
            raise Exception('called_get_all_labels() but no label extractor was provided to ConflabMultimodalDataset object')
        labels = [self.extractors['label'].extract(ex) for ex in self.examples]
        return np.array(labels)

    def auc(self, idxs, proba):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        return roc_auc_score(labels, proba)

    def accuracy(self, idxs, proba: np.ndarray):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        pred = np.argmax(proba, axis=1)

        correct = (pred == labels).sum().item()
        return correct / len(labels)

class ConflabLabelExtractor(Extractor):
    def __init__(self, labels_path):
        self.labels = []
        for p in Path(labels_path).glob('*.csv'):
            self.labels.append(pd.read_csv(p, index_col=False))

    def extract(self, example):
        pid, ini_time, len, _ = example

        seg, offset = time_to_seg(ini_time)
        num_frames = round(len*59.94)

        try:
            labels = self.labels[seg][str(pid)].iloc[offset: offset + num_frames]
        except:
            raise Exception(f'Error for pid={pid}, ini_time={ini_time}, seg={seg}, offset={offset}')

        return int(labels.mean() > 0.5)
