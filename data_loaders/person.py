from dataclasses import dataclass
import torch

from conflab.data_loaders.base import Extractor

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

    def __getitem__(self, idx):
        item = {}
        for k, ds in self.extractors.items():
            item[k] = ds.extract(self.examples[idx])

        if self.transform:
            item = self.transform(item)
        return item
