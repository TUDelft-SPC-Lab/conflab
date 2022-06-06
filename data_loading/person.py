from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple, Any
from pydantic import Extra

from sklearn.metrics import roc_auc_score

import torch
import numpy as np
from scipy.special import expit

from conflab.data_loading.base import Extractor


class ConflabSubset(torch.utils.data.Subset):
    """Subclasses Subset to support auc and accuracy methods in the dataset.
    These methods simply forward their arguments to the parent dataset.
    """    
    def random_split(self, size=0.1) -> Tuple[ConflabSubset, ConflabSubset]:
        """Splits the dataset into one of size round(len(self*size)) and one containing the rest of the examples.

        Args:
            size (int, optional): Fractional size of the new dataset. Defaults to 0.1.

        Returns:
            Tuple[ConflabSubset, ConflabSubset]: The new datasets
        """             
        num_samples = int(size*len(self))
        shuffled_indices = list(self.indices)
        random.shuffle(self.indices)
        return (
            ConflabSubset(self.dataset, shuffled_indices[:num_samples]),
            ConflabSubset(self.dataset, shuffled_indices[num_samples:])
        )  

    def auc(self, idxs, proba) -> float:
        return self.dataset.auc(idxs, proba)

    def accuracy(self, idxs, proba) -> float:
        return self.dataset.accuracy(idxs, proba)

class ConflabDataset(torch.utils.data.Dataset):
    '''
    This is the main dataset object. It implements a multimodal dataset
    by joining the output of single-modality Extractors passed to it. 
    '''

    def __init__(self,
                 examples: List[Tuple[int,  int, int, int]],
                 extractors: Dict[str, Extractor],
                 transform=None):
        """Takes a list of (person_id, ini_time, len, camera)
        Each element represents one data point.

        Args:
            examples List[Tuple[int,  int, int, int]]:  list of (person_id, ini_time, len, camera), each representing one example in the dataset
            extractors Dict[str, Extractor]: a map of {'modality_name': Extractor} with the Extractors to use  
            transform Function: A transform function (a la pytorch)"""                 
        self.examples = examples 
        self.extractors = extractors
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = {}
        for k, ds in self.extractors.items():
            item[k] = ds.extract(self.examples[idx])
        item['index'] = idx

        if self.transform:
            item = self.transform(item)
        
        return item

    def get_groups(self) -> List[int]:
        """Returns the first dimension (person id) of the examples in the dataset.
        Can be used to split the dataset per person.

        Returns:
            List[int]: List person IDs for every example in the dataset, in order
        """        
        return [e[0] for e in self.examples]

    def get_all_labels(self) -> np.array:
        if 'label' not in self.extractors:
            raise Exception('called_get_all_labels() but no label extractor was provided to ConflabDataset object')
        labels = [self.extractors['label'].extract(ex) for ex in self.examples]
        return np.array(labels)

    def auc(self, idxs: List[int], proba: np.array) -> float:
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        return roc_auc_score(labels, proba)

    def accuracy(self, idxs: List[int], proba: np.array) -> float:
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        pred = np.argmax(proba, axis=1)

        correct = (pred == labels).sum().item()
        return correct / len(labels)

