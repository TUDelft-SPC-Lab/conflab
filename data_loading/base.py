from abc import ABC, abstractmethod
from typing import List, Tuple

class Extractor(ABC):
    """Extractors know how to extract a single modality segment of the data given a tuple (person_id, ini_time, len, camera)
    The extract method takes in such a tuple and maps it to a section of the data in the corresponding modality
    """
        
    @abstractmethod
    def extract(examples: Tuple[int,  int, int, int]):
        raise NotImplementedError()