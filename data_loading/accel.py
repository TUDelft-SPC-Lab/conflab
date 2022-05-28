from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from conflab.data_loading.base import Extractor

class ConflabAccelExtractor(Extractor):
    """Extracts processed acceleration data samples for person_id,
    from time ini_time to ini_time+len given a tuple 
    (person_id, ini_time, len, _) 
    """    
    
    def __init__(self, accel_path: str, sr=50):
        self.load_accel(accel_path)
        self.sr = sr

    def load_accel(self, accel_path: str) -> None:
        self.accel = {}
        for accel_path in Path(accel_path).glob('*.csv'):
            pid = int(accel_path.stem)
            self.accel[pid] = pd.read_csv(accel_path)[['time', 'accelX', 'accelY', 'accelZ']]

    def extract(self, example: Tuple[int,  int, int, int]) -> np.array:
        """Extracts processed acceleration data samples for person_id,
        from time ini_time to ini_time+len given a tuple 
        (person_id, ini_time, len, _) 

        Args:
            example Tuple[int, int, int, int]: an example tuple (person_id, ini_time, len, _) to extract

        Returns:
            np.array: data of shape (len * sr, 3)
        """        
        pid, ini_time, len, _ = example

        ini_sample = round(ini_time * self.sr)
        len_in_samples = round(len * self.sr)

        return self.accel[pid].iloc[ini_sample: ini_sample+len_in_samples, 1:].to_numpy() # exclude time