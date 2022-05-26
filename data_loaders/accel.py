from pathlib import Path

import numpy as np
import pandas as pd

from conflab import constants
from conflab.data_loaders.base import Extractor

class ConflabAccelExtractor(Extractor):
    
    def __init__(self, accel_path, sr=50):
        self.load_accel(accel_path)
        self.sr = sr

    def load_accel(self, accel_path):
        self.accel = {}
        for accel_path in Path(accel_path).glob('*.csv'):
            pid = int(accel_path.stem)
            self.accel[pid] = pd.read_csv(accel_path)[['time', 'accelX', 'accelY', 'accelZ']]

    def extract(self, example):
        pid, ini_time, len, _ = example

        ini_sample = round(ini_time * self.sr)
        len_in_samples = round(len * self.sr)

        return self.accel[pid].iloc[ini_sample: ini_sample+len_in_samples, 1:].to_numpy() # exclude time