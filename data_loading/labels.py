from pathlib import Path
import pandas as pd

from conflab.data_loading.base import Extractor

from conflab.data_loading.utils import time_to_seg, vid_seg_to_segment

class ConflabLabelExtractor(Extractor):
    """Extracts processed action labels for person_id,
    from time ini_time to ini_time+len given a tuple 
    (person_id, ini_time, len, _) 
    """  
    def __init__(self, labels_path):
        self.labels = []
        for p in Path(labels_path).glob('*.csv'):
            self.labels.append(pd.read_csv(p, index_col=False))

    def extract(self, example):
        """Extracts processed action labels for person_id,
        from time ini_time to ini_time+len given a tuple 
        (person_id, ini_time, len, _) 

        Args:
            example Tuple[int, int, int, int]: an example tuple (person_id, ini_time, len, _) to extract

        Returns:
            int: label for the window
        """   
        pid, ini_time, len, _ = example

        seg, offset = time_to_seg(ini_time)
        num_frames = round(len*59.94)

        try:
            labels = self.labels[seg][str(pid)].iloc[offset: offset + num_frames]
        except:
            raise Exception(f'Error for pid={pid}, ini_time={ini_time}, seg={seg}, offset={offset}')

        return int(labels.mean() > 0.5)
