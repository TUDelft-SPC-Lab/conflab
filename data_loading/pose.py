import os
from pathlib import Path
import json
import pickle
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from conflab.data_loading.base import Extractor
from conflab.data_loading.utils import seg_to_offset, time_to_seg, vid_seg_to_segment

class ConflabPoseExtractor(Extractor):
    """Extracts processed pose data for person_id for a window
    from time ini_time to ini_time+len given a tuple 
    (person_id, ini_time, len, _) 
    """ 
    def __init__(self,
                 data_path: str,
                 return_occlusion=False):
        """Takes a path to the processed Conflab pose files (folder)

        Args:
            data_path (str): Path to processed segment files
            return_occlusion (bool, optional): Set to true to
                return the 17 occlusion labels in addition to the keypoint
                x, y location. Defaults to False.
        """                 
        self.data_path = data_path
        self.accel_ds = None
        self.return_occlusion = return_occlusion

    def load_from_pickle(self, data_path: str) -> None:
        self.tracks = pickle.load(open(data_path, 'rb'))

    def store_tracks(self, data_path: str) -> None:
        pickle.dump(self.tracks, open(data_path, 'wb'))

    def load_data(self):
        # load file list
        self.tracks = [{} for _ in range(8)]
                           # [seg: 
                           #    {[cam]: 
                           #          {tracks: {
                           #              pid: [np.array shape [track_len, 17*2 + 17 + 1]]
                           #          }
                           #     }
                           # ]

        for seg_file in tqdm(Path(self.data_path).glob('*.json')):
            parts = os.path.basename(seg_file).split('_')
            cam = int(parts[0][-1:])
            seg = vid_seg_to_segment(
                int(parts[1][-1:]),
                int(parts[2][-1:])
            )

            tracks = {}
            with open(seg_file) as f:
                coco_json = json.load(f)
                for frame_skeletons in coco_json['annotations']['skeletons']:
                    for fs in frame_skeletons.values():
                        pid = fs['id']

                        if pid not in tracks:
                            tracks[pid] = []
                        
                        tracks[pid].append({
                            'frame': fs['image_id'],
                            'kp': fs['keypoints'],
                            'occl': fs['occluded']
                        })
            # join list of tracks into a single track per pid
            for pid, track in tracks.items():
                new_track = [[e['frame'], *e['kp'], *e['occl']] for e in track]
                tracks[pid] = np.array(new_track, dtype=np.float64)

            self.tracks[seg][cam] = {
                'tracks': tracks
            }

    def make_examples(self, window_len=3, stride=1.5) -> List[Tuple[int,  int, int, int]]:
        """Splits the pose tracks into examples as tuples (person_id, ini_time, len, camera)
        with times in seconds.
        ini_time is relative to the dataset global start time 

        Args:
            window_len (int, optional): example length in seconds. Defaults to 3.
            stride (float, optional): stride in seconds. Defaults to 1.5.

        Returns:
            List[Tuple[int,  int, int, int]]: tuple of (person_id, ini_time, len, camera)
        """        
            #   seg, pid
        skip = [(1, 11), (1, 42)]

        examples = []
        for i, segment in enumerate(tqdm(self.tracks)):
            segment_offset = seg_to_offset(i)
            for cam, cam_data in segment.items():
                for pid, track in cam_data['tracks'].items():
                    assert track[-1,0] == len(track)-1
                    if (i, pid) in skip: continue

                    ini_times = np.arange(0, len(track)/59.94 - window_len, stride)
                    track_examples = [(pid, segment_offset + i, window_len, cam) for i in ini_times]
                    examples += track_examples

        return examples
 
    def extract(self, example):
        """Extracts processed pose data for person_id,
        from time ini_time to ini_time+len given a tuple 
        (person_id, ini_time, len, camera) 

        Args:
            example Tuple[int, int, int, int]: an example tuple 
                (person_id, ini_time, len, camera) to extract

        Returns:
            np.array: data of shape (round(len*59.94), 34)
        """           
        pid, ini_time, len, cam = example

        seg, offset = time_to_seg(ini_time)
        num_frames = round(len*59.94)
        track = self.tracks[seg][cam]['tracks'][pid]

        last_column = 52 if self.return_occlusion else 52-17
        return torch.from_numpy(track[offset: offset+num_frames, 1: last_column])


class ConflabToKinetics(object):
    """Transforms the pose stream from Conflab's 17 keypoints to Kinetics 18 keypoints."""

    def __call__(self, sample):
        pose = sample['pose']

        # fill data_numpy
        # output has shape [channel, seq_len, num_keypoints, num_people]
        new_sample = torch.zeros((3, pose.shape[0], 18, 2))
        new_sample[0, :, 0:17, 0] = pose[:,0:34:2]
        new_sample[1, :, 0:17, 0] = pose[:,1:34:2]
        new_sample[2, :, :, 0] = 0.5

        # map from conflab to kinetics joints
        # Kinetics (target) joint index:
        # {0,  "Nose"}
        # {1,  "Neck"},
        # {2,  "RShoulder"},
        # {3,  "RElbow"},
        # {4,  "RWrist"},
        # {5,  "LShoulder"},
        # {6,  "LElbow"},
        # {7,  "LWrist"},
        # {8,  "RHip"},
        # {9,  "RKnee"},
        # {10, "RAnkle"},
        # {11, "LHip"},
        # {12, "LKnee"},
        # {13, "LAnkle"},
        # {14, "REye"},
        # {15, "LEye"},
        # {16, "REar"},
        # {17, "LEar"},
        new_sample[:,:,:,0] = new_sample[:,:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0],0]
        new_sample[2, :, 14:18, 0] = 0 # 14-18 are not in conflab

        # centralization
        new_sample[0:2] = new_sample[0:2] - 0.5
        new_sample[1:2] = -new_sample[1:2]

        new_sample = torch.nan_to_num(new_sample) # new: convert NaN to zero
        new_sample[0][new_sample[2] == 0] = 0
        new_sample[1][new_sample[2] == 0] = 0

        sample['pose'] = new_sample
        return sample
