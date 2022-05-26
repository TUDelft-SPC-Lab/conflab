import os
import glob
from pathlib import Path
import json
import pickle
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from conflab.data_loaders.base import Extractor


class ConflabPoseExtractor(Extractor):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
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
    Arguments:
        data_path: the path to folder with skeletons in COCO JSON format
        label_path: the path to label
        window_size: The length of the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 return_occlusion=False):
        self.data_path = data_path
        self.accel_ds = None
        self.return_occlusion = return_occlusion

        self.load_data()

    def _time_to_seg(self, time):
        '''
        Converts a time (s) in the annotated section of the dataset 
        into a segment number and offset that can be used to localize
        a window within the loaded tracks
        '''
        seg = int(time // 120) # 2 min segments
        offset = int((time % 120) * 59.97)
        return seg, offset

    def _vid_seg_to_segment(self, vid, seg):
        return {
            (2,8): 0,
            (2,9): 1,
            (3,1): 2,
            (3,2): 3,
            (3,3): 4,
            (3,4): 5,
            (3,5): 6,
            (3,6): 7
        }[(vid,seg)]

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
            seg = self._vid_seg_to_segment(
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

    def make_examples(self, window_len=3, stride=1.5):
        '''
        Splits data into examples.
        '''

        examples = []
        for i, segment in enumerate(tqdm(self.tracks)):
            segment_offset = i * 120 # 2 minutes per segment
            for cam, cam_data in segment.items():
                for pid, track in cam_data['tracks'].items():
                    assert track[-1,0] == len(track)-1

                    ini_times = np.arange(0, len(track)/59.94, stride)
                    track_examples = [(pid, segment_offset + i, window_len, cam) for i in ini_times[:-1]]
                    examples += track_examples

        return examples
 
    def extract(self, example):

        pid, ini_time, len, cam = example

        seg, offset = self._time_to_seg(ini_time)
        num_frames = round(len*59.97)
        track = self.tracks[seg][cam]['tracks'][pid]

        last_column = 52 if self.return_occlusion else 52-17
        return track[offset: offset+num_frames, 1: last_column]


class ConflabToKinetics(object):
    """Transforms the pose stream from Conflab's 17 keypoints to Kinetics 18 keypoints."""

    def __call__(self, sample):
        pose = sample['pose']

        # fill data_numpy
        data_numpy = np.zeros((3, pose.shape[0], 18))
        data_numpy[0, :, 0:17] = pose[:,0:34:2]
        data_numpy[1, :, 0:17] = pose[:,1:34:2]
        data_numpy[2, :, :] = 0.5

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
        data_numpy[:,:,:] = data_numpy[:,:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0]]
        data_numpy[2, :, 14:18] = 0 # 14-18 are not in conflab

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]

        data_numpy = np.nan_to_num(data_numpy) # new: convert NaN to zero
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        sample['pose'] = data_numpy
        return sample



def gen_accel_data_from_feeder(feeder, out_folder):
    val_size = 0.2

    subjects = list(set([ex['pid'] for ex in feeder.examples]))
    train_subjects, val_subjects = train_test_split(subjects, test_size=val_size, random_state=22, shuffle=True)

    idxs = {}
    idxs['train'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in train_subjects]
    idxs['val'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in val_subjects]
    print(f'train sz: {len(idxs["train"])}, val sz: {len(idxs["val"])}')
    accel = feeder.make_accel_dataset()

    for p in ['val', 'train']:
        sample_label = []
        fp = np.zeros((len(idxs[p]), 3, 165), dtype=np.float32)
        for j, i in enumerate(idxs[p]):
            label = feeder.examples[i]['label']
            fp[j, :, :] = accel[i, :, :]
            sample_label.append(label)

        data_out_path = '{}/accel3s_{}.npy'.format(out_folder, p)
        label_out_path = '{}/accel3s_{}_label.pkl'.format(out_folder, p)
        sample_name = [str(i) for i in range(0, len(idxs[p]))]
        with open(label_out_path, 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)

        np.save(data_out_path, fp)

