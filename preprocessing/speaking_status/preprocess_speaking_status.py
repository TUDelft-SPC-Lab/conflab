import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from conflab.constants import conflab_speaking_status_path, conflab_raw_speaking_status_path
from conflab.preprocessing.tools import interpolate

def parse_fname(fname):
    if '-' in fname:
        person_part = fname.split('-')[0]
        conf_part = fname.split('-')[1]
        pid = int(person_part[6:])
        subm = int(conf_part.split('_')[1])
        return pid, 'conf', subm
    else:
        person_part = fname.split('_')[0]
        subm = int(fname.split('_')[1])
        pid = int(person_part[6:])
        return pid, 'ss', subm

def read_labels(raw_labels_path):
    labels = {} # {[segment]: {[pid]: {ss: dataframe, conf: dataframe}}}
    for seg_folder in tqdm(Path(raw_labels_path).glob('*')):
        seg_name = os.path.basename(seg_folder)
        labels[seg_name] = {}
        for fpath in seg_folder.glob('*.csv'):
            fname = os.path.basename(fpath).split('.')[0]
            if '_av_' in fname or 'Sample' in fname:
                continue
            pid, signal, subm= parse_fname(fname)
            if pid not in labels[seg_name]:
                labels[seg_name][pid] = {'ss': None, 'conf': None}

            d = pd.read_csv(fpath)
            labels[seg_name][pid][signal] = d
    return labels

def preprocess_labels_in_place(labels):
    for seg, people in tqdm(labels.items()):
        for pid in people.keys():
            people[pid]['ss'] = interpolate(
                people[pid]['ss'], 
                method='nearest',
                axis=0).ffill().bfill()
            if people[pid]['conf'] is not None:
                people[pid]['conf'] = interpolate(
                    people[pid]['conf'], 
                    method='linear',
                    limit= 1000,
                    limit_direction= 'both',
                    axis=0)

def write_csv_labels(labels: dict, processed_labels_path):
    if not os.path.exists(os.path.join(processed_labels_path, 'speaking')):
        os.mkdir(os.path.join(processed_labels_path, 'speaking'))
    if not os.path.exists(os.path.join(processed_labels_path, 'confidence')):
        os.mkdir(os.path.join(processed_labels_path, 'confidence'))
    for seg, people in tqdm(labels.items()):
        speaking_data = [(pid, people[pid]['ss']['data0']) for pid in people.keys()]
        confidence_data = [(pid, people[pid]['conf']['data0']) for pid in people.keys() if people[pid]['conf'] is not None]
        
        # store the speaking data
        pids, data = zip(*speaking_data)
        df=pd.concat(data, axis=1)
        df.columns=pids
        df.to_csv(
            os.path.join(processed_labels_path, 'speaking', f'{seg}.csv'),
            index=False
        )

        # store the confidence data
        pids, data = zip(*confidence_data)
        df=pd.concat(data, axis=1)
        df.columns=pids
        df.to_csv(
            os.path.join(processed_labels_path, 'confidence', f'{seg}.csv'),
            index=False
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This script converts the labels from covfee output files into CSV dataframes')
    parser.add_argument('--raw_labels_path', help='path to raw data (output of Covfee)', required=False, default=conflab_raw_speaking_status_path)
    parser.add_argument('--processed_labels_path', help='path to output folder', required=False, default=conflab_speaking_status_path)
    args = parser.parse_args()

    labels = read_labels(args.raw_labels_path)
    preprocess_labels_in_place(labels)
    write_csv_labels(labels, args.processed_labels_path)