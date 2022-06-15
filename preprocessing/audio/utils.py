import os
from datetime import datetime

import pandas as pd
import numpy as np


def get_accel_chunks(fname, very_first_ts=1571927168.657):
    times = pd.read_csv(fname)['time'].to_list()

    timestamps = [float(datetime.strptime(
        t, '%Y-%m-%d %H:%M:%S.%f').timestamp()) for t in times]

    timestamps = np.array(timestamps[30:])
    diffs = np.diff(timestamps)

    idxs = np.where(diffs > 0.04)[0]
    times_first = timestamps[idxs] - very_first_ts
    times_last = timestamps[idxs + 1] - very_first_ts
    lengths = (times_last - times_first) / 0.017

    adjusted_times = pd.DataFrame(list(zip(
        times_first, times_last, lengths)), columns=['time_first', 'time_last', 'len'])
    # remove very small holes
    adjusted_times = adjusted_times[adjusted_times['len'] > 3]
    return adjusted_times


class AudioFile:
    def __init__(self, fname, chunk_size=256):
        with open(fname, 'rb') as f:
            self.bytes_list = list()
            byte = True
            while byte:
                byte = f.read(chunk_size)
                self.bytes_list.append(byte)

            
    def get_chunk(self, ini, end):
        return self.bytes_list[ini: end]

    def get_repeats(self):
        previous_byte = None
        in_repeat = False

        repeat_chunks_idxs = list()  # list of tuples containing the segments
        good_chunks_idxs = list()


        # print(len(bytes_list))
        first_repeat_i = 0
        first_good_i = 0
        num_chunks = 0

        for i, byte in enumerate(self.bytes_list):
            num_chunks += 1

            if byte == previous_byte:

                # going into a repeat
                if not in_repeat:
                    # add good chunk
                    good_chunks_idxs.append((first_good_i, i - 1))
                    first_repeat_i = i - 1
                in_repeat = True
            else:

                # going out of a repeat into a good chunk
                if in_repeat:
                    repeat_chunks_idxs.append(
                        (first_repeat_i, i))  # add repeat chunk
                    first_good_i = i
                in_repeat = False

            previous_byte = byte

        # now get the chunks from the audio bytes
        repeat_chunks = list()
        good_chunks = list()
        for c in good_chunks_idxs:
            good_chunks.append(
                {
                    'first': c[0],
                    'last': c[1],
                    'chunks': self.get_chunk(c[0], c[1]),
                    'tfirst': c[0] * 0.256 * 8 / 20,
                    'tlast': c[1] * 0.256 * 8 / 20
                })

        for c in repeat_chunks_idxs:
            repeat_chunks.append(
                {
                    'first': c[0],
                    'last': c[1],
                    'chunks': self.get_chunk(c[0], c[1]),
                    'tfirst': c[0] * 0.256 * 8 / 20,
                    'tlast': c[1] * 0.256 * 8 / 20
                })

        return good_chunks, repeat_chunks



def get_audio_chunks(fname, chunk_size=256):
    '''
    Returns a list of audio chunks.
    '''

    previous_byte = None
    in_repeat = False

    bytes_list = list()

    repeat_chunks_idxs = list()  # list of tuples containing the segments
    good_chunks_idxs = list()

    with open(fname, 'rb') as f:
        byte = True
        while byte:
            byte = f.read(chunk_size)
            bytes_list.append(byte)

    # print(len(bytes_list))
    first_repeat_i = 0
    first_good_i = 0
    num_chunks = 0

    for i, byte in enumerate(bytes_list):
        num_chunks += 1

        if byte == previous_byte:

            # going into a repeat
            if not in_repeat:
                # add good chunk
                good_chunks_idxs.append((first_good_i, i - 1))
                first_repeat_i = i - 1
            in_repeat = True
        else:

            # going out of a repeat into a good chunk
            if in_repeat:
                repeat_chunks_idxs.append(
                    (first_repeat_i, i))  # add repeat chunk
                first_good_i = i
            in_repeat = False

        previous_byte = byte

    # print(len(good_chunks_idxs), len(repeat_chunks_idxs))

    # now get the chunks from the audio bytes
    repeat_chunks = list()
    good_chunks = list()
    for c in good_chunks_idxs:
        good_chunks.append(
            {
                'first': c[0],
                'last': c[1],
                'chunks': bytes_list[c[0]:c[1]],
                'tfirst': c[0] * 0.256 * 8 / 20,
                'tlast': c[1] * 0.256 * 8 / 20
            })

    for c in repeat_chunks_idxs:
        repeat_chunks.append(
            {
                'first': c[0], 
                'last': c[1], 
                'chunks': bytes_list[c[0]:c[1]],
                'tfirst': c[0] * 0.256 * 8 / 20,
                'tlast': c[1] * 0.256 * 8 / 20
            })

    return good_chunks, repeat_chunks
