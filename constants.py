import os

from datetime import datetime, timedelta

# path to the conflab dataset files
conflab_path = '/mnt/e/data/conflab/release'

conflab_pose_path = os.path.join(conflab_path, 'pose')

midge_raw_data_path = os.path.join(conflab_path, 'data_raw' 'wearables', 'raw')
midge_data_path = os.path.join(conflab_path, 'data_processed', 'wearables')

conflab_speaking_status_path = os.path.join(conflab_path, 'annotations', 'actions', 'speaking_status', 'processed')
conflab_raw_speaking_status_path = os.path.join(conflab_path, 'annotations', 'actions', 'speaking_status', 'raw')

ff_raw_data_path = os.path.join(conflab_path, 'annotations', 'f_formations', 'raw')
ff_processed_data_path = os.path.join(conflab_path, 'annotations', 'f_formations', 'processed')

# timecode information
# used to match video timecode with Midge timecode
vid2_start = datetime(2019, 10, 24, 16, 49, 36, int(1000000 * 58/59.94)) # start timecode of vid2 : 14:49:36:58
vid3_start = datetime(2019, 10, 24, 17, 7, 13, int(1000000 * 58/59.94)) # start timecode of vid3 : 15:07:13:58

vid_timecodes = {
    'vid2_seg8': vid2_start + timedelta(minutes=14),
    'vid2_seg9': vid2_start + timedelta(minutes=16),
    'vid3_seg1': vid3_start + timedelta(minutes=0),
    'vid3_seg2': vid3_start + timedelta(minutes=2),
    'vid3_seg3': vid3_start + timedelta(minutes=4),
    'vid3_seg4': vid3_start + timedelta(minutes=6),
    'vid3_seg5': vid3_start + timedelta(minutes=8),
    'vid3_seg6': vid3_start + timedelta(minutes=10),
}

annotated_section_start = vid_timecodes['vid2_seg8']
annotated_section_end = vid3_start + timedelta(minutes=12)
vid3_offset_from_vid2 = vid3_start - annotated_section_start
annotated_section_len = (annotated_section_end - annotated_section_start)