import os

from datetime import datetime, timedelta

# path to the conflab dataset files
conflab_path = '/mnt/e/data/conflab/release'


# paths to raw data
raw_ss_path = os.path.join(conflab_path, 'annotations', 'actions', 'speaking_status', 'raw')
raw_audio_path = os.path.join(conflab_path, 'data_raw', 'audio')
raw_wearables_path = os.path.join(conflab_path, 'data_raw', 'wearables')
raw_ff_data_path = os.path.join(conflab_path, 'annotations', 'f_formations', 'raw')

# paths to processed data
processed_ss_path = os.path.join(conflab_path, 'annotations', 'actions', 'speaking_status', 'processed')
processed_pose_path = os.path.join(conflab_path, 'annotations', 'pose', 'coco')
processed_wearables_path = os.path.join(conflab_path, 'data_processed', 'wearables')
processed_ff_data_path = os.path.join(conflab_path, 'annotations', 'f_formations', 'processed')


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

vid_deltas = {
    'vid2_seg8': timedelta(minutes=14),
    'vid2_seg9':  timedelta(minutes=16),
    'vid3_seg1':  timedelta(minutes=0),
    'vid3_seg2':  timedelta(minutes=2),
    'vid3_seg3':  timedelta(minutes=4),
    'vid3_seg4': timedelta(minutes=6),
    'vid3_seg5':  timedelta(minutes=8),
    'vid3_seg6':  timedelta(minutes=10),
}

camera_raw_to_segment = {
    "cam2": {2: "GH020003.MP4", 3: "GH030003.MP4", 4: "GH040003.MP4", 5: "GH050003.MP4"},
    "cam4": {2: "GH020010.MP4", 3: "GH030010.MP4", 4: "GH040010.MP4", 5: "GH050010.MP4"},
    "cam6": {2: "GH020162.MP4", 3: "GH030162.MP4", 4: "GH040162.MP4", 5: "GH050162.MP4"},
    "cam8": {2: "GH020165.MP4", 3: "GH030165.MP4", 4: "GH040165.MP4", 5: "GH050165.MP4"},
    "cam10": {2: "GH020009.MP4", 3: "GH030009.MP4", 4: "GH040009.MP4", 5: "GH050009.MP4"},
}

camera_was_rotated_map = {
    "cam2": False,
    "cam4": True,
    "cam6": False,
    "cam8": False,
    "cam10": False,
}

annotated_section_start = vid_timecodes['vid2_seg8']
annotated_section_end = vid3_start + timedelta(minutes=12)
vid3_offset_from_vid2 = vid3_start - annotated_section_start
annotated_section_len = (annotated_section_end - annotated_section_start)

