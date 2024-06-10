import os

from datetime import datetime, timedelta
from pathlib import Path

# path to the conflab dataset files
conflab_path = "/mnt/e/data/conflab/release"


# paths to raw data
raw_ss_path = os.path.join(
    conflab_path, "annotations", "actions", "speaking_status", "raw"
)
raw_audio_path = os.path.join(conflab_path, "data_raw", "audio")
raw_wearables_path = os.path.join(conflab_path, "data_raw", "wearables")
raw_ff_data_path = os.path.join(conflab_path, "annotations", "f_formations", "raw")

# paths to processed data
processed_ss_path = os.path.join(
    conflab_path, "annotations", "actions", "speaking_status", "processed"
)
processed_pose_path = os.path.join(conflab_path, "annotations", "pose", "coco")
processed_wearables_path = os.path.join(conflab_path, "data_processed", "wearables")
processed_ff_data_path = os.path.join(
    conflab_path, "annotations", "f_formations", "processed"
)

RAW_VIDEOS_FRAMERATE: float = 59.94

# timecode information / manually retrieved by the original authors of
# conflab.
# Used to match video timecode with Midge timecode
CAM4_VID2_START_TIMECODE = datetime(
    2019, 10, 24, 16, 49, 36, int(1000000 * 58 / RAW_VIDEOS_FRAMERATE)
)  # start timecode of vid2 : 14:49:36:58
CAM4_VID3_START_TIMECODE = datetime(
    2019, 10, 24, 17, 7, 13, int(1000000 * 58 / RAW_VIDEOS_FRAMERATE)
)  # start timecode of vid3 : 15:07:13:58

# All synched audio files start at the same time
TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES = datetime.fromtimestamp(1571927168.657)

NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE: int = 50
PARTICIPANTS_IDS_TO_IGNORE: list[int] = [38, 39]
CAMERAS_OF_INTEREST = [4, 2, 6, 8, 10]

cam4_vid_timecodes: dict[str, datetime] = {
    "vid2_seg8": CAM4_VID2_START_TIMECODE + timedelta(minutes=14),
    "vid2_seg9": CAM4_VID2_START_TIMECODE + timedelta(minutes=16),
    "vid3_seg1": CAM4_VID3_START_TIMECODE + timedelta(minutes=0),
    "vid3_seg2": CAM4_VID3_START_TIMECODE + timedelta(minutes=2),
    "vid3_seg3": CAM4_VID3_START_TIMECODE + timedelta(minutes=4),
    "vid3_seg4": CAM4_VID3_START_TIMECODE + timedelta(minutes=6),
    "vid3_seg5": CAM4_VID3_START_TIMECODE + timedelta(minutes=8),
    "vid3_seg6": CAM4_VID3_START_TIMECODE + timedelta(minutes=10),
}

camera_raw_to_segment = {
    "cam2": {
        2: "GH020003.MP4",
        3: "GH030003.MP4",
        4: "GH040003.MP4",
        5: "GH050003.MP4",
    },
    "cam4": {
        2: "GH020010.MP4",
        3: "GH030010.MP4",
        4: "GH040010.MP4",
        5: "GH050010.MP4",
    },
    "cam6": {
        2: "GH020162.MP4",
        3: "GH030162.MP4",
        4: "GH040162.MP4",
        5: "GH050162.MP4",
    },
    "cam8": {
        2: "GH020165.MP4",
        3: "GH030165.MP4",
        4: "GH040165.MP4",
        5: "GH050165.MP4",
    },
    "cam10": {
        2: "GH020009.MP4",
        3: "GH030009.MP4",
        4: "GH040009.MP4",
        5: "GH050009.MP4",
    },
}

camera_was_rotated_map = {
    "cam2": False,
    "cam4": True,
    "cam6": False,
    "cam8": False,
    "cam10": False,
}

annotated_section_start = cam4_vid_timecodes["vid2_seg8"]
annotated_section_end = CAM4_VID3_START_TIMECODE + timedelta(minutes=12)
vid3_offset_from_vid2 = CAM4_VID3_START_TIMECODE - annotated_section_start
annotated_section_len = annotated_section_end - annotated_section_start

# Paths to the raw dataset in the staff-bulk storage, mounted in the current PC.
RAW_VIDEOS_FOLDER_IN_STAFF_BULK = Path(
    "/data/conflab/data_raw/cameras/"
)
VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK = Path(
    "/data/conflab/data_processed/cameras/video_segments/"
)
SYNCED_AUDIO_FOLDER_IN_STAFF_BULK = Path(
    "/data/conflab/data_raw/audio/synced/"
)

def check_if_staff_bulk_is_mounted():
    if not RAW_VIDEOS_FOLDER_IN_STAFF_BULK.exists():
        raise FileNotFoundError(
            "Mount the bulk storage first in /mnt/staff-bulk\nUse: sshfs -o ro NETID@sftp.tudelft.nl:/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/release/release-final/data-raw/cameras/video/ /data/conflab/data_raw/cameras/"
        )


# Paths to the local storage
VIDEO_SEGMENTS_FOLDER_IN_LOCAL = Path("/data/conflab/data_processed/cameras/video_segments/")
SYNCED_AUDIO_FOLDER_IN_LOCAL = Path.home() / "Videos" / "conflab" / "audio" / "synced"
AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER_FOR_ALL_CAMS_IN_LOCAL = Path.home() / "Videos" / "conflab" / "audio" / "per_participant"
