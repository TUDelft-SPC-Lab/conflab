
from pathlib import Path
import sys

grandparent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_id_to_dict_of_video_index_to_raw_video_file_basename,
    CAMERAS_OF_INTEREST,
    TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES,
    RAW_VIDEOS_FOLDER_IN_STAFF_BULK,
    VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK,
    VIDEO_SEGMENTS_FOLDER_IN_LOCAL,
    SYNCED_AUDIO_FOLDER_IN_LOCAL,
    SYNCED_AUDIO_FOLDER_IN_STAFF_BULK,
    NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE,
    PARTICIPANTS_IDS_TO_IGNORE,
    AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER_FOR_ALL_CAMS_IN_LOCAL,
    CAM4_VID2_START_TIMECODE,
    CAM4_VID3_START_TIMECODE,
    check_if_staff_bulk_is_mounted,
)

import cv2

selected_camera_index: int = 10

def on_trackbar_change(pos):
    global frame_index
    frame_index = pos

def main():
    video_path = '/path/to/video.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('Video Player')
    cv2.createTrackbar('Frame', 'Video Player', 0, total_frames - 1, on_trackbar_change)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video Player', frame)

        # Update the trackbar position
        frame_index = cv2.getTrackbarPos('Frame', 'Video Player')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Check for keyboard input
        key = cv2.waitKey(int(1000 / fps))
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()