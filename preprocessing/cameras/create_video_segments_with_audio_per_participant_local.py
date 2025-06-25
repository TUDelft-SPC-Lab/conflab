"""
This script creates video segments with synchronized audio for each participant.
This is a local version that works without staff-bulk mounting.
"""

from pathlib import Path
import sys
from datetime import datetime, timedelta
from collections import defaultdict

grandparent_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_id_to_dict_of_video_index_to_raw_video_file_basename,
    CAMERAS_OF_INTEREST,
    TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES,
    NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE,
    PARTICIPANTS_IDS_TO_IGNORE,
)
from ffmpeg_utils import (  # noqa: E402
    extract_timecode_from_conflab_raw_go_pro_video_metadata,
    get_video_duration_in_seconds,
    subprocess_run_with_guardfile,
)

##############################################################
# CONFIGURATION OF THE SCRIPT
DISPLAY_PROGRESS_BARS = True
PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO: bool = False  # only audio videos
CREATE_CSV_FILE_WITH_TIMECODES: bool = True
IGNORE_EXISTING_FILES: bool = False
##############################################################

# Local paths configuration
RAW_VIDEOS_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras/video")
VIDEO_SEGMENTS_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/modification/conflab")
SYNCED_AUDIO_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/audio/synced")
AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/modification/conflab/audio_segments_per_participant")

def extract_timecode_from_conflab_raw_go_pro_video_metadata_from_cam_and_video_index(
    camera_index: int, video_index: int
) -> datetime:
    raw_video_file_path = (
        RAW_VIDEOS_FOLDER
        / f"cam{camera_index:02}"
        / camera_id_to_dict_of_video_index_to_raw_video_file_basename[
            f"cam{camera_index}"
        ][video_index]
    )
    return extract_timecode_from_conflab_raw_go_pro_video_metadata(raw_video_file_path)

def main():
    timecodes_of_4G_raw_videos_for_cam_index: dict[int, dict[int, datetime]] = defaultdict(dict)

    if CREATE_CSV_FILE_WITH_TIMECODES:
        csv_file = open(Path(__file__).resolve().parents[1] / "timecodes.csv", "w")
        csv_file.write(
            ",".join(
                [
                    "CAMERA",
                    "AUDIO_TIMECODE",
                    "RAW_VIDEO",
                    "RAW_VIDEO_TIMECODE",
                    "SEGMENT",
                    "DURATION",
                    "SEGMENT_TIMECODE",
                    "AUDIO_START",
                ]
            )
        )
        csv_file.write("\n")

    for camera_index in CAMERAS_OF_INTEREST:
        # Extract timecodes for all raw videos
        for video_index in camera_id_to_dict_of_video_index_to_raw_video_file_basename[
            f"cam{camera_index}"
        ].keys():
            timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index] = (
                extract_timecode_from_conflab_raw_go_pro_video_metadata_from_cam_and_video_index(
                    camera_index=camera_index, video_index=video_index
                )
            )

        # Create folders for audio segments and video segments
        audio_segments_per_participant_folder = (
            AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER
            / f"cam{camera_index}"
        )
        audio_segments_per_participant_folder.mkdir(parents=True, exist_ok=True)

        video_with_audio_or_only_audio_subfolder = (
            "segments-with-audio-per-participant"
            if not PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO
            else "segments-only-audio-per-participant"
        )
        video_segments_with_participant_audio_for_camera_folder_path = (
            VIDEO_SEGMENTS_FOLDER
            / video_with_audio_or_only_audio_subfolder
            / f"cam{camera_index}"
        )
        video_segments_with_participant_audio_for_camera_folder_path.mkdir(
            parents=True, exist_ok=True
        )

        # Process each raw video
        for video_index, raw_video_file_basename in camera_id_to_dict_of_video_index_to_raw_video_file_basename[
            f"cam{camera_index}"
        ].items():
            # Process each 2-minute segment
            for segment_index in range(1, 10):
                video_segment_file_basename = (
                    f"vid{video_index}-seg{segment_index}-scaled-denoised.mp4"
                )
                video_segment_path = (
                    VIDEO_SEGMENTS_FOLDER
                    / f"cam{camera_index}"
                    / video_segment_file_basename
                )

                if not video_segment_path.exists():
                    continue

                # Calculate segment timing
                video_segment_duration: timedelta = timedelta(
                    seconds=get_video_duration_in_seconds(video_segment_path)
                )
                timecode_for_video_segment: datetime = (
                    timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index]
                    + timedelta(minutes=2 * (segment_index - 1))
                )
                video_segment_start_time_wrt_synced_audio_wav_files: timedelta = (
                    timecode_for_video_segment
                    - TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES
                )

                if CREATE_CSV_FILE_WITH_TIMECODES:
                    csv_file.write(
                        ",".join(
                            [
                                f"cam{camera_index}",
                                str(TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES),
                                camera_id_to_dict_of_video_index_to_raw_video_file_basename[
                                    f"cam{camera_index}"
                                ][video_index],
                                str(timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index]),
                                f"vid{video_index}_seg{segment_index}",
                                str(video_segment_duration),
                                str(timecode_for_video_segment),
                                str(video_segment_start_time_wrt_synced_audio_wav_files),
                            ]
                        )
                    )
                    csv_file.write("\n")
                    csv_file.flush()

                # Process each participant's audio
                for participant_index in range(1, NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE + 1):
                    if participant_index in PARTICIPANTS_IDS_TO_IGNORE:
                        continue

                    # Get participant audio file
                    participant_audio_wav_file_path = SYNCED_AUDIO_FOLDER / f"{participant_index}.wav"
                    if not participant_audio_wav_file_path.exists():
                        print(f"Warning: Audio file not found for participant {participant_index}")
                        continue

                    # Define output paths
                    video_segment_wav_audio_for_participant_file_path = (
                        audio_segments_per_participant_folder
                        / f"vid{video_index}_seg{segment_index}-participant{participant_index}.wav"
                    )

                    video_segment_with_participant_audio_path = (
                        video_segments_with_participant_audio_for_camera_folder_path
                        / f"{video_segment_path.stem}-participant{participant_index}.mp4"
                    )

                    # 1. Clip the audio
                    if not video_segment_wav_audio_for_participant_file_path.exists() or IGNORE_EXISTING_FILES:
                        cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel", "error",
                            "-i", str(participant_audio_wav_file_path),
                            "-vcodec", "copy",
                            "-acodec", "copy",
                            "-copyinkf",
                            "-ss", f"0{video_segment_start_time_wrt_synced_audio_wav_files}",
                            "-t", f"0{video_segment_duration}",
                            "-y",
                            str(video_segment_wav_audio_for_participant_file_path),
                        ]
                        subprocess_run_with_guardfile(
                            cmd,
                            video_segment_wav_audio_for_participant_file_path.with_suffix(".isincomplete.txt"),
                        )

                    # 2. Combine video and audio
                    if not video_segment_with_participant_audio_path.exists() or IGNORE_EXISTING_FILES:
                        if PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO:
                            cmd = [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel", "error",
                                "-f", "lavfi",
                                "-i", "color=c=black:s=256x144:r=59.94",
                                "-i", str(video_segment_wav_audio_for_participant_file_path),
                                "-c:v", "libx264",
                                "-c:a", "aac",
                                "-shortest",
                                str(video_segment_with_participant_audio_path),
                                "-y",
                            ]
                        else:
                            cmd = [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel", "panic",
                                "-i", str(video_segment_path),
                                "-i", str(video_segment_wav_audio_for_participant_file_path),
                                "-c:v", "copy",
                                "-map", "0:v:0",
                                "-map", "1:a:0",
                                str(video_segment_with_participant_audio_path),
                                "-y",
                            ]

                        subprocess_run_with_guardfile(
                            cmd,
                            video_segment_with_participant_audio_path.with_suffix(".isincomplete.txt"),
                        )

if __name__ == "__main__":
    main() 