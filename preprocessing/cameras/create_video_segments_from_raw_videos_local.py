"""
This script traverses over the raw videos for each camera, and extracts video segments of 2 minutes each.
This is a local version that works without staff-bulk mounting.
"""

from pathlib import Path
import sys
import math
import subprocess
from datetime import datetime, timedelta
import os

grandparent_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_id_to_dict_of_video_index_to_raw_video_file_basename,
    camera_was_rotated_map,
    CAMERAS_OF_INTEREST,
)
from ffmpeg_utils import get_video_duration_in_seconds, subprocess_run_with_guardfile  # noqa: E402

# Local paths configuration
RAW_VIDEOS_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras/video")  # Local folder for raw videos
VIDEO_SEGMENTS_FOLDER = Path("/home/zonghuan/tudelft/projects/datasets/modification/conflab/segments")  # Local folder for processed segments

def get_video_timecode(video_path):
    """Get the timecode of a video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=timecode",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timecode = result.stdout.strip()
        if timecode:
            # Parse timecode (HH:MM:SS:FF format)
            hours, minutes, seconds, frames = map(int, timecode.split(':'))
            return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=frames*1000/59.94)  # Assuming 30fps
        return timedelta(0)
    except subprocess.CalledProcessError:
        print(f"Warning: Could not get timecode for {video_path}")
        return timedelta(0)

def main():
    # We iterate over all the cameras for which we are interested in extracting segments
    for camera_index in CAMERAS_OF_INTEREST:
        camera_was_rotated = camera_was_rotated_map[f"cam{camera_index}"]

        # Then we iterate over all the raw videos for the given camera
        for (
            video_index,
            raw_video_file_basename,
        ) in camera_id_to_dict_of_video_index_to_raw_video_file_basename[
            f"cam{camera_index}"
        ].items():
            if video_index < 2:
                # For the MINGLE experiments, we want to annotate video segments AFTER vid3-seg6.
                continue

            raw_video_file_path = (
                RAW_VIDEOS_FOLDER
                / f"cam{camera_index:02}"
                / raw_video_file_basename
            )

            if not raw_video_file_path.exists():
                print(f"Raw video not found: {raw_video_file_path}")
                continue

            if camera_was_rotated:
                # Handle rotated videos
                raw_rotated_file_basename = raw_video_file_basename.replace(
                    ".MP4", "_rot.MP4"
                )
                raw_rotated_video_file_path = (
                    RAW_VIDEOS_FOLDER
                    / f"cam{camera_index:02}"
                    / raw_rotated_file_basename
                )

                if not raw_rotated_video_file_path.exists():
                    raw_rotated_video_file_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(raw_video_file_path),
                        "-c",
                        "copy",
                        "-metadata:s:v:0",
                        "rotate=0",
                        str(raw_rotated_video_file_path),
                    ]
                    with open(os.devnull, 'w') as devnull:
                        subprocess_run_with_guardfile(
                            cmd,
                            raw_rotated_video_file_path.with_suffix(".isincomplete.txt"),
                            stdout=devnull,
                            stderr=devnull
                        )

                # raw_video_file_path = raw_rotated_video_file_path

            # Get initial timecode of the raw video
            initial_timecode = get_video_timecode(raw_video_file_path)

            # Extract duration and calculate number of segments
            video_duration_in_seconds = get_video_duration_in_seconds(raw_video_file_path)
            number_of_segments = math.ceil(video_duration_in_seconds / 120)

            # Create output directory
            video_segments_folder_path = VIDEO_SEGMENTS_FOLDER / f"cam{camera_index}"
            video_segments_folder_path.mkdir(parents=True, exist_ok=True)

            # Process each segment
            for segment_index in range(1, number_of_segments + 1):
                video_segment_file_basename = f"vid{video_index}-seg{segment_index}.mp4"
                
                # Calculate segment start timecode
                segment_start_timecode = initial_timecode + timedelta(minutes=2 * (segment_index - 1))
                print(f"cam{camera_index:02} {video_segment_file_basename}: {segment_start_timecode}")

                # Check if final output already exists
                final_output_path = video_segments_folder_path / video_segment_file_basename
                if final_output_path.exists():
                    continue

                # Trim the video segment
                fast_seek_position = (
                    f"00:{(2*(segment_index-1)-1):02}:40"
                    if segment_index > 1
                    else "00:00:00"
                )
                slow_seek_position = "00:00:20" if segment_index > 1 else "00:00:00"
                
                cmd = [
                    "ffmpeg",
                    "-ss",
                    fast_seek_position,
                    "-i",
                    str(raw_video_file_path),
                    "-vcodec",
                    "copy",
                    "-acodec",
                    "copy",
                    "-copyinkf",
                    "-ss",
                    slow_seek_position,
                    "-t",
                    "00:02:00",
                    str(final_output_path),
                ]
                with open(os.devnull, 'w') as devnull:
                    subprocess_run_with_guardfile(
                        cmd,
                        final_output_path.with_suffix(".isincomplete.txt"),
                        stdout=devnull,
                        stderr=devnull
                    )


if __name__ == "__main__":
    main() 