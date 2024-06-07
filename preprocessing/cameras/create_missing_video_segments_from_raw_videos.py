"""
This script traverses over the raw videos for each camera, and extracts video segments of 2 minutes each.

It replicates the ffmpeg commands which are listed in the videoSplitCamX.sh scripts, found in staff-bulk

 staff-bulk/ewi/insy/SPCDataSets/conflab-mm/processed/annotation/videoSegments/cam2/videoSplitCam2.sh

"""

from pathlib import Path
import subprocess
import os
import sys
import math

grandparent_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_raw_to_segment,
    camera_was_rotated_map,
    CAMERAS_OF_INTEREST,
    RAW_VIDEOS_FOLDER_IN_STAFF_BULK,
    VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK,
    VIDEO_SEGMENTS_FOLDER_IN_LOCAL,
    check_if_staff_bulk_is_mounted,
)
from ffmpeg_utils import get_video_duration_in_seconds, subprocess_run_with_guardfile  # noqa: E402

check_if_staff_bulk_is_mounted()


# We iterate over all the cameras for which we are interested in extracting segments
for camera_index in CAMERAS_OF_INTEREST:
    raw_videos_for_cam = camera_raw_to_segment[f"cam{camera_index}"]
    camera_was_rotated = camera_was_rotated_map[f"cam{camera_index}"]

    # Then we iterate over all the raw videos for the given camera, each will be referred
    # with a specific video_index, when generating the output videos. For example vid{video_index}-seg1.mp4
    for video_index, raw_video_file_basename in camera_raw_to_segment[
        f"cam{camera_index}"
    ].items():
        if video_index < 3:
            # For the MINGLE experiments, we want to annotate video segments AFTER vid3-seg6.
            # We don't get specific into the segment, because the logic below skips if the desired
            # segment video already exists anyway.
            continue

        raw_video_file_path = (
            RAW_VIDEOS_FOLDER_IN_STAFF_BULK
            / f"cam{camera_index:02}"
            / raw_video_file_basename
        )
        if camera_was_rotated:
            # Some cameras were rotated, and we make sure that the rotated version exists
            raw_rotated_file_basename = raw_video_file_basename.replace(
                ".MP4", "_rot.MP4"
            )
            raw_rotated_video_file_path_in_staff_bulk = (
                RAW_VIDEOS_FOLDER_IN_STAFF_BULK
                / f"cam{camera_index:02}"
                / raw_rotated_file_basename
            )

            if raw_rotated_video_file_path_in_staff_bulk.exists():
                raw_video_file_path = raw_rotated_video_file_path_in_staff_bulk
            else:
                raw_rotated_video_file_path_in_local = (
                    VIDEO_SEGMENTS_FOLDER_IN_LOCAL
                    / "raw_overhead"
                    / f"cam{camera_index:02}"
                    / raw_rotated_file_basename
                )

                if not raw_rotated_video_file_path_in_local.exists():
                    raw_rotated_video_file_path_in_local.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    # Reference from /mnt/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/raw/video/overhead/cam04/videoRot.sbatch
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(raw_video_file_path),
                        "-c",
                        "copy",
                        "-metadata:s:v:0",
                        "rotate=0",
                        str(raw_rotated_video_file_path_in_local),
                    ]
                    print("================= ROTATING =======================")
                    subprocess_run_with_guardfile(
                        cmd,
                        raw_rotated_video_file_path_in_local.with_suffix(
                            ".isincomplete.txt"
                        ),
                    )

                raw_video_file_path = raw_rotated_video_file_path_in_local

        # We extract the duration of the raw video to know how many video segments of 2 minutes we need to generate
        video_duration_in_seconds = get_video_duration_in_seconds(raw_video_file_path)
        number_of_segments = math.ceil(video_duration_in_seconds / 120)

        # Now we iterate over all video segments, i.e., the videos clipped in time for 2 minutes each
        for segment_index in range(1, number_of_segments + 1):
            # Now, for each of the segments we generate the video segment, scale it and denoise it.
            # However, we only do this if the target video does not exist, with priority in staff-bulk, followed
            # by the local path.

            # We generate the video segment for the given raw video
            video_segments_folder_path_in_staff_bulk_for_camera = (
                VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK / f"cam{camera_index}"
            )
            video_segments_folder_path_in_local_path_for_camera = (
                VIDEO_SEGMENTS_FOLDER_IN_LOCAL / f"cam{camera_index}"
            )

            video_segment_file_basename = f"vid{video_index}-seg{segment_index}.mp4"
            video_segment_scaled_file_basename = (
                f"vid{video_index}-seg{segment_index}-scaled.mp4"
            )
            video_segment_scaled_and_denoised_file_basename = (
                f"vid{video_index}-seg{segment_index}-scaled-denoised.mp4"
            )

            # If the target video had been processed, and exists in staff-bulk, we skip the processing.
            check_if_staff_bulk_is_mounted()
            video_segment_scaled_and_denoised_file_path_in_staff_bulk = (
                video_segments_folder_path_in_staff_bulk_for_camera
                / video_segment_scaled_and_denoised_file_basename
            )
            if video_segment_scaled_and_denoised_file_path_in_staff_bulk.exists():
                print(
                    f"[STAFFBULK] Video segment {video_segment_scaled_and_denoised_file_path_in_staff_bulk} already exists"
                )
                continue

            # If the target video had been processed, and exists locally, we skip the processing.
            video_segment_scaled_and_denoised_file_path_in_local = (
                video_segments_folder_path_in_local_path_for_camera
                / video_segment_scaled_and_denoised_file_basename
            )
            if video_segment_scaled_and_denoised_file_path_in_local.exists():
                print(
                    f"[LOCAL] Video segment {video_segment_scaled_and_denoised_file_path_in_local} already exists"
                )
                continue
            else:
                print(
                    f"Video segment {video_segment_scaled_and_denoised_file_basename} needs to be generated"
                )

            video_segments_folder_path_in_local_path_for_camera.mkdir(
                parents=True, exist_ok=True
            )

            # The target video doesn't exist, so we will create it.
            video_segment_file_path = (
                video_segments_folder_path_in_local_path_for_camera
                / video_segment_file_basename
            )

            # First - we trim the video for the respective segment, if it doesn't exist already
            if not video_segment_file_path.exists():
                fast_seek_position = (
                    f"00:{(2*(segment_index-1)-1):02}:40"
                    if segment_index > 1
                    else "00:00:00"
                )
                # The slow seek position is 20 seconds after the fast seek position, i.e.,
                # it is defined relative to the fast seek position.
                slow_seek_position = "00:00:20" if segment_index > 1 else "00:00:00"
                # Our desired position is fast_seek_position + slow_seek_position
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
                    str(video_segment_file_path),
                ]
                print("================= Trimming =======================")
                subprocess_run_with_guardfile(
                    cmd, video_segment_file_path.with_suffix(".isincomplete.txt")
                )

            # Second - we scale the video segment if it doesn't exist already
            video_segment_scaled_file_path = (
                video_segments_folder_path_in_local_path_for_camera
                / video_segment_scaled_file_basename
            )
            if not video_segment_scaled_file_path.exists():
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_segment_file_path),
                    "-s",
                    "960x540",
                    "-c:a",
                    "copy",
                    "-copyinkf",
                    str(video_segment_scaled_file_path),
                ]
                print("================= Scaling =======================")
                subprocess_run_with_guardfile(
                    cmd, video_segment_scaled_file_path.with_suffix(".isincomplete.txt")
                )

            # Third - we denoise the video segment
            cmd = [
                "ffmpeg",
                "-i",
                str(video_segment_scaled_file_path),
                "-vf",
                "hqdn3d=luma_tmp=30",
                "-vcodec",
                "libx264",
                "-tune",
                "film",
                str(video_segment_scaled_and_denoised_file_path_in_local),
            ]
            print("================= Denoising =======================")
            subprocess_run_with_guardfile(
                cmd,
                video_segment_scaled_and_denoised_file_path_in_local.with_suffix(
                    ".isincomplete.txt"
                ),
            )
