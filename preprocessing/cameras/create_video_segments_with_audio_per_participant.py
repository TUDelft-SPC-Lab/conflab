from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime, timedelta
from collections import defaultdict

grandparent_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent_dir))

from constants import (  # noqa: E402
    camera_raw_to_segment,
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
from ffmpeg_utils import (  # noqa: E402
    extract_timecode_from_conflab_raw_go_pro_video_metadata,
    get_video_duration_in_seconds,
    subprocess_run_with_guardfile,
)

##############################################################3
# CONFIGURATION OF THE SCRIPT
DISPLAY_PROGRESS_BARS = True
PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO: bool = False  # only audio videos
CREATE_CSV_FILE_WITH_TIMECODES: bool = True
IGNORE_EXISTING_FILES: bool = False
##############################################################3

check_if_staff_bulk_is_mounted()

timecodes_of_4G_raw_videos_for_cam_index: dict[int, dict[int, datetime]] = defaultdict(
    dict
)

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


for camera_index in tqdm(
    CAMERAS_OF_INTEREST, desc="Camera", disable=not DISPLAY_PROGRESS_BARS
):
    # We extract the timecodes embedded in the raw videos for all 4GB segments
    for video_index, raw_video_file_basename in camera_raw_to_segment[
        f"cam{camera_index}"
    ].items():
        raw_video_file_path = (
            RAW_VIDEOS_FOLDER_IN_STAFF_BULK
            / f"cam{camera_index:02}"
            / raw_video_file_basename
        )

        timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index] = (
            extract_timecode_from_conflab_raw_go_pro_video_metadata(raw_video_file_path)
        )

        if camera_index == 4 and video_index in [2, 3]:
            reference_timecode = (
                CAM4_VID2_START_TIMECODE
                if video_index == 2
                else CAM4_VID3_START_TIMECODE
            )
            problem_with_ffprobe_timecode_extraction_found: bool = (
                timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index]
                != reference_timecode
            )

            assert not problem_with_ffprobe_timecode_extraction_found, f"""
            FATAL ISSUE! The timecode being extracted by ffprobe seems inconsistent with
            the few reference values manually extracted for conflab and hardcoded in constants.py.
            This may be an issue with ffmpeg/ffprobe, and this problem has been observed in a subset
            of PCs, but not in all of them. Try fixing your system or try another PC. We have no
            clue what is causing this issue!!
            Extracted: {timecodes_of_4G_raw_videos_for_cam_index[camera_index][video_index]} vs Reference: {reference_timecode}
            """

    # We create the folder for the audio segments per participant
    audio_segments_per_participant_folder = (
        AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER_FOR_ALL_CAMS_IN_LOCAL
        / f"cam{camera_index}"
    )
    audio_segments_per_participant_folder.mkdir(parents=True, exist_ok=True)

    video_with_audio_or_only_audio_subfolder = (
        "video_with_audio"
        if not PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO
        else "only_audio"
    )
    video_segments_with_participant_audio_for_camera_folder_path = (
        VIDEO_SEGMENTS_FOLDER_IN_LOCAL
        / video_with_audio_or_only_audio_subfolder
        / f"cam{camera_index}"
    )
    video_segments_with_participant_audio_for_camera_folder_path.mkdir(
        parents=True, exist_ok=True
    )

    # Now we traverse raw videos of 4G, associated to the given camera_index
    for video_index, raw_video_file_basename in tqdm(
        camera_raw_to_segment[f"cam{camera_index}"].items(),
        desc=f"-Raw videos on cam{camera_index}",
        leave=False,
        disable=not DISPLAY_PROGRESS_BARS,
    ):
        raw_video_basename = camera_raw_to_segment[f"cam{camera_index}"][video_index]
        # Now we traverse over the video segments, i.e., the videos clipped in 2 minutes
        # segments, extracted from each raw video segment of 4G
        for segment_index in tqdm(
            range(1, 10),
            desc=f"--Segments on {raw_video_basename}",
            leave=False,
            disable=not DISPLAY_PROGRESS_BARS,
        ):
            video_segment_file_basename = (
                f"vid{video_index}-seg{segment_index}-scaled-denoised.mp4"
            )
            # We get the video path. First from the local storage, which is faster, and if
            # it does not exists (for example, it hasn't been copied) we get it from staff
            # bulk storage
            video_segment_path = (
                VIDEO_SEGMENTS_FOLDER_IN_LOCAL
                / f"cam{camera_index}"
                / video_segment_file_basename
            )
            if not video_segment_path.exists():
                video_segment_path = (
                    VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK
                    / f"cam{camera_index}"
                    / video_segment_file_basename
                )
            #

            # The start of the segment is the the start of the video plus the delta for the segment
            if not video_segment_path.exists():
                continue

            # Now we compute the start of the 2 mins video segment, but with respect to
            # the audio wav file, as we need to clipped the audio, to merge it with the video
            # segment.
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
                # For debugging purposes, we write the timecodes to a csv file
                csv_file.write(
                    ",".join(
                        [
                            f"cam{camera_index}",
                            str(TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES),
                            camera_raw_to_segment[f"cam{camera_index}"][video_index],
                            str(
                                timecodes_of_4G_raw_videos_for_cam_index[camera_index][
                                    video_index
                                ]
                            ),
                            f"vid{video_index}_seg{segment_index}",
                            str(video_segment_duration),
                            str(timecode_for_video_segment),
                            str(video_segment_start_time_wrt_synced_audio_wav_files),
                        ]
                    )
                )
                csv_file.write("\n")
                csv_file.flush()

            # Now that we have computed the time offsets between the video segment and
            # the audio, we proceed to clip the audio, and merge it with the video segment
            for participant_index in tqdm(
                range(1, NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE + 1),
                desc=f"---Participants for vid{video_index}_seg{segment_index}",
                leave=False,
                disable=not DISPLAY_PROGRESS_BARS,
            ):
                if participant_index in PARTICIPANTS_IDS_TO_IGNORE:
                    continue
                # We now define to the audio file path, with preference in local storage
                # for processing speed reasons
                participant_audio_wav_file_path = (
                    SYNCED_AUDIO_FOLDER_IN_LOCAL / f"{participant_index}.wav"
                )
                if not participant_audio_wav_file_path.exists():
                    participant_audio_wav_file_path = (
                        SYNCED_AUDIO_FOLDER_IN_STAFF_BULK / f"{participant_index}.wav"
                    )

                # We define the path for the clipped audio segment file for the participant
                video_segment_wav_audio_for_participant_file_path = (
                    audio_segments_per_participant_folder
                    / f"vid{video_index}_seg{segment_index}-participant{participant_index}.wav"
                )

                # We define the path for the clipped video segment with the audio of the participant
                video_segment_with_participant_audio_path = (
                    video_segments_with_participant_audio_for_camera_folder_path
                    / f"{video_segment_path.stem}-participant{participant_index}.mp4"
                )

                # Crop the audio to the same length and offset as the video

                ######## Now we execute the ffmpeg commands to clip the audio and merge with the video #############

                # 1) Clipping the audio
                # fmt: off
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
                # fmt: on
                if (
                    not video_segment_wav_audio_for_participant_file_path.exists()
                    or IGNORE_EXISTING_FILES
                ):
                    subprocess_run_with_guardfile(
                        cmd,
                        video_segment_wav_audio_for_participant_file_path.with_suffix(
                            ".isincomplete.txt"
                        ),
                    )

                # 2) Combine the video and audio
                if PRODUCE_VIDEO_PER_PARTICIPANT_WITH_BLACK_VIDEO:
                    # NOTE: this is re-encoding the audio because remuxing directly does not work
                    # fmt: off
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-f", "lavfi",
                        "-i", "color=c=black:s=256x144:r=59.94", # 144p resolution for faster processing and 59.94 fps, same as the video
                        "-i", str(video_segment_wav_audio_for_participant_file_path),
                        "-c:v", "libx264",
                        "-c:a", "aac", # Audio codec pcm_s16le is not supported in mp4, so re-encode to aac
                        "-shortest", # The black filter is infinite length, stop when the audio stops
                        str(video_segment_with_participant_audio_path),
                        "-y",
                    ]
                    # fmt: on
                else:
                    # fmt: off
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
                    # fmt: on

                if (
                    not video_segment_with_participant_audio_path.exists()
                    or IGNORE_EXISTING_FILES
                ):
                    subprocess_run_with_guardfile(
                        cmd,
                        video_segment_with_participant_audio_path.with_suffix(
                            ".isincomplete.txt"
                        ),
                    )
