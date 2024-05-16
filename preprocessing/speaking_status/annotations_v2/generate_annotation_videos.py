import subprocess
from pathlib import Path
import click
from tqdm import tqdm
import sys

sys.path.insert(0, "/home/era/code/neon-repos/conflab")
from constants import (
    vid_timecodes,
    vid_deltas,
    vid2_start,
    vid3_start,
    annotated_section_start,
)
from datetime import datetime, timedelta

# def tqdm(iterable, desc=None, total=None, leave=True):
#     return iterable

camera_raw_to_segment = {
    "cam2": {2: "GH020003.MP4", 3: "GH030003.MP4"},
    "cam4": {2: "GH020010.MP4", 3: "GH030010.MP4"},
    "cam6": {2: "GH020162.MP4", 3: "GH030162.MP4"},
    "cam8": {2: "GH020165.MP4", 3: "GH030165.MP4"},
    "cam10": {2: "GH020009.MP4", 3: "GH030009.MP4"},
}

camera_raw_timecodes = {
    "cam2": {2: None, 3: None},
    "cam4": {2: None, 3: None},
    "cam6": {2: None, 3: None},
    "cam8": {2: None, 3: None},
    "cam10": {2: None, 3: None},
}


@click.command()
@click.option("--wav-audio-folder", required=True, help="Folder with audio samples.")
@click.option("--raw-audio-folder", required=True, help="Folder with audio samples.")
@click.option(
    "--video-segments-folder", required=True, help="Folder with audio samples."
)
@click.option("--raw-videos-folder", required=True, help="Folder with audio samples.")
@click.option("--output-folder", required=True, help="Folder with audio samples.")
@click.option("--overwrite", is_flag=True, help="Overwrite the output folder.")
@click.option("--only-audio", is_flag=True, help="Produce video with only audio, the video will be black.")
def main(
    wav_audio_folder: str,
    raw_audio_folder: str,
    output_folder: str,
    video_segments_folder: str,
    raw_videos_folder: str,
    overwrite: bool,
    only_audio: bool,
):
    if not Path(raw_videos_folder).exists():
        raise ValueError("Mount the bulk storage first")

    for cam_folder in tqdm(
        sorted(Path(video_segments_folder).iterdir()), desc="Camera"
    ):
        if not cam_folder.is_dir():
            continue

        cam_name = cam_folder.stem

        if cam_name != "cam10":
            folder_cam_name = cam_name.replace("cam", "cam0")
        else:
            folder_cam_name = cam_name

        for raw_vid_4GB_segment in [2, 3]:
            raw_video = (
                Path(raw_videos_folder)
                / folder_cam_name
                / camera_raw_to_segment[cam_name][raw_vid_4GB_segment]
            )

            # fmt: off
            cmd = [
                    "ffprobe",
                    "-hide_banner",
                    "-show_streams", 
                    "-i", str(raw_video),
                ]
            # fmt: on
            res = subprocess.run(cmd, capture_output=True)
            video_timecode = res.stdout.decode("utf-8").rstrip()
            # Find TAG:timecode=<VAL> in the string
            video_timecode = video_timecode.split("TAG:timecode=")[1].split("\n")[0]
            video_date = [2019, 10, 24]
            video_time = video_timecode.split(":")
            video_time[-1] = round(1000000 * (int(video_time[-1]) / 59.94))
            video_time = list(map(int, video_time))

            video_timecode = datetime(*video_date, *video_time) + timedelta(hours=2)

            camera_raw_timecodes[cam_name][raw_vid_4GB_segment] = video_timecode

        for video_segment in tqdm(
            sorted(cam_folder.iterdir()), desc="Video segment", leave=False
        ):
            if video_segment.is_dir() or video_segment.suffix != ".mp4":
                continue
            video_name = video_segment.stem

            vid_num = int(video_name.split("-")[0][3:])
            seg_num = int(video_name.split("-")[1][3:])

            # The start of the segment is the the start of the video plus the delta for the segment
            segment_start = (
                camera_raw_timecodes[cam_name][vid_num]
                + vid_deltas[f"vid{vid_num}_seg{seg_num}"]
            )

            for audio_file in tqdm(
                sorted(Path(wav_audio_folder).iterdir()),
                desc="Participant",
                leave=False,
            ):
                if audio_file.suffix != ".wav":
                    continue

                # All synched audio files start at the same time
                audio_timecode = datetime.fromtimestamp(1571927168.657)

                audio_start_time = segment_start - audio_timecode
                audio_start_time_str = "0" + str(audio_start_time)

                if vid_num == 2 and seg_num == 9:
                    # The last segment of the video 2 is shorter
                    audio_length = "00:01:38.070000"
                else:
                    audio_length = "00:02:00"

                output_audio_path = (
                    Path(output_folder)
                    / "audio"
                    / cam_name
                    / f"{video_name}-participant{audio_file.stem}.wav"
                )
                output_audio_path.parent.mkdir(parents=True, exist_ok=True)

                # Crop the audio to the same length and offset as the video
                # fmt: off
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", str(audio_file),
                    "-vcodec", "copy",
                    "-acodec", "copy",
                    "-copyinkf",
                    "-ss", audio_start_time_str,
                    "-t", audio_length,
                    str(output_audio_path),
                ]
                if overwrite:
                    cmd.append("-y")
                # fmt: on
                subprocess.run(cmd)

                output_video_path = (
                    Path(output_folder)
                    / "video"
                    / cam_name
                    / f"{video_name}-participant{audio_file.stem}.mp4"
                )
                output_video_path.parent.mkdir(parents=True, exist_ok=True)

                # Combine the video and audio, this is re-encoding the audio because remuxing directly does not work
                if only_audio:
                    # fmt: off
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-f", "lavfi",
                        "-i", "color=c=black:s=256x144:r=59.94", # 144p resolution for faster processing and 59.94 fps, same as the video
                        "-i", str(output_audio_path),
                        "-c:v", "libx264",
                        "-c:a", "aac", # Audio codec pcm_s16le is not supported in mp4, so re-encode to aac
                        "-shortest", # The black filter is infinite length, stop when the audio stops
                        str(output_video_path),
                    ]
                else:
                    # fmt: off
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "panic",
                        "-i", str(video_segment),
                        "-i", str(output_audio_path),
                        "-c:v", "copy",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        str(output_video_path),
                    ]
                    # fmt: on
                if overwrite:
                    cmd.append("-y")
                subprocess.run(cmd)


if __name__ == "__main__":
    # Arguments that this script was called with
    # "--audio-folder", "/data/conflab/data_raw/audio/synced/",
    # "--video-segments-folder", "/data/conflab/data_processed/cameras/video_segments/",
    # "--raw-videos-folder", "/data/conflab/data_raw/cameras/",
    # "--output-folder", "/data/conflab/covfee-annotations/speak-laughter/",
    # "--overwrite",
    main()
