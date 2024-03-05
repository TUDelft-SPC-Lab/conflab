from pathlib import Path
from subprocess import run


def ask_yes(question: str) -> bool:
    return input(f"{question} (y/n): ").lower() == "y"


def ask_overwrite(file: Path) -> bool:
    return ask_yes(f"File {file} already exists. Overwrite?")


def main(allow_overwrite: bool, show_existing_points: bool, conflab_mm_path: Path, output_base_path: Path, extrinsic_script: Path):
    base_path = conflab_mm_path / "processed/camera-calibration/"

    intrinsics_paths = {}
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            intrinsics_paths[folder.name] = folder / "intrinsic.json"

    elevated_folder = conflab_mm_path / Path("raw/video/elevated")
    overhead_folder = conflab_mm_path / Path("raw/video/overhead")

    frame_time = "00:00:00.000"

    for video_folders in [elevated_folder, overhead_folder]:
        for camera_folder in sorted(video_folders.iterdir()):
            if not camera_folder.is_dir():
                continue
            for video_file in sorted(camera_folder.iterdir()):
                if video_file.suffix.lower() == ".mp4":
                    print(f"Processing camera: {camera_folder.name}")
                    cam_output_path = output_base_path / camera_folder.name
                    cam_output_path.mkdir(parents=True, exist_ok=True)
                    image_output_path = cam_output_path / "frame_000.jpg"
                    if image_output_path.exists() and allow_overwrite and ask_overwrite(image_output_path):
                        image_output_path.unlink()

                    if not image_output_path.exists():
                        # Extract one frame at time frame_time
                        # fmt: off
                        run(
                            [
                                "ffmpeg",
                                "-loglevel", "quiet",
                                "-hide_banner",
                                "-ss", frame_time,
                                "-i", str(video_file),
                                "-vframes", "1",
                                str(image_output_path),
                            ]
                        )
                        # fmt: on

                    extrinsic_output_path = cam_output_path / "extrinsic.json"
                    calibration_points_path = cam_output_path / "calibration_points.json"

                    camera_name = camera_folder.name.replace("cam0", "cam")
                    cmd = [
                        "python3",
                        str(extrinsic_script),
                        str(intrinsics_paths[camera_name]),
                        str(image_output_path),
                        str(extrinsic_output_path),
                    ]

                    if calibration_points_path.exists():
                        # Reuse the calibration points as input
                        existing_points_flag = ["--points", str(calibration_points_path)]

                        if allow_overwrite and ask_overwrite(calibration_points_path):
                            # Overwrite the output paths
                            cmd.extend(["--output_points", str(calibration_points_path)])
                            if not ask_yes("Use existing points as start point?"):
                                existing_points_flag = []
                        elif not show_existing_points:
                            print(f"Camera {camera_folder.name} already calibrated. Skipping.")
                            break

                        cmd.extend(existing_points_flag)
                    else:
                        # Create new calibration points
                        cmd.extend(["--output_points", str(calibration_points_path)])

                    run(cmd)
                    # Exit after the first video for this camera
                    break


if __name__ == "__main__":
    # The path to where staff-bulk/ewi/insy/SPCDataSets/conflab-mm/ is mounted or downloaded
    conflab_mm_path = ...

    # The output folder the extrinsic calibrations will be stored
    output_base_path = ...

    # The path to the local copy of scripts/extrinsic.py of this repo https://github.com/idiap/multicamera-calibration
    extrinsic_script = ...

    assert isinstance(conflab_mm_path, Path), "Set the conflab_mm_path location"
    assert isinstance(output_base_path, Path), "Set the output_base_path location"
    assert isinstance(extrinsic_script, Path), "Set the extrinsic_script location"

    main(
        allow_overwrite=False,
        show_existing_points=True,
        conflab_mm_path=conflab_mm_path,
        output_base_path=output_base_path,
        extrinsic_script=extrinsic_script,
    )
