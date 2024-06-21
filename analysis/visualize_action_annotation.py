from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import scipy.interpolate
from typing import Optional

grandparent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(grandparent_dir))

# from constants import (  # noqa: E402
#     camera_id_to_dict_of_video_index_to_raw_video_file_basename,
#     CAMERAS_OF_INTEREST,
#     TIMECODE_FOR_ALL_SYNCED_PARTICIPANT_AUDIO_WAV_FILES,
#     RAW_VIDEOS_FOLDER_IN_STAFF_BULK,
#     VIDEO_SEGMENTS_FOLDER_IN_STAFF_BULK,
#     VIDEO_SEGMENTS_FOLDER_IN_LOCAL,
#     SYNCED_AUDIO_FOLDER_IN_LOCAL,
#     SYNCED_AUDIO_FOLDER_IN_STAFF_BULK,
#     NUMBER_OF_PARTICIPANTS_WITH_WAV_FILE,
#     PARTICIPANTS_IDS_TO_IGNORE,
#     AUDIO_SEGMENTS_PER_PARTICIPANT_FOLDER_FOR_ALL_CAMS_IN_LOCAL,
#     CAM4_VID2_START_TIMECODE,
#     CAM4_VID3_START_TIMECODE,
#     check_if_staff_bulk_is_mounted,
# )


def _on_trackbar_change(cap: cv2.VideoCapture, frame_index: int):
    # Set the video capture to the position in the trackbar
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def add_annotation_square_and_progress(
    frame: np.ndarray,
    annotation_square: np.ndarray,
    frame_index: int,
    total_frames: int,
    annotations_img_height: int,
    progress_img_height: int,
) -> np.ndarray:
    """
    Add two square at the bottom of the video showing the annotation values and the current index.
    Green shows an annotation with a value of 1, black with a value of 0 and red a missing annotation (nan value)
    """
    size_with_annotation = list(frame.shape)
    size_with_annotation[0] += annotations_img_height + progress_img_height
    frame_with_annotations = np.zeros(size_with_annotation, dtype=frame.dtype)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    frame_with_annotations[:frame_height] = frame

    frame_with_annotations[
        -annotations_img_height - progress_img_height : -progress_img_height
    ] = annotation_square

    # Transform with frame_index to the width of the image
    image_width_frame_index = (frame_index / total_frames) * (frame_width - 1)
    image_width_frame_index = int(
        np.clip(image_width_frame_index, 0, (frame_width - 1))
    )

    cv2.rectangle(
        frame_with_annotations,
        (image_width_frame_index, frame_height + annotations_img_height),
        (
            image_width_frame_index,
            frame_height + annotations_img_height + progress_img_height,
        ),
        color=(255, 0, 0),
    )

    return frame_with_annotations


def create_annotation_square(
    annotations: np.ndarray,
    annotations_img_height: int,
    frame_width: int,
    dtype: np.dtype,
) -> np.ndarray:
    black_square = np.zeros((annotations_img_height, frame_width, 3), dtype=dtype)
    green_square = np.zeros(
        (annotations_img_height, frame_width, 3), dtype=dtype
    ) + np.array([0, 255, 0], dtype=dtype)
    red_square = np.zeros(
        (annotations_img_height, frame_width, 3), dtype=dtype
    ) + np.array([0, 0, 255], dtype=dtype)

    annotation_square = (
        black_square * (annotations == 0)[np.newaxis, :, np.newaxis]
        + green_square * (annotations == 1)[np.newaxis, :, np.newaxis]
        + red_square * (np.isnan(annotations))[np.newaxis, :, np.newaxis]
    )

    return annotation_square


def create_viewing_info(
    viewing_info: str, viewing_info_height: int, frame_width: int, dtype: np.dtype
):
    viewing_info_square = np.zeros((viewing_info_height, frame_width, 3), dtype=dtype)
    cv2.putText(
        viewing_info_square,
        viewing_info,
        (0, viewing_info_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(255, 255, 255),
        lineType=2,
    )
    return viewing_info_square


def get_user_input() -> str:
    user_input, user_key = "", None
    while user_key != 13:  # 13 is the Enter key
        user_key = cv2.waitKey()
        if user_key != 13:
            user_input += chr(user_key)
    return user_input


ACTIONS = {"s": "Speaking", "l": "Laughing", "u": "UsingMobilePhone", "d": "Drinking"}
MODE = {"n": "No_Audio", "w": "With_Audio", "o": "Only_Audio"}


def main():
    participant: int = 2
    window_created: bool = False
    vid_number: int = 2
    seg_number: int = 8
    cam_number: int = 6
    action: str = "Speaking"
    mode: str = "With_Audio"
    while True:
        segment_name = f"vid{vid_number}-seg{seg_number}"
        annotators = [1, 2, 3]

        # Size in pixels of the annotation square below the frames
        annotations_img_height = 20

        video_path = Path(
            f"/home/era/code/NEON/data/conflab/video_segments/cam{cam_number}/{segment_name}-scaled-denoised.mp4"
        )

        annotation_paths: list[Path] = []
        for i in annotators:
            annotation_paths.append(
                Path(
                    f"/home/era/Downloads/exported_csv_files/{action}/{mode}/{segment_name.replace('-','_')}_ann{i}.csv"
                )
            )

        # Read the annotations
        annotations_dfs = [
            pd.read_csv(annotation_path) for annotation_path in annotation_paths
        ]
        data_for_participant = [
            annotations_df[str(participant)].array for annotations_df in annotations_dfs
        ]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Nearest neighbour interpolate data_for_participant to the frame width
        # Create a function defined as y = f(x)
        interpolators_participant: list[scipy.interpolate.interp1d] = []
        for data_for_participant_i in data_for_participant:
            interpolators_participant.append(
                scipy.interpolate.interp1d(
                    x=np.arange(0, len(data_for_participant_i)),
                    y=data_for_participant_i,
                    kind="nearest",  # Nearest neighbour interpolation
                    fill_value=np.nan,  # Extrapolate with nans
                    bounds_error=False,  # Allow extrapolation
                    assume_sorted=True,  # Data is sorted on x
                )
            )
        data_for_participant_in_frame_width = None

        window_name = "Annotation viewer"
        if not window_created:
            cv2.namedWindow(window_name)

            def on_trackbar_change(frame_index: int):
                _on_trackbar_change(cap, frame_index)

            cv2.createTrackbar(
                "Frame", window_name, 0, total_frames - 1, on_trackbar_change
            )
            window_created = True

        viewing_info_str = f"Participant: {participant}, action: {action}, mode: {mode}, cam{cam_number} vid{vid_number}-seg{seg_number}"

        last_frame, annotation_square, viewing_info = None, None, None
        while True:
            ret, frame = cap.read()
            if not ret:
                # If end of video, just keep playing that one until the user quits or picks a different set of parameters
                frame = last_frame

            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if annotation_square is None:
                frame_width = frame.shape[1]
                annotation_squares = []
                for interpolators_participant, data_for_participant_i in zip(
                    interpolators_participant, data_for_participant
                ):
                    # Note that some annotations are longer than the video and this ignores all data that
                    # comes after the end of the video.
                    data_for_participant_in_frame_width = interpolators_participant(
                        np.linspace(0, total_frames - 1, frame_width)
                    )
                    annotation_squares.append(
                        create_annotation_square(
                            annotations=data_for_participant_in_frame_width,
                            annotations_img_height=annotations_img_height,
                            frame_width=frame_width,
                            dtype=frame.dtype,
                        )
                    )
                viewing_info = create_viewing_info(
                    viewing_info_str, annotations_img_height, frame_width, frame.dtype
                )
                annotation_square = np.concatenate(
                    annotation_squares + [viewing_info], axis=0
                )

            last_frame = frame
            frame_with_annotations = add_annotation_square_and_progress(
                frame,
                annotation_square,
                frame_index=frame_index,
                total_frames=total_frames,
                annotations_img_height=annotations_img_height * (len(annotators) + 1),
                progress_img_height=annotations_img_height,
            )

            cv2.imshow(window_name, frame_with_annotations)
            cv2.setTrackbarPos("Frame", window_name, frame_index)

            # Wait 1 millisecond for a keypress before moving to the next frame
            key = cv2.waitKey(1)
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord("p"):
                participant = int(get_user_input())
                break
            elif key == ord("a"):
                action = ACTIONS[get_user_input()]
                break
            elif key == ord("m"):
                mode = MODE[get_user_input()]
                break
            elif key == ord("c"):
                cam_number = int(get_user_input())
                break
            elif key == ord("v"):
                vid_number = int(get_user_input())
                seg_number = int(get_user_input())
                break
        cap.release()


if __name__ == "__main__":
    main()
