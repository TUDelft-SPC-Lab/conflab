from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import scipy.interpolate

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


def _on_trackback_change(cap: cv2.VideoCapture, frame_index: int):
    # Set the video capture to the position in the trackback
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def add_annotation_square_and_progress(
    frame: np.ndarray,
    annotation_square: np.ndarray,
    frame_index: int,
    total_frames: int,
    annotations_img_height: int,
) -> np.ndarray:
    """
    Add two square at the bottom of the video showing the annotation values and the current index.
    Green shows an annotation with a value of 1, black with a value of 0 and red a missing annotation (nan value)
    """
    size_with_annotation = list(frame.shape)
    size_with_annotation[0] += annotations_img_height * 2
    frame_with_annotations = np.zeros(size_with_annotation, dtype=frame.dtype)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    frame_with_annotations[:frame_height] = frame

    frame_with_annotations[-annotations_img_height * 2 : -annotations_img_height] = (
        annotation_square
    )

    # Transform with frame_index to the width of the image
    image_width_frame_index = (frame_index / total_frames) * (frame_width - 1)
    image_width_frame_index = int(
        np.clip(image_width_frame_index, 0, (frame_width - 1))
    )

    cv2.rectangle(
        frame_with_annotations,
        (image_width_frame_index, frame_height + annotations_img_height),
        (image_width_frame_index, frame_height + annotations_img_height * 2),
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


def main():
    segment_name = "vid2-seg8"

    video_path = Path(
        f"/home/era/code/NEON/data/conflab/video_segments/cam2/{segment_name}-scaled-denoised.mp4"
    )
    annotations = Path(
        f"/home/era/Downloads/exported_csv_files/Speaking/With_Audio/{segment_name.replace('-','_')}_ann1.csv"
    )
    participant = "2"

    # Read the annotations
    annotations_df = pd.read_csv(annotations)
    data_for_participant = annotations_df[participant].array

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Nearest neighbour interpolate data_for_participant to the frame width
    # Create a function defined as y = f(x)
    interpolator_participant = scipy.interpolate.interp1d(
        x=np.arange(0, len(data_for_participant)),
        y=data_for_participant,
        kind="nearest",  # Nearest neighbour interpolation
        fill_value=np.nan,  # Extrapolate with nans
        bounds_error=False,  # Allow extrapolation
        assume_sorted=True,  # Data is sorted on x
    )
    data_for_participant_in_frame_width = None

    window_name = f"Participant {participant} {annotations.parent.parent.name} {annotations.parent.name}"
    cv2.namedWindow(window_name)

    def on_trackback_change(frame_index: int):
        _on_trackback_change(cap, frame_index)

    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackback_change)
    annotations_img_height = 20 # Size in pixels of the annotation square below the frames

    last_frame, annotation_square = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            # If end of video, just keep playing that one until the user closes the window or presses q
            frame = last_frame

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if data_for_participant_in_frame_width is None:
            frame_width = frame.shape[1]
            data_for_participant_in_frame_width = interpolator_participant(
                np.linspace(0, len(data_for_participant), frame_width)
            )
            annotation_square = create_annotation_square(
                annotations=data_for_participant_in_frame_width,
                annotations_img_height=annotations_img_height,
                frame_width=frame_width,
                dtype=frame.dtype,
            )

        last_frame = frame
        frame_with_annotations = add_annotation_square_and_progress(
            frame,
            annotation_square,
            frame_index=frame_index,
            total_frames=total_frames,
            annotations_img_height=annotations_img_height,
        )

        cv2.imshow(window_name, frame_with_annotations)
        cv2.setTrackbarPos("Frame", window_name, frame_index)

        # Wait the 1 millisecond for a keypress before moving to the next frame
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
