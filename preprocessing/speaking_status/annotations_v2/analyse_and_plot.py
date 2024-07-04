from pathlib import Path
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import threading
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from json_file_data_model import (
    HITData,
    load_json_data,
)
from copy import deepcopy
import sys


def savefig(save_dir: Path, fig: Figure, ax: Axes, i: int, end_idx: int):
    ax.legend()

    # Set the size to 16:9 aspect ratio
    fig.set_size_inches(16, 9)

    fig.tight_layout()
    fig.savefig(save_dir / f"plot_{i}_{end_idx}.png", dpi=100)


def plot_annotations(
    participant_id: int,
    prolific_ids: np.ndarray,
    all_annotation_data: np.ndarray,
    multithreaded=True,
):
    thread_list = []

    save_dir = Path(f"plots/participant_{participant_id}")
    save_dir.mkdir(exist_ok=True, parents=True)

    interval_size = 1500
    max_size = all_annotation_data.shape[1]
    num_participants = all_annotation_data.shape[0]
    for i in range(0, max_size, interval_size):
        # fig.clf()

        fig, ax = plt.subplots()
        assert isinstance(ax, Axes)

        end_idx = min(i + interval_size, max_size)
        x_idx = np.arange(i, end_idx)

        # Do a bar plot instead where the top half bar is annotator 1 and the bottom half bar is annotator 2
        for j, annotation_data in enumerate(all_annotation_data):
            ax.bar(
                x_idx,
                annotation_data[i:end_idx] * 0.5,
                label=f"Annotator {prolific_ids[j]}",
                bottom=np.zeros_like(x_idx) + ((num_participants - j - 1) * 0.5),
            )

        ax.set_ylim(0, len(all_annotation_data) * 0.5)

        # Save at 1090x1080 resolution
        # fig.savefig(save_dir / f'plot_{i}_{end_idx}.png', dpi=100)
        if multithreaded:
            thread = threading.Thread(
                target=savefig, args=(save_dir, fig, ax, i, end_idx)
            )
            thread.start()
            thread_list.append(thread)
        else:
            savefig(save_dir, fig, ax, i, end_idx)

        plt.close(fig)

    for thread in tqdm(thread_list, desc="thread"):
        thread.join()


def majority_vote(all_annotation_data: np.ndarray, valid_ids: list[int]) -> np.ndarray:
    majority_vote = []
    if len(all_annotation_data) == 0:
        return np.empty(0)

    for i in range(all_annotation_data.shape[1]):
        if i in valid_ids:
            majority_vote.append(all_annotation_data[i, :])
    majority_vote = np.array(majority_vote)
    majority_vote = np.sum(majority_vote, axis=0)
    result = np.zeros_like(majority_vote)
    result[majority_vote > len(valid_ids) / 2] = 1
    return result


@dataclass
class AggrementData:
    data: list[float]
    mean: Optional[float] = None
    std: Optional[float] = None
    num_marked_missing: int = 0


def get_all_data_for_participant(
    participant_id: int,
    all_annotation_data: list[HITData],
    total_agreement: dict[str, AggrementData],
) -> tuple[np.ndarray, np.ndarray, dict[str, AggrementData]]:
    total_agreement = deepcopy(total_agreement)
    participant_annotations = []
    prolific_ids = []
    min_len = np.inf
    for all_annotation_data_i in all_annotation_data:
        for node in all_annotation_data_i.nodes.values():
            for response in node.responses:
                participant_annotation = [
                    annotations
                    for annotations in response.annotations.values()
                    if annotations.participant == f"Participant_{participant_id}"
                ]
                assert len(participant_annotation) == 1
                participant_annotation = participant_annotation[0]
                if len(participant_annotation.data) > 1:
                    participant_annotations.append(participant_annotation.data)
                    prolific_ids.append(response.journeys[0].prolific_id)
                    min_len = min(min_len, len(participant_annotation.data))
                elif len(participant_annotation.data) == 1:
                    if response.journeys[0].prolific_id not in total_agreement:
                        total_agreement[response.journeys[0].prolific_id] = (
                            AggrementData([])
                        )
                    total_agreement[
                        response.journeys[0].prolific_id
                    ].num_marked_missing += 1
                    print(
                        f"participant {participant_id} marked missing by {response.journeys[0].prolific_id}"
                    )
                else:
                    pass
                    # raise ValueError("Data error")

    participant_annotations = [
        participant_annotation[:min_len]
        for participant_annotation in participant_annotations
    ]
    return np.array(prolific_ids), np.array(participant_annotations), total_agreement


from collections import defaultdict
from tqdm import tqdm

def main(
    database_files: list[Path],
    num_annotated_participants: int,
    do_plots: bool = True,
    multithreaded: bool = True,
    min_aggrement: float = 0.8,
):
    print("\n\n")
    
    video_lengths = defaultdict(lambda: [])
    for database_file in tqdm(database_files):
        data = load_json_data(database_file, num_annotated_participants)
        vid_seg = data.global_unique_id[-9:]
        for node in data.nodes.values():
            for response in node.responses:
                annotator_all = []
                for annotation in response.annotations.values():
                    annotation_length = len(annotation.data)
                    if annotation_length != 1:
                        annotator_all.append(annotation_length)
                annotator_all = np.array(annotator_all)
                # if not np.all(annotator_all == annotator_all[0]):
                #     print(f"different annotation sizes for the same annotator {annotator_all}")
                video_lengths[vid_seg].extend(annotator_all.tolist())                    
        # print(f"Done {database_file}")
    print("done")
    shorter = ["vid2-seg9", "vid3-seg9", "vid4-seg9"]
    save_folder = Path("./length_diffs")
    save_folder.mkdir(exist_ok=True, parents=True)
    total = 0
    num_diff = 0
    for video_segment, lengths in video_lengths.items():
        lengths_arr = np.array(lengths)
        if video_segment in shorter:
            lengths_diff = lengths_arr - 5884
        else:
            lengths_diff = lengths_arr - 7200

        unique, counts = np.unique(lengths_diff, return_counts=True)
        total += len(lengths_diff)
        num_diff += np.sum(lengths_diff != 0)
        
        plt.figure()
        plt.bar(range(len(unique)), counts)
        plt.xticks(range(len(unique)), unique)
        plt.title(video_segment)
        plt.xlabel("Distance to reference length")
        plt.ylabel("Number of annotations")
        plt.savefig(save_folder / f"{video_segment}.png")

    print(num_diff / total)
    print("done")


if __name__ == "__main__":
    # 1. Download results.zip from the the covfee admin panel, assuming it's on ~/home/Downloads
    # 2. Run extract extract_json_files_from_results_zip_downloaded_from_covfee_admin.py
    # 3. Run this script
    # 4. Check the output in analysis_output.txt

    # Change this to None to print to stdout
    OUTPUT_FILE_TO_REDIRECT_PRINTS: Optional[Path] = None 
    # (
    #     Path(__file__).parent / "analysis_output.txt"
    # )
    JSON_FILES_TO_PROCESS_FILTER: Optional[str] = (
        None  # Example: "Speaking_With_Audio_v01"
    )

    if OUTPUT_FILE_TO_REDIRECT_PRINTS is not None:
        output_file_handle = open(OUTPUT_FILE_TO_REDIRECT_PRINTS, "w")
        sys.stdout = output_file_handle

    json_files = sorted((Path(__file__).parent / "json_files").glob("*.json"))

    main(
        json_files,
        num_annotated_participants=48,
        do_plots=False,
        min_aggrement=0.7,
    )
