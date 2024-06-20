from pathlib import Path
import matplotlib

matplotlib.use("Agg")
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


def main(
    database_files: list[Path],
    num_annotated_participants: int,
    do_plots: bool = True,
    multithreaded: bool = True,
    min_aggrement: float = 0.8,
):
    print("\n\n")
    all_annotation_data: list[HITData] = []
    for database_file in database_files:
        all_annotation_data.append(
            load_json_data(database_file, num_annotated_participants)
        )
        for _ in range(3):
            print(
                "---------------------------------------------------------------------"
            )
        print(all_annotation_data[-1].global_unique_id)
        print(database_file)
        for _ in range(3):
            print(
                "---------------------------------------------------------------------"
            )

    total_agreement: dict[str, AggrementData] = {}
    for participant_id in range(1, num_annotated_participants + 1):
        print("\nParticipant", participant_id)
        if participant_id in [38, 39]:
            print("Skipping participant", participant_id)
            continue
        prolific_ids, participant_data, total_agreement = get_all_data_for_participant(
            participant_id, all_annotation_data, total_agreement
        )
        if len(participant_data) == 0:
            print("No data found for participant", participant_id)
            continue
        vote_data = majority_vote(participant_data, range(len(prolific_ids)))
        agreement = np.sum(vote_data == participant_data, axis=1) / vote_data.shape[0]
        events = np.sum((participant_data[:, 1:] - participant_data[:, :-1]) == 1, axis=1)

        for annotator, agreement_i, events_i in zip(prolific_ids, agreement, events):
            print(f"annotator {annotator}, agreement {agreement_i*100:2.2f}%, keypress events {events_i}")
            if annotator not in total_agreement:
                total_agreement[annotator] = AggrementData([])
            # Participant 2 has good visibility and overall good aggrement with the annotators, so it's good to use as
            # a reference for screening the good annotators
            if True:  # participant_id == 2:
                total_agreement[annotator].data.append(agreement_i)

        participant_data = np.concatenate((participant_data, vote_data[np.newaxis, :]))
        prolific_ids = np.concatenate((prolific_ids, ["majority_vote"]))

        if do_plots:
            plot_annotations(
                participant_id,
                prolific_ids,
                participant_data,
                multithreaded=multithreaded,
            )

    print("\nAggregate results")
    for annotator, agreement in total_agreement.items():
        agreement.mean = np.nanmean(agreement.data)
        agreement.std = np.nanstd(agreement.data)
        print(
            f"annotator {annotator}, agreement: mean {agreement.mean*100:2.2f}%, std {agreement.std*100:2.2f}%"
        )
    print(f"Total {len(total_agreement)} annotators")

    print("\nProlific annotators that marked participants as missing")
    for annotator, agreement in total_agreement.items():
        if agreement.num_marked_missing > 0:
            print(
                f"{annotator}, marked missing {agreement.num_marked_missing} participants out of {num_annotated_participants}"
            )

    max_missing_threshold = 0
    print(
        f"\nProlific list over {min_aggrement*100:2.2f}% agreement and missing <= {max_missing_threshold}:"
    )
    i = 0
    for annotator, agreement in total_agreement.items():
        if (
            agreement.mean > min_aggrement
            and agreement.num_marked_missing <= max_missing_threshold
        ):
            print(f"{annotator},")
            i += 1
    print(f"Total {i} annotators")


if __name__ == "__main__":
    # 1. Download results.zip from the the covfee admin panel, assuming it's on ~/home/Downloads
    # 2. Run extract extract_json_files_from_results_zip_downloaded_from_covfee_admin.py
    # 3. Run this script
    # 4. Check the output in analysis_output.txt

    # Change this to None to print to stdout
    OUTPUT_FILE_TO_REDIRECT_PRINTS: Optional[Path] = (
        Path(__file__).parent / "analysis_output.txt"
    )
    JSON_FILES_TO_PROCESS_FILTER: Optional[str] = (
        None  # Example: "Speaking_With_Audio_v01"
    )

    if OUTPUT_FILE_TO_REDIRECT_PRINTS is not None:
        output_file_handle = open(OUTPUT_FILE_TO_REDIRECT_PRINTS, "w")
        sys.stdout = output_file_handle

    for json_file_path in sorted((Path(__file__).parent / "json_files").glob("*.json")):
        if (
            JSON_FILES_TO_PROCESS_FILTER is not None
            and JSON_FILES_TO_PROCESS_FILTER not in json_file_path.stem
        ):
            continue
        main(
            [json_file_path],
            num_annotated_participants=48,
            do_plots=False,
            min_aggrement=0.7,
        )
