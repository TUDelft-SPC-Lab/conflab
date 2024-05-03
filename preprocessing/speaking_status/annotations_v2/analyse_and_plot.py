from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import threading
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from data_model import (
    AnnotationData,
    load_json_data,
)


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
                label=f"Annotator {prolific_ids[j][0]}",
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

    for thread in tqdm(thread_list, desc="thread"):
        thread.join()


def majority_vote(all_annotation_data: np.ndarray, valid_ids: list[int]) -> np.ndarray:
    majority_vote = []
    for i in range(all_annotation_data.shape[1]):
        if i in valid_ids:
            majority_vote.append(all_annotation_data[i, :])
    majority_vote = np.array(majority_vote)
    majority_vote = np.sum(majority_vote, axis=0)
    result = np.zeros_like(majority_vote)
    result[majority_vote > len(valid_ids) / 2] = 1
    return result


def get_all_data_for_participant(
    participant_id: int, all_annotation_data: list[AnnotationData]
):
    participant_annotations = []
    prolific_ids = []
    min_len = np.inf
    for all_annotation_data_i in all_annotation_data:
        for node in all_annotation_data_i.nodes.values():
            for response in node.responses:
                if len(response.annotations) > 0:
                    participant_annotation = response.annotations[participant_id - 1]
                    if len(participant_annotation) > 1:
                        participant_annotations.append(participant_annotation)
                        prolific_ids.append(response.prolific_id)
                        min_len = min(min_len, len(participant_annotation))
                    else:
                        print(
                            f"participant {participant_id} marked missing by {response.prolific_id}"
                        )

    participant_annotations = [
        participant_annotation[:min_len]
        for participant_annotation in participant_annotations
    ]
    return np.array(prolific_ids), np.array(participant_annotations)


@dataclass
class AggrementData:
    data: list[float]
    mean: Optional[float] = None
    std: Optional[float] = None


def main(
    database_files: list[Path],
    num_annotated_participants: int,
    do_plots: bool = True,
    multithreaded: bool = True,
    min_aggrement: float = 0.8,
):
    all_annotation_data = []
    for database_file in database_files:
        all_annotation_data.append(
            load_json_data(database_file, num_annotated_participants)
        )

    total_agreement: dict[str, AggrementData] = {}
    for participant_id in [1, 2]:  # Analyse dat for participant 1 and 2
        print("\nParticipant", participant_id)
        prolific_ids, participant_data = get_all_data_for_participant(
            participant_id, all_annotation_data
        )
        vote_data = majority_vote(participant_data, range(len(prolific_ids)))
        agreement = np.sum(vote_data == participant_data, axis=1) / vote_data.shape[0]

        for annotator, agreement_i in zip(prolific_ids, agreement):
            print(f"annotator {annotator}, agreement {agreement_i*100:2.2f}%")
            if annotator[0] not in total_agreement:
                total_agreement[annotator[0]] = AggrementData([])
            total_agreement[annotator[0]].data.append(agreement_i)

        participant_data = np.concatenate((participant_data, vote_data[np.newaxis, :]))
        prolific_ids = np.concatenate((prolific_ids, [["majority_vote"]]))

        if do_plots:
            plot_annotations(
                participant_id,
                prolific_ids,
                participant_data,
                multithreaded=multithreaded,
            )

    print("\nAggregate results")
    for annotator, agreement in total_agreement.items():
        agreement.mean = np.mean(agreement.data)
        agreement.std = np.std(agreement.data)
        print(
            f"annotator {annotator}, agreement, mean {agreement.mean*100:2.2f}%, std {agreement.std*100:2.2f}%"
        )

    print(f"\nProlific list over {min_aggrement*100:2.2f}% agreement:")
    for annotator, agreement in total_agreement.items():
        if agreement.mean > min_aggrement:
            print(f"{annotator},")


if __name__ == "__main__":
    data_files = [
        Path("/home/era/code/covfee-repos/database_pilot_v0.json"),
        Path("/home/era/code/covfee-repos/database_pilot_v1.json"),
    ]

    main(data_files, num_annotated_participants=2, do_plots=True, min_aggrement=0.8)