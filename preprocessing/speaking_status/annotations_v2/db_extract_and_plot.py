
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import json
import threading
from tqdm import tqdm
from typing import List, Tuple

def extract_data_for_participant(participant_id: str, database_file: Path, table_name: str) -> Tuple[np.ndarray, np.ndarray]:
    # Connect to the SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Print all the tables in the database
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())
    
    cursor.execute(f"SELECT * FROM 'annotators'")
    column_names = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    id_idx = None
    prolific_id_idx = None
    for i, column_name in enumerate(column_names):
        if column_name == 'id':
            id_idx = i
        elif column_name == 'prolific_id':
            prolific_id_idx = i

    assert id_idx is not None and prolific_id_idx is not None

    annotator_id_to_prolific_pid = {}
    for row in rows:
        annotator_id_to_prolific_pid[row[id_idx]] = row[prolific_id_idx]
    
    cursor.execute(f"SELECT * FROM '{table_name}' WHERE participant = 'Participant_{participant_id}'")

    # Fetch all rows from the cursor
    rows = cursor.fetchall()
    conn.close()

    # Get the column names
    column_names = [description[0] for description in cursor.description]

    id_idx = None
    data_idx = None
    for i, column_name in enumerate(column_names):
        if column_name == 'task_id':
            id_idx = i
        elif column_name == 'data_json':
            data_idx = i

    assert id_idx is not None and data_idx is not None

    prolific_ids = []
    all_annotation_data = []
    min_length = float('inf')
    for row in rows:
        data_json = row[data_idx]
        if data_json is None:
            continue

        parsed_data = json.loads(data_json)
        if len(parsed_data) == 1:
            continue

        if row[id_idx] not in annotator_id_to_prolific_pid:
            print(f"Annotator {row[id_idx]} not found in annotators table")
            continue

        prolific_ids.append(annotator_id_to_prolific_pid[row[id_idx]])       
        all_annotation_data.append(parsed_data)

        if len(all_annotation_data[-1]) < min_length:
            min_length = len(all_annotation_data[-1])

    # Truncate all the annotation data to the minimum length
    all_annotation_data = [annotation_data[:min_length] for annotation_data in all_annotation_data]

    prolific_ids = np.array(prolific_ids)
    all_annotation_data = np.array(all_annotation_data)

    return prolific_ids, all_annotation_data

def savefig(save_dir: Path, fig: Figure, ax: Axes, i: int, end_idx: int):
    ax.legend()

    # Set the size to 16:9 aspect ratio
    fig.set_size_inches(16, 9)

    fig.tight_layout()
    fig.savefig(save_dir / f'plot_{i}_{end_idx}.png', dpi=100)


def plot_annotations(participant_id: str, prolific_ids: np.ndarray, all_annotation_data: np.ndarray, multithreaded = True):
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
            ax.bar(x_idx, annotation_data[i:end_idx] * 0.5, label=f'Annotator {prolific_ids[j]}', bottom=np.zeros_like(x_idx) + ((num_participants - j - 1) * 0.5))

        ax.set_ylim(0, len(all_annotation_data) * 0.5)

        # Save at 1090x1080 resolution
        # fig.savefig(save_dir / f'plot_{i}_{end_idx}.png', dpi=100)
        if multithreaded:
            thread = threading.Thread(target=savefig, args=(save_dir, fig, ax, i, end_idx))
            thread.start()
            thread_list.append(thread)
        else:
            savefig(save_dir, fig, ax, i, end_idx)

    for thread in tqdm(thread_list, desc="thread"):
        thread.join()   

def majority_vote(all_annotation_data: np.ndarray, valid_ids: List[int]) -> np.ndarray:
    majority_vote = []
    for i in range(all_annotation_data.shape[1]):
        if i in valid_ids:
            majority_vote.append(all_annotation_data[i, :])
    majority_vote = np.array(majority_vote)
    majority_vote = np.sum(majority_vote, axis=0)
    result = np.zeros_like(majority_vote)
    result[majority_vote > len(valid_ids) / 2] = 1
    return result

def main(table_name: str, database_files: List[Path], multithreaded: bool = True):    
    for participant_id in range(2, 3):
        prolific_ids, all_annotation_data = None, None
        for database_file in database_files:
            prolific_ids_i, all_annotation_data_i = extract_data_for_participant(participant_id, database_file, table_name)
            if prolific_ids is None:
                prolific_ids = prolific_ids_i
                all_annotation_data = all_annotation_data_i
            else:
                prolific_ids = np.concatenate((prolific_ids, prolific_ids_i))
                all_annotation_data = np.concatenate((all_annotation_data, all_annotation_data_i))


        if participant_id == 1:
            pass
        elif participant_id == 2:
            # number 0, '617150973c939363d2a8bac4', is not trustworthy
            vote_data = majority_vote(all_annotation_data, [1, 2, 3])

            agreement = np.sum(vote_data == all_annotation_data, axis=1) / vote_data.shape[0]
            for annotator, agreement_i in zip(prolific_ids, agreement):
                print(f"annotator {annotator}, agreement {agreement_i*100:2.2f}%")

            all_annotation_data = np.concatenate((all_annotation_data, vote_data[np.newaxis, :]))
            prolific_ids = np.concatenate((prolific_ids, ['majority_vote']))

        plot_annotations(participant_id, prolific_ids, all_annotation_data, multithreaded=multithreaded)



if __name__ == '__main__':    
    table_name = 'ContinuousAnnotationTask.annotations'
    database_files = [
        Path('/home/era/Downloads/database_pilot_v0.covfee.db'),
        Path('/home/era/Downloads/database_pilot_v1.covfee.db')
    ]

    main(table_name, database_files)
