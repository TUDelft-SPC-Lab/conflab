from json_file_data_model import load_json_data
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple
from tqdm import tqdm
import pandas as pd
import numpy as np

# Number of participants, mainly used as a sanity check when parsing the json files
NUMBER_OF_PARTICIPANTS_IN_CONFLAB_ANNOTATIONS: int = 48


class AnnotationMetadata(NamedTuple):
    annotation_file: str
    annotator_prolific_id: str
    prolific_study_id: str


def get_prolific_study_id_for_batches_01_and_02(
    modality: str, annotator_prolific_id: str
) -> str:
    if modality == "With_Audio": # Batch02
        return "6644c80d51295858c73da36e"  
    else: # Batch01 - split in 4 studies
        if annotator_prolific_id in [
            "63a32488e685c05e9fb5988e",
            "63469effac5117e39352c13b",
            "6634e4fcdfc27d1ee891dfe9",
        ]:
            return "6641fe0517d1d0092c3f26d9"
        elif annotator_prolific_id == "6606e9876725dc8f799372e4":
            return "6642260c2ff93f1951a24afd"
        elif annotator_prolific_id in [
            "660ff6494ebe0e0fb10fc3ab",
            "5f4e0155cf03293db269f873",
        ]:
            return "66434a4e2c2b61ddf141dd7b"
        else:
            return "663b33189a503f2f262ebadf"


def main(output_dir: Path):
    all_annotation_metadata: list[AnnotationMetadata] = []
    for json_file_path in tqdm(
        sorted((Path(__file__).parent / "json_files").glob("*.json")), "JSON Files"
    ):
        hit_data = load_json_data(
            json_file_path, NUMBER_OF_PARTICIPANTS_IN_CONFLAB_ANNOTATIONS
        )
        # Note: when conducting the study for the MINGLE project, the global_unique_id for the hit
        #       strictly followed the pattern of "CATEGORY_MODALITY_VERSION_SEGMENT".
        #       - The category, version and segment never contained underscores, so they are easy to parse out
        #       - The modality did contain underscores, but thankfully can be parsed as the remaining substring
        #       - The version was added as a preemptive measure, in case we needed to run the same hit
        #         multiple times, but it was never the case in practice, so we ignore.
        category = hit_data.global_unique_id.split("_")[0]
        modality = "_".join(hit_data.global_unique_id.split("_")[1:-2])
        segment = hit_data.global_unique_id.split("_")[-1]

        for node_count, node in enumerate(hit_data.nodes.values()):
            annotator = node_count + 1  # 1-indexed

            annotator_file_relative_path = (
                Path(category)
                / modality
                / f"{segment.replace('-','_')}_ann{annotator}.csv"
            )
            segment_annotator_file_path = output_dir / annotator_file_relative_path

            assert (
                len(node.responses) == 1
            ), f"Expected 1 response, got {len(node.responses)}"  # Should contain 1
            response = node.responses[0]
            annotation_data_per_participant = defaultdict(dict)
            max_array_length = 0
            for (
                annotation
            ) in response.annotations.values():  # 1 annotation per participant
                participant = int(annotation.participant.split("_")[-1])
                if len(annotation.data) > 1:
                    annotation_data_per_participant[participant] = (
                        annotation.data.astype(float)
                    )
                    if len(annotation.data) > max_array_length:
                        max_array_length = len(annotation.data)

            # Pad the arrays with NaN so that they are coverted to empty cells in the CSV
            for participant, data in annotation_data_per_participant.items():
                padding_length = max_array_length - len(data)
                if padding_length > 0:
                    annotation_data_per_participant[participant] = np.pad(
                        data, (0, padding_length), "constant", constant_values=np.nan
                    )

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame.from_dict(annotation_data_per_participant, orient="index")

            # Sort the DataFrame by column names (participant numbers)
            df = df.sort_index(axis=1)

            # Write the DataFrame to a CSV file
            segment_annotator_file_path.parent.mkdir(exist_ok=True, parents=True)
            df.transpose().to_csv(
                segment_annotator_file_path, index=False, float_format="%.0f"
            )

            prolific_study_id = response.journeys[0].prolific_study_id
            annotator_prolific_id = response.journeys[0].prolific_id
            if prolific_study_id is None:
                # Note: Batch01 and Batch02 were executed with a version of covfee missing
                #       the prolific_study_id field. 
                assert category == "Speaking"
                prolific_study_id = get_prolific_study_id_for_batches_01_and_02(
                    modality, annotator_prolific_id
                )
            all_annotation_metadata.append(
                AnnotationMetadata(
                    annotator_file_relative_path,
                    annotator_prolific_id,
                    prolific_study_id,
                )
            )

    # Convert all_annotation_metadata to a DataFrame
    metadata_df = pd.DataFrame(all_annotation_metadata)

    # Write the DataFrame to a CSV file
    metadata_file_path = output_dir / ".." / "annotation_metadata.csv"
    metadata_df.to_csv(metadata_file_path, index=False)


if __name__ == "__main__":
    main(Path(__file__).parent / "exported_csv_files")
