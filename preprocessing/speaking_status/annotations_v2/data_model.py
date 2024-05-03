from pathlib import Path
import numpy as np
from pathlib import Path
import json
import json
from pydantic import BaseModel
from typing import Optional, Annotated, Any
from pydantic_core import core_schema
from typing_extensions import Annotated
from datetime import datetime
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue

def ndarray_serialiser(obj: np.ndarray) -> list[Any]:
    return obj.tolist()


def ndarray_deserialiser(value: Any) -> np.ndarray:
    return np.array(value)


# Make np.ndarray compatible with pydantic,
# see https://docs.pydantic.dev/2.3/usage/types/custom/#handling-third-party-types
class _NDArrayPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * `Any` will be parsed as `ndarray` instances
        * `ndarray` instances will be parsed as `ndarray` instances without any changes
        * Nothing else will pass validation
        * Serialization will return a list[Any] representation of the ndarray
        """
        from_list_schema = core_schema.no_info_plain_validator_function(
            ndarray_deserialiser
        )

        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(np.ndarray),
                    from_list_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: ndarray_serialiser(instance)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # When deserialising from a json, use the custom deserialiser
        return handler(
            core_schema.no_info_plain_validator_function(ndarray_deserialiser)
        )


AnnotatedNDArray = Annotated[np.ndarray, _NDArrayPydanticAnnotation]


class State(BaseModel):
    selectedAnnotationIndex: int
    mediaPaused: Optional[bool]


class Response(BaseModel):
    created: datetime
    submitted: bool
    state: Optional[State]
    annotations: list[AnnotatedNDArray]
    prolific_id: list[str]


class Node(BaseModel):
    responses: list[Response]


class Journey(BaseModel):
    nodes: list[int]


class AnnotationData(BaseModel):
    hit_id: str
    nodes: dict[int, Node]
    journeys: list[Journey]

def remove_partial_annotations_(annotation_data: AnnotationData, num_annotated_participants: int) -> None:
    for node in annotation_data.nodes.values():
        for response in node.responses:
            if (
                len(response.annotations) > 0
                and len(response.annotations) != num_annotated_participants
            ):
                response.annotations = []
                print(f"Removing partial annotations done by {response.prolific_id}")
            assert len(response.prolific_id) == 0 or len(response.prolific_id) == 1

def load_json_data(database_file: Path, num_annotated_participants: int) -> AnnotationData:
    with open(database_file, "r") as f:
        annotation_data = AnnotationData.model_validate(json.load(f))
        remove_partial_annotations_(annotation_data, num_annotated_participants)
        return annotation_data
