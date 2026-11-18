"""
Pydantic models for Gymnasium Space objects, enabling native Hearth networking support.

Space objects are converted to Pydantic models that Hearth can serialize/deserialize natively.
Observations and actions use primitives and numpy arrays already supported by Hearth.

Each model carries every constructor parameter its Space retains, so that a round trip is
lossless: a field these models do not declare is dropped silently, and the agent then plays the
rest of the episode against a subtly different action space than the one the gateway sent.

`seed` is the one exception, because a Space does not keep it. `Space.__init__` stores the
argument only when it is already a Generator; an integer seed is used to build the RNG lazily and
then forgotten, and there is no attribute to read it back from. Sampling is the agent's business
anyway, so an unseeded space on the far side costs nothing.
"""

# future annotations allows us to have a bidirectional relationships more easily:
# composite models contain types within SpaceModelUnion, which itself contains the composite models.
from __future__ import annotations

import abc
from typing import Annotated, Any, Literal

import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, BeforeValidator, model_validator

# Shim for Gymnasium < 1.0.0 which aicomp-sdk requires
if not hasattr(spaces, 'OneOf'):
    spaces.OneOf = type('OneOf', (object,), {})

from remote_game_drivers.core.base_classes import BaseSerializer

# Fundamental spaces:
# https://gymnasium.farama.org/api/spaces/fundamental/


def _convert_space_to_model(v: Any) -> Any:
    # Used to handle composite spaces
    if isinstance(v, spaces.Space):
        return SPACE_TO_MODEL[type(v)].model_validate(v, from_attributes=True)
    return v


def _to_list(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, frozenset):
        return list(v)
    return v


def _to_dtype_name(v: Any) -> Any:
    if isinstance(v, (np.dtype, type)):
        return str(np.dtype(v))
    return str(v)


# Pydantic has limited support for numpy types and nested lists, so the array-valued fields cross
# the wire as plain lists. Declaring the coercion on the type rather than per model keeps every
# space that holds an array using the same one; `nvec` and `start` differing was how MultiDiscrete
# ended up rejecting a multi-dimensional nvec.
ListLike = Annotated[list[Any], BeforeValidator(_to_list)]
ScalarOrListLike = Annotated[float | int | list[Any], BeforeValidator(_to_list)]
DtypeName = Annotated[str, BeforeValidator(_to_dtype_name)]


class SpaceModel(abc.ABC, BaseModel):
    """Abstract base class for all space models."""

    @abc.abstractmethod
    def to_space(self) -> spaces.Space[Any]:
        raise NotImplementedError


class BoxModel(SpaceModel):
    type: Literal['Box'] = 'Box'
    low: ScalarOrListLike  # SupportsFloat | NDArray[Any]
    high: ScalarOrListLike  # SupportsFloat | NDArray[Any]
    shape: list[int]
    dtype: DtypeName

    def to_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array(self.low, dtype=np.dtype(self.dtype)),
            high=np.array(self.high, dtype=np.dtype(self.dtype)),
            shape=tuple(self.shape),
            dtype=np.dtype(self.dtype),  # type: ignore
        )


class DiscreteModel(SpaceModel):
    type: Literal['Discrete'] = 'Discrete'
    n: int
    start: int = 0

    def to_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.n, start=self.start)


class MultiBinaryModel(SpaceModel):
    type: Literal['MultiBinary'] = 'MultiBinary'
    n: int | list[int]

    def to_space(self) -> spaces.MultiBinary:
        return spaces.MultiBinary(self.n)


class MultiDiscreteModel(SpaceModel):
    type: Literal['MultiDiscrete'] = 'MultiDiscrete'
    nvec: ListLike
    start: ListLike | None = None
    dtype: DtypeName = 'int64'

    def to_space(self) -> spaces.MultiDiscrete:
        dtype = np.dtype(self.dtype)
        return spaces.MultiDiscrete(
            nvec=np.array(self.nvec, dtype=dtype),
            dtype=dtype,  # ty: ignore[invalid-argument-type]
            start=np.array(self.start, dtype=dtype) if self.start is not None else None,
        )


class TextModel(SpaceModel):
    type: Literal['Text'] = 'Text'
    max_length: int
    min_length: int = 1
    charset: Annotated[list[str] | str, BeforeValidator(_to_list)]

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.Text):
            return {
                'type': 'Text',
                'max_length': data.max_length,
                'min_length': data.min_length,
                'charset': data._char_set,
            }
        return data

    def to_space(self) -> spaces.Text:
        return spaces.Text(
            max_length=self.max_length,
            min_length=self.min_length,
            charset=frozenset(self.charset) if isinstance(self.charset, list) else self.charset,
        )


# Composite spaces:
# https://gymnasium.farama.org/api/spaces/composite/


class DictModel(SpaceModel):
    type: Literal['Dict'] = 'Dict'
    spaces: dict[str, SpaceModelUnion]

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.Dict):
            return {'type': 'Dict', 'spaces': {k: _convert_space_to_model(s) for k, s in data.spaces.items()}}
        return data

    def to_space(self) -> spaces.Dict:
        return spaces.Dict({k: s.to_space() for k, s in self.spaces.items()})


class TupleModel(SpaceModel):
    type: Literal['Tuple'] = 'Tuple'
    spaces: list[SpaceModelUnion]

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.Tuple):
            return {'type': 'Tuple', 'spaces': [_convert_space_to_model(s) for s in data.spaces]}
        return data

    def to_space(self) -> spaces.Tuple:
        return spaces.Tuple([s.to_space() for s in self.spaces])


class SequenceModel(SpaceModel):
    type: Literal['Sequence'] = 'Sequence'
    feature_space: SpaceModelUnion
    stack: bool = False

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.Sequence):
            return {'type': 'Sequence', 'feature_space': _convert_space_to_model(data.feature_space), 'stack': data.stack}
        return data

    def to_space(self) -> spaces.Sequence:
        return spaces.Sequence(self.feature_space.to_space(), stack=self.stack)


class GraphModel(SpaceModel):
    type: Literal['Graph'] = 'Graph'
    node_space: BoxModel | DiscreteModel
    edge_space: BoxModel | DiscreteModel | None = None

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.Graph):
            return {
                'type': 'Graph',
                'node_space': _convert_space_to_model(data.node_space),
                'edge_space': _convert_space_to_model(data.edge_space) if data.edge_space else None,
            }
        return data

    def to_space(self) -> spaces.Graph:
        node_space = self.node_space.to_space()
        edge_space = self.edge_space.to_space() if self.edge_space else None
        return spaces.Graph(node_space=node_space, edge_space=edge_space)


class OneOfModel(SpaceModel):
    type: Literal['OneOf'] = 'OneOf'
    spaces: list[SpaceModelUnion]

    @model_validator(mode='before')
    @classmethod
    def convert_from_space(cls, data: Any) -> Any:
        if isinstance(data, spaces.OneOf):
            return {'type': 'OneOf', 'spaces': [_convert_space_to_model(s) for s in data.spaces]}
        return data

    def to_space(self) -> spaces.OneOf:
        return spaces.OneOf([s.to_space() for s in self.spaces])


SpaceModelUnion = (
    BoxModel | DiscreteModel | MultiBinaryModel | MultiDiscreteModel | TextModel | DictModel | TupleModel | SequenceModel | GraphModel | OneOfModel
)

SPACE_TO_MODEL = {
    spaces.Box: BoxModel,
    spaces.Discrete: DiscreteModel,
    spaces.MultiBinary: MultiBinaryModel,
    spaces.MultiDiscrete: MultiDiscreteModel,
    spaces.Text: TextModel,
    spaces.Dict: DictModel,
    spaces.Tuple: TupleModel,
    spaces.Sequence: SequenceModel,
    spaces.Graph: GraphModel,
    spaces.OneOf: OneOfModel,
}


class GymSerializer(BaseSerializer[spaces.Space[Any]]):
    """Serialization and deserialization support for all Gymnasium Space object types.

    Leverages Hearth's native networking support for Pydantic models. Space objects are
    converted to Pydantic models and passed through directly to the networking layer.
    Observation values use types already supported by default (int, str, np.array, dict, etc).
    """

    def serialize(self, data: Any) -> Any:
        """Serialize game data for networking.

        Converts Gymnasium Space objects to Pydantic models (natively supported by Hearth).
        Other data types (observations, actions) are passed through as-is since they use
        primitives and numpy arrays that Hearth already supports.

        Returns:
            Pydantic model for Space objects, or the original data for other types.
        """
        if isinstance(data, spaces.Space):
            # Convert Space to Pydantic model - Hearth will handle the rest
            pydantic_model_class = SPACE_TO_MODEL[type(data)]
            model = pydantic_model_class.model_validate(data, from_attributes=True)
            return model
        else:
            # Observations and actions use primitives/numpy arrays - pass through
            return data

    def deserialize(self, data: Any) -> Any:
        """Deserialize data from networking format.

        Converts Pydantic Space models back to Gymnasium Space objects.
        All other types (observations, actions) are passed through as-is.

        Args:
            data: Either a Pydantic SpaceModel or a native type (primitives, numpy arrays, etc).

        Returns:
            Gymnasium Space object or observation value.
        """
        # If it's a Pydantic SpaceModel, convert to Space
        if isinstance(data, SpaceModel):
            return data.to_space()

        # Otherwise, it's already a native type (observation/action) - pass through
        return data
