"""
Mapping between Python objects and the Payload protobuf message used by the relay.

Supports (nested) primitives plus special data science types like DataFrames and np.ndarrays.
Split out from relay.py so the wire format can be read, tested, and reviewed without the gRPC
plumbing that carries it.

Complex types -- Enums, dataclasses, and Pydantic models -- travel as their contents under one wire
case, `rebuildable_object_value`: a bare class name plus either a model's fields or an enum member's
value. Nothing on the wire names code to import. The name is only a key into the classes the
receiver's own `declared_type` names for that position, so a sender picks among the receiver's
choices but cannot add to them. See `_rebuild` and `_names_in`.

That makes the match exact in both directions, which is the property the rest of the repo defers to
this docstring for: a dict is never promoted to a class it merely fits, and a rebuildable object no
declared type accounts for is refused rather than handed over as the contents it came apart into.

The same reasoning runs below the classes. Every type has its own case in Payload, so a declared
builtin is matched against the case the sender set rather than against whatever Pydantic could
convert into it: `"5"` is not an int, a tuple is not a list, and an `np.int64` is neither -- declare
`int | np.int64` to take one. See `_match_wire_case`. It stops at the edge of a model, whose fields
are its author's schema and keep its own config, as they would under FastAPI.
"""

import dataclasses
import enum
import functools
import io
import types
import typing
from typing import Any, Union, get_args, get_origin

import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import pyarrow.ipc
import pydantic

from kaggle_evaluation.core import proto_compiler

# Compile the protobuf / gRPC stubs into a temp dir on sys.path so the gateway
# and inference server can run on different docker-python image versions without
# protobuf/grpcio version-mismatch errors. relay.py does the same; ensure_compiled is
# idempotent, and each module needs the stubs on sys.path before its own protobuf import.
proto_compiler.ensure_compiled()

import kaggle_evaluation_pb2 as kaggle_evaluation_proto  # ty: ignore[unresolved-import]


class DeserializationError(Exception):
    """A payload could not be mapped back to a Python object.

    Raised only for peer-supplied content that the receiver rejects, never for a fault on the
    receiving side. Gateways rely on that distinction to report the failure against the submission
    instead of against Kaggle; see BaseGateway.handle_server_error.
    """


_POLARS_TYPE_DENYLIST = {pl.Object, pl.Unknown}


_ENUM_VALUE_TYPES = (str, int, float, bool)


def _config(target: type) -> pydantic.ConfigDict:
    """A class's Pydantic config, wherever it keeps it: `model_config` on a model, `__pydantic_config__` on a dataclass.

    Consulted by both directions, so `extra='allow'` means the same thing being sent as being rebuilt.
    An empty config is the same answer as no config, so falsy ones can fall through to the next.
    """
    return getattr(target, 'model_config', None) or getattr(target, '__pydantic_config__', None) or pydantic.ConfigDict()


### Serialization & deserialization utils


def _pandas_to_ipc(df: pd.DataFrame) -> bytes:
    """Serialize a pandas DataFrame to uncompressed Arrow IPC bytes.

    Goes through the same pandas -> Arrow conversion the parquet path used
    (``pyarrow.Table.from_pandas`` with the index dropped), so dtypes, extension types, and
    column names round-trip identically, but skips parquet's columnar encoding. Compression is
    omitted: the relay runs over loopback between same-host containers, where it costs more CPU
    than the transfer bytes it saves.

    The index is dropped in transfer. A MultiIndex always carries information that would be
    silently lost, so reject it rather than discard it.
    """
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError(  # noqa: TRY004 -- ValueError is the documented, user-facing contract here; the input type is fine, its content is not.
            'KaggleEvaluation drops the pandas index during transfer, but a MultiIndex carries '
            'information that would be lost. Call .reset_index() to move it into columns, or '
            '.reset_index(drop=True) to discard it explicitly.'
        )
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    buffer = io.BytesIO()
    with pyarrow.ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def _ipc_to_pandas(data: bytes) -> pd.DataFrame:
    """Deserialize uncompressed Arrow IPC bytes back to a pandas DataFrame.

    Only the Arrow parse is treated as a bad payload. The pandas conversion that follows fails for
    local reasons instead - an extension type this container's pyarrow or pandas can't materialize -
    so those errors propagate unchanged rather than being blamed on the peer.
    """
    try:
        with pyarrow.ipc.open_stream(data) as reader:
            table = reader.read_all()
    except pyarrow.ArrowInvalid as e:
        raise DeserializationError(f'Payload is not a readable Arrow IPC stream: {e}') from e
    return table.to_pandas()


def _npy_to_numpy(data: bytes) -> np.ndarray:
    """Deserialize .npy bytes, rejecting anything that would require pickling to load."""
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as e:
        raise DeserializationError(f'Payload is not a readable numpy array: {e}') from e


def serialize(data: Any) -> kaggle_evaluation_proto.Payload:
    """Maps input data of one of several allow-listed types to a protobuf message to be sent over gRPC.

    Args:
        data: The input data to be mapped. Any of the types listed below are accepted.

    Returns:
        The Payload protobuf message.

    Raises:
        TypeError if data is of an unsupported type.
    """
    # Python primitives and Numpy scalars
    if isinstance(data, np.generic):
        # Numpy functions that return a single number return numpy scalars instead of python primitives.
        # In some cases this difference matters: https://numpy.org/devdocs/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
        # Ex: np.mean(1,2) yields np.float64(1.5) instead of 1.5.
        # Check for numpy scalars first since most of them also inherit from python primitives.
        # For example, `np.float64(1.5)` is an instance of `float` among many other things.
        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        assert data.shape == ()  # Additional validation that the np.generic type remains solely for scalars
        assert isinstance(data, (np.number, np.bool_))  # No support for bytes, strings, objects, etc
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_scalar_value=buffer.getvalue())
    elif isinstance(data, enum.Enum):
        # Ahead of the primitive checks: enum.StrEnum, enum.IntEnum, and any other mixin enum are
        # instances of str/int, so this branch would never be reached for them otherwise. Reaching
        # it is what applies the value check below to a mixin enum too.
        if not isinstance(data.value, _ENUM_VALUE_TYPES):
            # A member travels as its value, so a value the wire can't carry would send but never
            # arrive: a tuple-valued member comes back a list and matches no member. Refuse it here,
            # where the sender can still see which member it was.
            raise TypeError(
                f'{type(data).__name__}.{data.name} has a {type(data.value).__name__} value, which the wire format cannot carry. '
                f'Enum values must be str, int, float, or bool.'
            )
        return _serialize_rebuildable(class_name=type(data).__qualname__, value=serialize(data.value))
    elif isinstance(data, str):
        return kaggle_evaluation_proto.Payload(str_value=data)
    elif isinstance(data, bool):  # bool is a subclass of int, so check that first
        return kaggle_evaluation_proto.Payload(bool_value=data)
    elif isinstance(data, int):
        return kaggle_evaluation_proto.Payload(int_value=data)
    elif isinstance(data, float):
        return kaggle_evaluation_proto.Payload(float_value=data)
    elif data is None:
        return kaggle_evaluation_proto.Payload(none_value=True)
    # Iterables for nested types
    if isinstance(data, list):
        return kaggle_evaluation_proto.Payload(list_value=kaggle_evaluation_proto.PayloadList(payloads=map(serialize, data)))
    elif isinstance(data, tuple):
        return kaggle_evaluation_proto.Payload(tuple_value=kaggle_evaluation_proto.PayloadList(payloads=map(serialize, data)))
    elif isinstance(data, dict):
        return kaggle_evaluation_proto.Payload(dict_value=_serialize_map(data))
    # Allowlisted special types
    if isinstance(data, pd.DataFrame):
        return kaggle_evaluation_proto.Payload(pandas_dataframe_value=_pandas_to_ipc(data))
    elif isinstance(data, pl.DataFrame):
        data_types = {i.base_type() for i in data.dtypes}
        banned_types = _POLARS_TYPE_DENYLIST.intersection(data_types)
        if banned_types:
            raise TypeError(f'Unsupported Polars data type(s): {banned_types}')

        buffer = io.BytesIO()
        # Polars' native IPC writer stays in Rust, avoiding a polars -> pyarrow copy (the dominant
        # serialization cost). The stream format (vs the random-access file format) is both the
        # right fit for a sequential wire transfer and measurably cheaper. Compression is left off:
        # the gateway and inference server talk over loopback between containers on the same host,
        # where (de)compression costs more CPU than the transfer bytes it saves, and the gRPC
        # message size is uncapped.
        data.write_ipc_stream(buffer, compression='uncompressed')
        return kaggle_evaluation_proto.Payload(polars_dataframe_value=buffer.getvalue())
    elif isinstance(data, pd.Series):
        # A pd.Series can't be converted directly; wrap it in a one-column DataFrame first.
        return kaggle_evaluation_proto.Payload(pandas_series_value=_pandas_to_ipc(pd.DataFrame(data)))
    elif isinstance(data, pl.Series):
        df_payload = serialize(pl.DataFrame(data))
        return kaggle_evaluation_proto.Payload(polars_series_value=df_payload.polars_dataframe_value)
    elif isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_array_value=buffer.getvalue())
    elif isinstance(data, io.BytesIO):
        return kaggle_evaluation_proto.Payload(bytes_io_value=data.getvalue())
    elif isinstance(data, bytes):
        return kaggle_evaluation_proto.Payload(bytes_value=data)
    elif isinstance(data, frozenset):
        return kaggle_evaluation_proto.Payload(frozenset_value=kaggle_evaluation_proto.PayloadList(payloads=map(serialize, data)))
    elif isinstance(data, set):
        return kaggle_evaluation_proto.Payload(set_value=kaggle_evaluation_proto.PayloadList(payloads=map(serialize, data)))
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        # getattr per field rather than dataclasses.asdict, and dict(model) rather than model_dump:
        # both go one level deep and leave each value to serialize, so a field may hold anything the
        # wire carries. A deep dump would have to flatten a DataFrame field into JSON, which it can't.
        contents = {f.name: getattr(data, f.name) for f in dataclasses.fields(data)}
        if _config(type(data)).get('extra') == 'allow':
            # Extras live outside `fields()`, so a Pydantic dataclass that asked for them would
            # otherwise drop them here while a model keeps them -- `dict(model)` already yields both.
            # Gated on the opt-in: every dataclass has stray instance attributes in `vars`, and only
            # this config makes them part of what the class declares.
            contents |= {name: value for name, value in vars(data).items() if name not in contents}
        return _serialize_rebuildable(class_name=type(data).__qualname__, fields=_serialize_map(contents))
    elif isinstance(data, pydantic.BaseModel):
        return _serialize_rebuildable(class_name=type(data).__qualname__, fields=_serialize_map(dict(data)))

    raise TypeError(f'Type {type(data)} not supported for KaggleEvaluation.')


def _serialize_map(mapping: dict[str, Any]) -> kaggle_evaluation_proto.PayloadMap:
    serialized = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f'KaggleEvaluation only supports dicts with keys of type str, found {type(key)}.')
        serialized[key] = serialize(value)
    return kaggle_evaluation_proto.PayloadMap(payload_map=serialized)


def _serialize_rebuildable(**content: Any) -> kaggle_evaluation_proto.Payload:
    """Send something the receiver has to rebuild: `class_name` plus exactly one of `fields`/`value`."""
    return kaggle_evaluation_proto.Payload(rebuildable_object_value=kaggle_evaluation_proto.RebuildableObject(**content))


def deserialize(payload: kaggle_evaluation_proto.Payload, declared_type: Any = Any) -> Any:
    """Maps a Payload protobuf message to a value of the type the receiver declared.

    Every case but one decodes to exactly the Python object it names. A `rebuildable_object_value` names
    a class instead of carrying one, and is rebuilt only if `declared_type` names that class at that
    position; anything else is refused rather than handed over as the fields it arrived as, so a
    model never quietly becomes a dict. A declared builtin is held to the case that decodes to it in
    the same spirit: `"5"` does not satisfy a declared `int`. See `_match_wire_case`.

    Args:
        payload: The message to be mapped.
        declared_type: The type this side expects, from a handler's parameter annotation or a
            caller's `response_type`. Validated with Pydantic, so unions, generics, and arbitrary
            classes such as pl.DataFrame all work. A model with a field of an arbitrary type needs
            `model_config = ConfigDict(arbitrary_types_allowed=True)`, as it would under FastAPI;
            a stdlib dataclass spells that `__pydantic_config__ = ConfigDict(...)`.

    Returns:
        A value of the declared type, or of one of several allow-listed types if none was declared.

    Raises:
        DeserializationError: If the payload is unreadable, doesn't satisfy `declared_type`, or
            carries a rebuildable object `declared_type` says nothing about. Faults on this side, such as
            a dtype this container can't materialize, propagate unchanged so the gateway reports
            them against Kaggle. See BaseGateway.handle_server_error.
    """
    value = _decode(payload, declared_type)
    if declared_type is Any:  # Nothing was declared; the decoded value is the answer.
        return value
    try:
        return _validator(_resolve(declared_type)).validate_python(value)
    except pydantic.ValidationError as e:
        raise DeserializationError(f'Payload does not match the declared type {_describe(declared_type)}: {e}') from e


def _decode(payload: kaggle_evaluation_proto.Payload, declared: Any) -> Any:
    """Map a Payload to a Python value, rebuilding rebuildable objects from the classes `declared` names.

    `declared` is the annotation for this position specifically: the whole declared type at the top,
    a field's own annotation inside a rebuildable object. See `_rebuild`.
    """
    match payload.WhichOneof('value'):
        # Primitives
        case 'str_value':
            return payload.str_value
        case 'bool_value':
            return payload.bool_value
        case 'int_value':
            return payload.int_value
        case 'float_value':
            return payload.float_value
        case 'none_value':
            return None
        # Iterables for nested types
        case 'list_value':
            return [_decode(item, declared) for item in payload.list_value.payloads]
        case 'tuple_value':
            return tuple(_decode(item, declared) for item in payload.tuple_value.payloads)
        case 'dict_value':
            return {key: _decode(value, declared) for key, value in payload.dict_value.payload_map.items()}
        case 'rebuildable_object_value':
            return _rebuild(payload.rebuildable_object_value, declared)
        # Allowlisted special types
        case 'pandas_dataframe_value':
            return _ipc_to_pandas(payload.pandas_dataframe_value)
        case 'polars_dataframe_value':
            try:
                return pl.read_ipc_stream(io.BytesIO(payload.polars_dataframe_value))
            except pl.exceptions.PolarsError as e:
                # Every polars version we support both writes and reads Arrow IPC view types, so a
                # read failure here means the bytes are bad rather than that the peer outran us.
                raise DeserializationError(f'Payload is not a readable Arrow IPC stream: {e}') from e
        case 'pandas_series_value':
            df = _ipc_to_pandas(payload.pandas_series_value)
            if len(df.columns) != 1:
                raise DeserializationError(f'Payload declared a pandas Series but decoded to {len(df.columns)} columns; expected exactly 1.')
            return pd.Series(df[df.columns[0]])
        case 'polars_series_value':
            df_payload = kaggle_evaluation_proto.Payload(polars_dataframe_value=payload.polars_series_value)
            return _decode(df_payload, declared).to_series()
        case 'numpy_array_value':
            return _npy_to_numpy(payload.numpy_array_value)
        case 'numpy_scalar_value':
            data = _npy_to_numpy(payload.numpy_scalar_value)
            # As of Numpy 2.0.2, np.load for a numpy scalar yields a dimensionless array instead of a scalar.
            # Validate before restoring the scalar type, because that conversion throws on a non-scalar array.
            if data.shape != () or not issubclass(data.dtype.type, (np.number, np.bool_)):
                raise DeserializationError(
                    f'Payload declared a numpy scalar but decoded to dtype {data.dtype} with shape {data.shape}. '
                    'Only numeric and boolean scalars are supported, not bytes, strings, or objects.'
                )
            return data.dtype.type(data)  # Restore the expected numpy scalar type.
        case 'bytes_io_value':
            return io.BytesIO(payload.bytes_io_value)
        case 'bytes_value':
            return payload.bytes_value
        # No honest sender holds a set of unhashable values, but the wire can nest a `list_value` in a
        # `set_value`, and the TypeError that follows is not a DeserializationError, so unguarded it is
        # charged to us. Only the construction is guarded: a TypeError below it really is our bug.
        case ('set_value' | 'frozenset_value') as kind:
            build = set if kind == 'set_value' else frozenset
            members = [_decode(item, declared) for item in getattr(payload, kind).payloads]
            try:
                return build(members)
            except TypeError as e:
                raise DeserializationError(f'Payload declared a {build.__name__} but holds a value that cannot be a member of one: {e}.') from e
        case _:
            raise DeserializationError(f'Payload contains no value this version can decode; found case {payload.WhichOneof("value")}.')


### What the receiver declared


def _is_rebuildable(annotation: Any) -> bool:
    """Whether an annotation names a class the sender would have sent as a `rebuildable_object_value`."""
    return isinstance(annotation, type) and (issubclass(annotation, (pydantic.BaseModel, enum.Enum)) or dataclasses.is_dataclass(annotation))


def _bare_name(qualname: str) -> str:
    """The class's own name, dropping the scope `__qualname__` prefixes it with.

    What both sides are matched on: a class defined inside a function carries that scope
    (`make_gateway.<locals>.Address`), and the two sides need not agree on where it was defined.
    """
    return qualname.rpartition('.')[2]


@functools.cache
def _names_in(annotation: Any) -> dict[str, type | None]:
    """The rebuildable classes an annotation names, keyed by name -- the classes a payload here may pick.

    Deliberately shallow: `list[Point]` and `Point | None` both name Point, but `Rectangle` names
    only Rectangle. The Points its fields hold belong to those fields, and are reached through
    `_field_annotations` when a Rectangle is actually being rebuilt. Keeping the levels apart is what
    stops a sender nominating any class reachable from anywhere in the declared type.

    A name two classes answer to maps to None: the wire carries the bare name and nothing else, so
    nothing could say which was meant.

    Only ever called with a type, never with an enum member: these caches key on hashing and equality,
    and `Status.OK == Result.OK` holds for any two str enums sharing a value, so a member would answer
    to whichever class reached the cache first. The loop below converts before it recurses.

    Cached, so the dict is the one every later caller gets: read it, never edit it. What it holds is
    the set of classes a payload at this position may name, and an entry added here is an entry added
    to that.
    """
    if _is_rebuildable(annotation):
        return {_bare_name(annotation.__qualname__): annotation}
    found: dict[str, type | None] = {}
    for argument in get_args(annotation):
        # A `Literal` names an enum by naming a member, the one thing an annotation can name that is a
        # value rather than a type -- and how Pydantic spells a discriminated union's tag, so without
        # this the standard tagged-union model cannot be sent. Which member stays validation's
        # question, and it still refuses `Kind.DOG` here.
        named = type(argument) if isinstance(argument, enum.Enum) else argument
        for name, target in _names_in(named).items():
            if found.setdefault(name, target) is not target:
                found[name] = None
    return found


@functools.cache
def _field_annotations(target: type) -> dict[str, Any]:
    """What a model's or dataclass's fields are declared to hold, keyed by field name.

    One level down from `_names_in(target)`: each field is decoded against its own annotation and no
    other, so every position the whole way down is governed by something the receiver wrote. Only
    ever called for a model or a dataclass; `_rebuild` has already sent an enum on its way.

    Cached and shared, on the same terms as `_names_in`: read it, never edit it.
    """
    if issubclass(target, pydantic.BaseModel):
        if not target.__pydantic_complete__:
            # A model that still has unresolved forward references reports its fields as the
            # ForwardRefs they were written as, and a ForwardRef names no class, so every rebuildable
            # below it would be refused as one the receiver never declared -- blaming the sender for
            # the receiver's own unbuilt model. Pydantic rebuilds lazily on first validation, which
            # for a receiving process is after this.
            # Depth 0 because the namespace that can resolve these is the model's own module, not
            # whichever Hearth frame happened to decode a payload first.
            # `raise_errors=False` because a reference that cannot be resolved at all is a different
            # problem from one that merely hasn't been yet: leaving it to fail below keeps it a
            # DeserializationError the gateway can classify, rather than a PydanticUndefinedAnnotation
            # escaping a core module as an unhandled error.
            target.model_rebuild(_parent_namespace_depth=0, raise_errors=False)
        return {name: field.annotation for name, field in target.model_fields.items()}
    # Names from `dataclasses.fields`, types from the hints. Taking the names from the hints too would
    # add everything else annotated in the MRO -- ClassVars, and whatever a non-dataclass base declared
    # -- opening positions no real payload can occupy. `field.type` won't do for the types, being a
    # string under `from __future__ import annotations`. An annotation that won't resolve is left to
    # raise, so the host hears their own NameError rather than the peer being blamed for it.
    hints = typing.get_type_hints(target)
    fields = dataclasses.fields(target)  # ty: ignore[invalid-argument-type] -- `_is_rebuildable` admits only a dataclass here.
    return {field.name: hints[field.name] for field in fields}


@functools.cache
def _instance_only(target: type) -> Any:
    """`target`, narrowed to reject anything that is not already an instance of it.

    A validator wrapping the class, not `pydantic.InstanceOf`, which says the same thing more plainly
    but says it as a schema of its own -- and a variant whose schema isn't a model's cannot be
    discriminated, so `InstanceOf` costs `Field(discriminator=...)` entirely. Wrapping keeps the
    class's schema, and so keeps the tag.

    A wrap validator rather than a `BeforeValidator` so that schema is read but never entered.
    `_rebuild` has already built the object through it, and `revalidate_instances='always'` would
    otherwise run every validator the host wrote a second time over data that already passed.
    """

    def require_instance(value: Any, _build: pydantic.ValidatorFunctionWrapHandler) -> Any:
        if not isinstance(value, target):
            # ValueError despite the check being a type check: Pydantic folds only ValueError and
            # AssertionError into a ValidationError, and anything else escapes as a fault of ours.
            raise ValueError(f'expected an instance of {target.__name__}, not a {type(value).__name__} to be built into one')  # noqa: TRY004
        return value

    return typing.Annotated[target, pydantic.WrapValidator(require_instance)]  # ty: ignore[invalid-type-form] -- built at runtime, so `target` is a value here.


# The builtins a Payload case decodes to directly, and so the only declared types held to the case
# the sender set. Pydantic raises at build time for a strict model or a strict pl.DataFrame, and both
# are already matched exactly by other means.
_STRICT_LEAVES = (str, bytes, int, float, bool, list, tuple, dict, set, frozenset)


def _match_wire_case(declared: Any, resolved: Any) -> Any:
    """`resolved`, held to the wire case `declared` decodes from, if `declared` is one of those leaves.

    `Strict()` alone covers the conversions between builtins but not a numpy scalar, which is a
    registered stand-in for the builtin it resembles: it satisfies the check underneath and Pydantic
    hands back a plain copy with the dtype stripped. `Strict()` even takes an np.float32 for a
    declared float, unrelated as the two are by inheritance.

    Applied per leaf rather than through the TypeAdapter config, which cannot express that second
    half and would not reach inside a `list[...]` if it could. `declared` is the bare type or origin
    the rule is keyed on; `resolved` is what `_resolve` made of it, its members already carrying
    their own copy of the rule.
    """
    if declared not in _STRICT_LEAVES:
        return resolved

    def refuse_lookalike(value: Any) -> Any:
        if isinstance(value, np.generic):
            # ValueError despite the check being a type check, for the reason `_instance_only` gives.
            raise ValueError(  # noqa: TRY004
                f'sender transmitted np.{type(value).__name__}, but a plain {declared.__name__} was declared here; '
                f'annotate the receiver `{declared.__name__} | np.{type(value).__name__}` to accept one'
            )
        return value

    return typing.Annotated[(resolved, pydantic.Strict(), pydantic.BeforeValidator(refuse_lookalike))]


@functools.cache
def _resolve(annotation: Any) -> Any:
    """Rewrite a declared type so that everything in it is held to what the sender actually sent.

    Every rebuildable class accepts only a real instance. `_rebuild` has already built each
    rebuildable object into an instance of the class the receiver named, so "was this sent as a
    model?" and "is this a model?" are the same question by now. `_instance_only` asks it. Without it
    Pydantic would build a declared model out of any dict with the right keys -- the coercion the
    separate wire case exists to prevent -- and would pick between `Fahrenheit(degrees: float)` and
    `Celsius(degrees: float)` by shape, which can't tell them apart.

    Conversely, a declared builtin accepts only the wire case that decodes to it; see
    `_match_wire_case`. The two rules are the same rule read from either end.

    Applied wherever the class appears, but not inside a model it names: those fields are the model's
    own schema, so a dict may still become a nested model there, and a str may still coerce into a
    `float` field, as they would under FastAPI.
    """
    if _is_rebuildable(annotation):
        return _instance_only(annotation)
    origin = get_origin(annotation)
    if origin is None:
        return _match_wire_case(annotation, annotation)
    if origin is typing.Annotated:
        # Rebuilt inside out: the host's metadata goes back on against the class, with the instance
        # check outside it. Pydantic applies a constraint to the schema directly beneath it and the
        # check is not one, so the other order raises while the validator is built -- refusing an
        # annotation Pydantic itself accepts.
        target, *metadata = get_args(annotation)
        if _is_rebuildable(target):
            return typing.Annotated[(target, *metadata, *_instance_only(target).__metadata__)]
        return typing.Annotated[(_resolve(target), *metadata)]
    # An enum member -- a `Literal`'s argument -- has nothing to resolve and is passed through, but
    # passed through here rather than by a recursive call: see `_names_in` on why a member must not
    # reach one of these caches. Returning the wrong member would rewrite the Literal itself.
    members = tuple(member if isinstance(member, enum.Enum) else _resolve(member) for member in get_args(annotation))
    # `list[X]`, `dict[str, X]` and friends all rebuild by re-subscripting the origin with the
    # resolved arguments; a union is spelled with `Union` however it was written.
    if origin in (types.UnionType, Union):
        return Union[members]  # noqa: UP007 -- only the subscript form takes a tuple.
    return _match_wire_case(origin, origin[members])


def _rebuild_member(rebuildable: kaggle_evaluation_proto.RebuildableObject, target: type[enum.Enum]) -> enum.Enum:
    """Look up the member of `target` the payload's value names, refusing anything short of an exact match."""
    value = _decode(rebuildable.value, Any)  # A primitive from any honest sender; serialize refuses the rest.
    try:
        member = target(value)
    except (ValueError, TypeError) as e:  # No such value, or one an enum cannot even look up.
        raise DeserializationError(f'Payload carries {rebuildable.class_name}, but no member of {target.__name__} has the value {value!r}.') from e
    if type(member.value) is not type(value):
        # Enum lookup matches on `==`, under which True == 1 and 1.0 == 1, so a member would answer
        # to values of types it was never given. Nothing else in this format lets a bool stand in
        # for an int, and a member picked that way is one the sender did not send.
        raise DeserializationError(
            f'Payload carries {rebuildable.class_name} with a {type(value).__name__} value {value!r}, but '
            f'{target.__name__}.{member.name} has the {type(member.value).__name__} value {member.value!r}.'
        )
    return member


def _rebuild(rebuildable: kaggle_evaluation_proto.RebuildableObject, declared: Any) -> Any:
    """Turn a rebuildable object into an instance of the class the receiver declared under that name.

    The sender's class name selects the class, but only from the ones the receiver named *for this
    position*: it is a key into `_names_in(declared)`, never something to import. So declaring
    `Person` does not invite a bare `Address`, even though Person has one as a field. A name that
    isn't there is refused before anything is constructed rather than delivered as the fields it
    arrived as, which is what keeps a `Celsius` from arriving as the `Fahrenheit` it fits perfectly.

    Each field is then decoded against its own annotation, so the same rule applies the whole way down.
    """
    name = _bare_name(rebuildable.class_name)
    classes = _names_in(declared)
    if name not in classes:
        # Spelled out because a subclass is the case hosts hit first: a declared type names the classes
        # it accepts, and inheriting from one is not being one. Three forms because the right fix
        # differs by position -- an unannotated one wants `Color` rather than `Any | Color`, and a
        # container wants `list[Person | Address]` rather than the `list | Address` a union suggests.
        origin = get_origin(declared)
        if declared is Any:
            fix = f'Annotate the receiver with {name}, where the position takes an annotation -- *args, **kwargs, and undeclared extra fields do not.'
        elif origin is None or origin in (types.UnionType, Union):
            fix = f'Annotate the receiver with every class this position accepts, e.g. `{_describe(declared)} | {name}`.'
        else:
            fix = f'Add {name} to the annotation at the position it arrives in, within {_describe(declared)}.'
        raise DeserializationError(
            f'Payload carries {rebuildable.class_name}, which the declared type {_describe(declared)} does not account for. '
            f'{fix} Alternatively, have the sender transmit a plain dict.'
        )
    target = classes[name]
    if target is None:
        raise DeserializationError(
            f'Payload carries {rebuildable.class_name}, which the declared type {_describe(declared)} uses for more than one class, '
            'so the name cannot say which was meant. Rename one of them, or declare only the one this position should accept.'
        )
    # Which content is right is decided by the class the receiver named, not the one the payload
    # carries: a model handed a bare value would be a constructor called with peer-chosen arguments,
    # and an enum handed a map of fields is not a member of anything.
    content = rebuildable.WhichOneof('content')
    if issubclass(target, enum.Enum):
        if content != 'value':
            raise DeserializationError(
                f'Payload carries {rebuildable.class_name} without a value, but a member of {target.__name__} is rebuilt from its value.'
            )
        return _rebuild_member(rebuildable, target)
    if content != 'fields':
        raise DeserializationError(
            f'Payload carries {rebuildable.class_name} without fields, but a {target.__name__} is rebuilt from its fields; only an enum travels as a bare value.'
        )
    annotations = _field_annotations(target)
    unaccounted = rebuildable.fields.payload_map.keys() - annotations.keys()
    # Unless the class asked for extras, in which case they are part of what its author declared.
    if unaccounted and _config(target).get('extra') != 'allow':
        # Pydantic would drop these silently. Both sides of a competition are built together against
        # one set of classes, so there is no version skew to absorb and a field the receiver does not
        # have is a disagreement about the contract rather than something to quietly discard.
        raise DeserializationError(
            f'Payload carries {rebuildable.class_name} with {sorted(unaccounted)}, which {target.__name__} does not declare. '
            f'Its fields are {sorted(annotations)}.'
        )
    # An extra lands on `Any`, so a rebuildable object among them is refused: a field the class never
    # declared has no annotation to say which class could legitimately arrive there.
    fields = {key: _decode(value, annotations.get(key, Any)) for key, value in rebuildable.fields.payload_map.items()}
    try:
        # By name because that is what the sender put on the wire, the whole way down. Pydantic
        # otherwise populates by alias alone, so a `Field(alias=...)` model would serialize fine and
        # then refuse to come back.
        return _validator(target).validate_python(fields, by_name=True)
    except pydantic.ValidationError as e:
        raise DeserializationError(f'Payload carries {rebuildable.class_name}, whose fields do not build a {target.__name__}: {e}') from e


# Needed for any declared type Pydantic has no schema for, such as pl.DataFrame, where the check
# degrades to isinstance. Rejected outright for types that carry their own config; see below.
_ARBITRARY_TYPES = pydantic.ConfigDict(arbitrary_types_allowed=True)


@functools.cache
def _validator(annotation: Any) -> pydantic.TypeAdapter:
    try:
        return pydantic.TypeAdapter(annotation, config=_ARBITRARY_TYPES)
    except pydantic.PydanticUserError:
        # Pydantic refuses a config argument for any type entitled to carry its own: models,
        # dataclasses, TypedDicts. A stdlib dataclass with an arbitrary-typed field must therefore
        # set `__pydantic_config__` itself; without it Pydantic raises rather than guessing.
        return pydantic.TypeAdapter(annotation)


def _describe(declared_type: Any) -> str:
    """A declared type as the host who wrote it would recognize it, for error messages.

    `__name__` for a plain class, which `str` spells `<class 'models.Point'>`, but never for a
    parameterized generic: `list[Point].__name__` is `list`, dropping the argument that a refusal
    is usually about and leaving two different declarations looking identical in the message.
    """
    if get_origin(declared_type) is not None:
        return str(declared_type)
    return getattr(declared_type, '__name__', None) or str(declared_type)
