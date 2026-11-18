"""
Core implementation of the client module, implementing generic communication
patterns with Python in / Python out supporting many (nested) primitives +
special data science types like DataFrames or np.ndarrays, with gRPC + protobuf
as a backing implementation.
"""

import inspect
import json
import socket
import time
import typing
from concurrent import futures
from types import FunctionType, MethodType
from typing import Any

import grpc

from kaggle_evaluation.core import proto_compiler, telemetry
from kaggle_evaluation.core.serialization import deserialize, serialize

# Compile the protobuf / gRPC stubs into a temp dir on sys.path so the gateway
# and inference server can run on different docker-python image versions without
# protobuf/grpcio version-mismatch errors.
proto_compiler.ensure_compiled()

import kaggle_evaluation_pb2 as kaggle_evaluation_proto  # ty: ignore[unresolved-import]
import kaggle_evaluation_pb2_grpc as kaggle_evaluation_grpc  # ty: ignore[unresolved-import]


class GRPCDeadlineError(Exception):
    pass


class ServerDiedError(Exception):
    pass


_SERVICE_CONFIG = {
    # Service config proto: https://github.com/grpc/grpc-proto/blob/ec886024c2f7b7f597ba89d5b7d60c3f94627b17/grpc/service_config/service_config.proto#L377
    'methodConfig': [
        {
            'name': [{}],  # Applies to all methods
            # See retry policy docs: https://grpc.io/docs/guides/retry/
            'retryPolicy': {
                'maxAttempts': 5,
                'initialBackoff': '0.1s',
                'maxBackoff': '1s',
                'backoffMultiplier': 1,  # Ensure relatively rapid feedback in the event of a crash
                'retryableStatusCodes': ['UNAVAILABLE'],
            },
        }
    ]
}

NUM_FALLBACK_PORTS = 5
MAX_NUM_SERVERS = 10
PORT_SPACING = NUM_FALLBACK_PORTS + 5

# Include potential fallback ports
GRPC_PORTS = [50051] + [60053 + i * PORT_SPACING for i in range(MAX_NUM_SERVERS)]

_GRPC_CHANNEL_OPTIONS = [
    # -1 for unlimited message send/receive size
    # https://github.com/grpc/grpc/blob/v1.64.x/include/grpc/impl/channel_arg_names.h#L39
    ('grpc.max_send_message_length', -1),
    ('grpc.max_receive_message_length', -1),
    # https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    ('grpc.keepalive_time_ms', 60_000),  # Time between heartbeat pings
    ('grpc.keepalive_timeout_ms', 5_000),  # Time allowed to respond to pings
    ('grpc.http2.max_pings_without_data', 0),  # Remove another cap on pings
    ('grpc.keepalive_permit_without_calls', 1),  # Allow heartbeat pings at any time
    ('grpc.http2.min_ping_interval_without_data_ms', 1_000),
    ('grpc.service_config', json.dumps(_SERVICE_CONFIG)),
]


DEFAULT_DEADLINE_SECONDS = 60 * 60
_RETRY_SLEEP_SECONDS = 1 / len(GRPC_PORTS)
# Enforce a relatively strict server startup time so users can get feedback quickly if they're not
# configuring KaggleEvaluation correctly. We really don't want notebooks timing out after nine hours
# somebody forgot to start their inference_server. Slow steps like loading models
# can happen during the first inference call if necessary.
STARTUP_LIMIT_SECONDS = 60 * 15
# Bounds that first inference call, which the comment above invites users to load models during, and
# which endpoint_deadline_seconds is routinely far too tight for (StandardGateway defaults to five
# minutes). Unbounded, it burns the whole notebook budget without ever blaming the submission.
FIRST_CALL_LIMIT_SECONDS = 60 * 60


def _get_available_port(ports: list[int] | None = None) -> int:
    """Identify the first available port out of all GRPC_PORTS"""
    ports_to_check = ports if ports is not None else GRPC_PORTS
    for port in ports_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
            except OSError:
                continue
        return port

    raise ValueError(f'None of the expected ports {ports_to_check} are available.')


def _get_ports_to_try(port: int | None, ports: list[int] | None) -> list[int]:
    if port is not None:
        preferred = [port] + (ports or [port + i for i in range(1, NUM_FALLBACK_PORTS + 1)])
    elif ports is not None:
        preferred = ports
    else:
        return GRPC_PORTS

    preferred_set = set(preferred)
    preferred += [p for p in GRPC_PORTS if p not in preferred_set]
    return preferred


### Client code


class Client:
    """
    Class which allows callers to make KaggleEvaluation requests.
    """

    def __init__(self, channel_address: str = 'localhost', port: int | None = None, ports: list[int] | None = None) -> None:
        self.channel_address = channel_address
        self.port = port
        self.ports = ports
        self.channel: grpc.Channel | None = None
        self._made_first_connection = False
        self.endpoint_deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
        self.stub: kaggle_evaluation_grpc.KaggleEvaluationServiceStub | None = None
        self._hostname_was_valid = False

    def _send_to_stub(self, request, timeout: float) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        """Send one request, reporting a timeout as GRPCDeadlineError rather than a raw gRPC error.

        Both send paths go through here: left as an _InactiveRpcError, a first call deadline would
        reach handle_server_error looking like a connection failure.
        """
        if self.stub is None:
            raise RuntimeError('Stub is not initialized')
        try:
            return self.stub.Send(request, wait_for_ready=False, timeout=timeout)
        except grpc._channel._InactiveRpcError as err:  # ty: ignore
            if 'StatusCode.DEADLINE_EXCEEDED' in str(err):
                # Both deadlines share an error type, so the message is all that distinguishes a
                # submission that blew its model loading budget from one that is merely slow.
                message = f'The inference server did not answer `{request.name}` within {timeout:g} seconds.'
                if not self._made_first_connection:
                    message += ' This was the first call, which is when models are expected to load.'
                raise GRPCDeadlineError(message)
            raise

    def _send_with_deadline(self, request) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        """Sends a message to the server while also:
        - Throwing an error as soon as the inference_server container has been shut down.
        - Allowing up to STARTUP_LIMIT_SECONDS for the inference_server to start listening, and
          FIRST_CALL_LIMIT_SECONDS for it to answer once it has.
        """
        if self._made_first_connection:
            return self._send_to_stub(request, self.endpoint_deadline_seconds)

        first_call_time = time.time()
        ports_to_try = _get_ports_to_try(port=self.port, ports=self.ports)
        telemetry.detailed_event('startup_wait_begin', {'ports_to_try': len(ports_to_try)})

        # Allow time for the server to start as long as its container is running
        while time.time() - first_call_time < STARTUP_LIMIT_SECONDS:
            for port in ports_to_try:
                self.channel = grpc.insecure_channel(f'{self.channel_address}:{port}', options=_GRPC_CHANNEL_OPTIONS)
                self.stub = kaggle_evaluation_grpc.KaggleEvaluationServiceStub(self.channel)
                try:
                    # max() so a competition that deliberately set a long endpoint deadline doesn't get a
                    # shorter first call. A deadline here ends the call rather than moving on to the next
                    # port: the server accepted the request, so retrying it elsewhere would run it twice.
                    response = self._send_to_stub(request, max(FIRST_CALL_LIMIT_SECONDS, self.endpoint_deadline_seconds))
                    self._made_first_connection = True
                    telemetry.detailed_event('connection_established', {'port': port, 'wait_seconds': time.time() - first_call_time})
                    self._hostname_was_valid = True
                    return response
                except grpc._channel._InactiveRpcError as err:  # ty: ignore
                    telemetry.detailed_event('connection_attempt_failed', {'port': port})
                    if 'StatusCode.UNAVAILABLE' not in str(err):
                        raise
                # Confirm the inference_server container is still alive & it's worth waiting on the server.
                # If the inference_server container is no longer running or the channel address is invalid this will throw a socket.gaierror.
                try:
                    socket.gethostbyname(self.channel_address)
                    self._hostname_was_valid = True
                except socket.gaierror as err:
                    if self._hostname_was_valid:
                        raise ServerDiedError(f'The inference_server container has died after startup, causing: {err}')
                    else:
                        raise
                time.sleep(_RETRY_SLEEP_SECONDS)

        if not self._made_first_connection:
            raise RuntimeError(f'Failed to connect to server after waiting {STARTUP_LIMIT_SECONDS} seconds')

        raise RuntimeError('Unreachable')

    def serialize_request(self, name: str, *args, **kwargs) -> kaggle_evaluation_proto.KaggleEvaluationRequest:
        """Serialize a single request. Exists as a separate function from `send`
        to enable gateway concurrency for some competitions.
        """
        already_serialized = (len(args) == 1) and isinstance(args[0], kaggle_evaluation_proto.KaggleEvaluationRequest)
        if already_serialized:
            return args[0]  # args is a tuple of length 1 containing the request
        return kaggle_evaluation_proto.KaggleEvaluationRequest(
            name=name, args=map(serialize, args), kwargs={key: serialize(value) for key, value in kwargs.items()}
        )

    def send(self, name: str, *args, response_type: Any = Any, **kwargs) -> Any:
        """Sends a single KaggleEvaluation request.

        Args:
            name: The endpoint name for the request.
            *args: Variable-length/type arguments to be supplied on the request.
            response_type: The type this caller expects back. The response is validated against it,
                which is what rebuilds a data model the far side sent. Nothing the responder says
                influences the choice; see serialization.deserialize.
            **kwargs: Key-value arguments to be supplied on the request.

        Returns:
            The response, of `response_type` if one was given, otherwise of one of several
            allow-listed data types.
        """
        with telemetry.sampled_span('Client.send', {'rpc.method': name, 'rpc.system': 'grpc'}) as s:
            request = self.serialize_request(name, *args, **kwargs)
            if s is not None:
                s.set_attribute('request.size_bytes', request.ByteSize())
            telemetry.log_payload('relay.request', {'method': name, 'args': args, 'kwargs': kwargs}, category=telemetry.PayloadLogging.ALL)
            response = self._send_with_deadline(request)
            if s is not None:
                s.set_attribute('response.size_bytes', response.payload.ByteSize())
            result = deserialize(response.payload, response_type)
            telemetry.log_payload('relay.response', {'method': name, 'result': result}, category=telemetry.PayloadLogging.RESPONSES)
            return result

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()


### Server code


def _get_endpoint_name(func: FunctionType | MethodType) -> str:
    """Get the endpoint name for a function or method.

    For functions, returns the function name.
    For bound methods, returns 'ClassName.method_name' to ensure uniqueness.
    """
    if isinstance(func, MethodType):
        class_name = type(func.__self__).__name__
        return f'{class_name}.{func.__name__}'
    return func.__name__


def _declared_parameter_types(func: FunctionType | MethodType) -> dict[str | int, Any]:
    """The types an endpoint declares for its parameters, keyed by name and, where one can arrive
    positionally, by position too: a request carries `args` unnamed and `kwargs` named, so the same
    parameter has to be findable either way.

    This is the receiving side of the relay's type contract: an argument is validated against
    whatever the handler's annotation says it should be, and an unannotated parameter takes whatever
    arrives, so long as it isn't a model or an enum -- those only arrive if something declared them.
    *args and **kwargs parameters are unannotatable in this sense and so are absent from the mapping.

    An annotation that doesn't resolve leaves its own parameter untyped rather than failing the
    endpoint: `if TYPE_CHECKING: import polars as pl` with `batch: pl.DataFrame` is ordinary Python.
    Resolved one at a time because `typing.get_type_hints` is all or nothing, and an unresolvable
    *return* type -- which never crosses the wire -- would otherwise cost every parameter its type,
    leaving rebuildable objects nowhere to land and refused as though the sender were at fault.
    """
    hints = {}
    for name, annotation in getattr(func, '__annotations__', {}).items():
        # get_type_hints needs something to read a namespace from; a stub in the handler's module
        # resolves the annotation exactly as the handler's own module would.
        stub = type('_Stub', (), {'__annotations__': {name: annotation}, '__module__': func.__module__})
        try:
            hints |= typing.get_type_hints(stub)
        # Skipped rather than raised on: a TYPE_CHECKING-guarded annotation is unresolvable at runtime
        # by design, and this runs at registration, so raising would cost the whole endpoint over one
        # parameter. Safe because untyped fails closed -- such a position refuses a model or an enum
        # rather than rebuilding one, so this can deny a legitimate payload but never admit a class
        # the receiver never declared.
        except Exception:  # noqa: BLE001, S112
            continue

    declared: dict[str | int, Any] = {}
    # `enumerate`'s index is the position an argument arrives at, since every kind that can be passed
    # positionally comes before the ones that can't. Counting only the positional parameters would
    # drift at `*args`, giving the keyword-only parameters after it positions its own arguments hold.
    for position, (name, parameter) in enumerate(inspect.signature(func).parameters.items()):
        if name not in hints or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        declared[name] = hints[name]
        if parameter.kind is not parameter.KEYWORD_ONLY:
            declared[position] = hints[name]
    return declared


class KaggleEvaluationServiceServicer(kaggle_evaluation_grpc.KaggleEvaluationServiceServicer):
    """
    Class which allows serving responses to KaggleEvaluation requests. The inference_server will run this service to listen for and respond
    to requests from the Gateway. The Gateway may also listen for requests from the inference_server in some cases.
    """

    def __init__(self, listeners: tuple[FunctionType | MethodType, ...]) -> None:
        self.listeners_map = {}
        # Resolved once here rather than per request: typing.get_type_hints re-evaluates every
        # annotation, which is far more expensive than the validation it feeds.
        self.declared_types: dict[str, dict[str | int, Any]] = {}
        for func in listeners:
            qualified_name = _get_endpoint_name(func)
            simple_name = func.__name__
            # Register under the qualified name (ClassName.method_name for methods)
            self.listeners_map[qualified_name] = func
            # Also register under the simple name so gateways can call without knowing the class or whether the callable is a method or function.
            if simple_name != qualified_name:
                if simple_name in self.listeners_map:
                    raise ValueError(f"Endpoint '{simple_name}' is already registered. Cannot register {qualified_name} under the same simple name.")
                self.listeners_map[simple_name] = func
            declared = _declared_parameter_types(func)
            self.declared_types[qualified_name] = self.declared_types[simple_name] = declared

    def Send(
        self, request: kaggle_evaluation_proto.KaggleEvaluationRequest, context: grpc.ServicerContext
    ) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        """Handler for gRPC requests that deserializes arguments, calls a user-registered function for handling the
        requested endpoint, then serializes and returns the response.

        Each argument is deserialized against the type the handler declares for that parameter, so
        a handler annotated `predict(self, batch: Batch)` receives a Batch rather than the dict it
        arrived as. The annotation is the only thing that decides; the request has no say.

        Args:
            request: The KaggleEvaluationRequest protobuf message.
            context: (Unused) gRPC context.

        Returns:
            The KaggleEvaluationResponse protobuf message.

        Raises:
            NotImplementedError if the caller has not registered a handler for the requested endpoint.
        """
        if request.name not in self.listeners_map:
            raise NotImplementedError(f'No listener for {request.name} was registered.')

        declared = self.declared_types[request.name]
        args = [deserialize(payload, declared.get(i, Any)) for i, payload in enumerate(request.args)]
        kwargs = {key: deserialize(value, declared.get(key, Any)) for key, value in request.kwargs.items()}
        response_function = self.listeners_map[request.name]
        response_payload = serialize(response_function(*args, **kwargs))
        return kaggle_evaluation_proto.KaggleEvaluationResponse(payload=response_payload)


class _Server:
    """Thin wrapper around grpc.Server that fixes a port-leak bug.

    grpc.Server.add_insecure_port() binds the port at the OS level immediately,
    but stop() only releases it if start() was called first. A server that is
    bound but never started will hold the port for the lifetime of the process,
    and worse, the port will accept TCP connections but never complete the gRPC
    handshake, causing clients to hang for ~20 s per attempt instead of failing
    fast. This wrapper ensures stop() always releases the port.
    """

    def __init__(self, server: grpc.Server):
        self._server = server
        self._started = False

    def __getattr__(self, name):
        return getattr(self._server, name)

    def start(self):
        self._started = True
        return self._server.start()

    def stop(self, grace):
        if not self._started:
            self._server.start()
        return self._server.stop(grace)


def define_server(*endpoint_listeners: FunctionType | MethodType, port: int | None = None, ports: list[int] | None = None) -> tuple[_Server, int]:
    """Registers the endpoints that the container is able to respond to, then starts a server which listens for
    those endpoints. The endpoints that need to be implemented will depend on the specific competition.

    Args:
        endpoint_listeners: Tuple of functions or bound methods that define how requests to the endpoint of the
            function/method name should be handled.
        port: Optional specific port to bind to. If None, finds an available port.
        ports: Optional list of ports to try binding to. Used if port is None.

    Returns:
        A tuple of (gRPC server object, port number). Server should be stopped at exit time.

    Raises:
        ValueError if parameter values are invalid.
    """
    if not endpoint_listeners:
        raise ValueError('Must pass at least one endpoint listener, e.g. `predict`')
    for func in endpoint_listeners:
        if not isinstance(func, (FunctionType, MethodType)):
            # ValueError is the documented, user-facing contract for every bad argument to define_server.
            raise ValueError(f'Endpoint listeners must be functions or bound methods, got {type(func)}')  # noqa: TRY004
        if func.__name__ == '<lambda>':
            raise ValueError('Endpoint listeners must be named (lambdas are not allowed)')

    server = _Server(grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=_GRPC_CHANNEL_OPTIONS))
    kaggle_evaluation_grpc.add_KaggleEvaluationServiceServicer_to_server(KaggleEvaluationServiceServicer(endpoint_listeners), server._server)
    ports_to_try = _get_ports_to_try(port=port, ports=ports)
    grpc_port = _get_available_port(ports_to_try)
    server.add_insecure_port(f'[::]:{grpc_port}')
    return server, grpc_port
