"""S37 research-only deterministic duplicate *failed edit* interceptor.

An ADK BasePlugin intercepts native tool calls before execution. Only an exact
edit_file invocation already rejected against the same unchanged workspace
revision is blocked. No model weights, prompts, grader, or source edits here.

NOT competition-packagable: Python callbacks are not authorized by the closed
adk-submission registry. Keep this file OUTSIDE submission.zip.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


def arguments_digest(name: str, arguments: dict[str, Any]) -> str | None:
    """Hash canonical argument structure, without retaining source literals."""
    if not isinstance(arguments, dict):
        return None
    try:
        data = json.dumps(
            {"tool": name, "args": arguments}, sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def error_kind(value: Any) -> str | None:
    """Classify deterministic exact-edit failures; all uncertainty fails open."""
    if isinstance(value, dict):
        if isinstance(value.get("result"), (dict, str)):
            inner = error_kind(value["result"])
            if inner:
                return inner
        if isinstance(value.get("error"), (dict, str)):
            inner = error_kind(value["error"])
            if inner:
                return inner
        error_type = str(value.get("error_type") or "")
        message = str(value.get("error_message") or "")
        if error_type == "FileEditError":
            return "FileEditError"
        # Broken parser/tool args are not always dispatched via callback.
        if "mandatory input parameters" in message.lower():
            return "MissingArguments"
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            # Some tool wrappers return plain error text.
            lower = value.lower()
            if "fileediterror" in lower and ("not found" in lower or
                                             "occurs" in lower):
                return "FileEditError"
            if "mandatory input parameters" in lower:
                return "MissingArguments"
            return None
        return error_kind(parsed)
    return None


def was_synthetically_blocked(value: Any) -> bool:
    """Do not let ADK's after-tool hook erase the very failure we blocked."""
    if isinstance(value, dict):
        if value.get("error_type") == "RepeatedFailedEdit":
            return True
        return was_synthetically_blocked(value.get("result"))
    if isinstance(value, str):
        try:
            return was_synthetically_blocked(json.loads(value))
        except (ValueError, TypeError):
            return False
    return False


def session_id(tool_context: Any) -> str | None:
    """Fail open if a stable task/session identity cannot be established."""
    ctx = getattr(tool_context, "_invocation_context", None)
    session = getattr(ctx, "session", None)
    value = getattr(session, "id", None)
    if isinstance(value, str) and value:
        return value
    return None


class RepetitionGuard:
    """Tiny per-session failure ledger with conservative workspace epochs.

    Every run_command or write_file attempt may mutate source and therefore
    invalidates all failed anchors BEFORE dispatch, even if execution raises.
    An edit with different arguments also invalidates before dispatch, as it
    may change source without producing an after-tool callback. read_file and
    get_status never invalidate. No symbolic similarity guesses.
    """

    def __init__(self):
        self._epochs: dict[str, int] = {}
        self._failures: dict[str, set[tuple[int, str]]] = {}
        self._blocked_counts: dict[str, Counter[tuple[int, str]]] = {}
        self.max_blocks_per_failed_edit = 2
        self.stats: Counter[str] = Counter()
        self.events: list[dict[str, Any]] = []

    def _record(self, session: str | None, event: str,
                key: str | None = None, kind: str | None = None):
        self.stats[event] += 1
        # Bounded audit; never store the old/new source code or raw arguments.
        if len(self.events) < 300:
            row = {"session_sha256": (
                hashlib.sha256(session.encode()).hexdigest()[:12]
                if session else None), "event": event}
            if key:
                row["fingerprint"] = key[:16]
            if kind:
                row["kind"] = kind
            self.events.append(row)

    def before(self, session: str | None, name: str,
               args: dict[str, Any]) -> dict[str, Any] | None:
        if not session:
            self._record(None, "missing_session")
            return None
        self._record(session, "callback_tool_calls")
        if name in ("run_command", "write_file"):
            # Must run BEFORE execution: tools may mutate then raise, and an
            # earlier plugin may handle the exception without an after hook.
            self._advance(session, "potential_workspace_change")
            return None
        if name != "edit_file":
            return None
        self._record(session, "edit_attempts")
        key = arguments_digest(name, args)
        if key is None:
            # Unknown arguments may still produce a mutating tool invocation.
            self._advance(session, "uncanonicalizable_arguments")
            return None
        token = (self._epochs.get(session, 0), key)
        failures = self._failures.get(session, set())
        if token not in failures:
            if failures:
                # A different edit may mutate before the after-hook executes,
                # including on cancellation. Do not block a stale old anchor.
                self._advance(session, "new_edit_attempt")
            return None
        counts = self._blocked_counts.setdefault(session, Counter())
        if counts[token] >= self.max_blocks_per_failed_edit:
            # Do not allow infinite, uncharged model/tool-call loops.
            # Let the native tool budget eventually terminate repetition.
            self._record(session, "block_cap_reached", key)
            return None
        counts[token] += 1
        self._record(session, "duplicate_failures_blocked", key)
        return {
            "status": "error",
            "error_type": "RepeatedFailedEdit",
            "error_message": (
                "Identical edit_file arguments already failed on this "
                "unchanged workspace revision. The tool was not executed. "
                "Read the current source and generate a DIFFERENT exact "
                "old_string/new_string or change approach. Repeating this "
                "call will not progress."
            ),
        }

    def after(self, session: str | None, name: str,
              args: dict[str, Any], response: Any) -> None:
        if not session:
            return
        if name in ("run_command", "write_file"):
            # Already invalidated by before(); preserve that revision here.
            return
        if name != "edit_file":
            return
        if was_synthetically_blocked(response):
            return
        key = arguments_digest(name, args)
        kind = error_kind(response)
        if kind and key:
            self._failures.setdefault(session, set()).add(
                (self._epochs.get(session, 0), key))
            self._record(session, "deterministic_edit_failures", key, kind)
        elif not kind:
            # A nonerror tool response may be a success. Fail open.
            self._advance(session, "edit_success_or_unknown")
        else:
            self._record(session, "uncanonicalizable_arguments")

    def on_error(self, session: str | None, name: str,
                 args: dict[str, Any], error: BaseException) -> None:
        if not session:
            return
        if name in ("run_command", "write_file"):
            # before() already invalidated even if this callback is skipped.
            return
        if name == "edit_file" and type(error).__name__ == "FileEditError":
            key = arguments_digest(name, args)
            if key:
                self._failures.setdefault(session, set()).add(
                    (self._epochs.get(session, 0), key))
                self._record(session, "deterministic_edit_failures", key,
                             "FileEditError")

    def _advance(self, session: str, reason: str) -> None:
        self._epochs[session] = self._epochs.get(session, 0) + 1
        self._failures.pop(session, None)
        self._blocked_counts.pop(session, None)
        self._record(session, reason)

    def report(self) -> dict:
        return {"counts": dict(self.stats), "events": self.events}


def new_adk_plugin():
    """Lazy import keeps tests and offline trace replay independent of ADK."""
    from google.adk.plugins.base_plugin import BasePlugin

    class S37DuplicateFailedEditPlugin(BasePlugin):
        def __init__(self):
            super().__init__(name="s37_duplicate_failed_edit")
            self.guard = RepetitionGuard()

        async def before_tool_callback(
            self, *, tool, tool_args, tool_context
        ):
            return self.guard.before(session_id(tool_context), tool.name,
                                     tool_args)

        async def after_tool_callback(
            self, *, tool, tool_args, tool_context, result
        ):
            self.guard.after(session_id(tool_context), tool.name,
                             tool_args, result)
            return None

        async def on_tool_error_callback(
            self, *, tool, tool_args, tool_context, error
        ):
            self.guard.on_error(session_id(tool_context), tool.name,
                                tool_args, error)
            return None

    return S37DuplicateFailedEditPlugin()


def patch_app_initializer(original_init, plugin_factory, attached: list):
    """Inject one new plugin into every harness-created ADK App.

    Scoped only to the research runner. Do not patch the grader or the native
    tool implementation. Abort if the App constructor contract changed.
    """
    def initialize(app, *args, **kwargs):
        if args:
            raise RuntimeError("Unsupported ADK App positional initialization")
        if "plugins" in kwargs and not isinstance(kwargs["plugins"], (tuple, list)):
            raise RuntimeError("Unsupported ADK App.plugins type")
        plugin = plugin_factory()
        # The guard must inspect raw tool responses before another plugin
        # can short-circuit callback dispatch with a non-None result.
        kwargs["plugins"] = [plugin, *(kwargs.get("plugins") or [])]
        original_init(app, **kwargs)
        attached.append(plugin)
    return initialize
