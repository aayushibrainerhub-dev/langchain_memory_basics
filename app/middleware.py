"""
middleware.py — Middleware, hooks, and dynamic prompting for the LangChain/LangGraph chatbot.

This module provides:

  ModelRequest        — Dataclass wrapping an LLM call before it is made.
  ModelResponse       — Dataclass wrapping an LLM response after it returns.
  @before_model       — Decorator to register a pre-LLM hook function.
  @after_model        — Decorator to register a post-LLM hook function.
  SummarizationMiddleware — Compresses old messages into a running summary when
                            the conversation grows beyond a configurable threshold.
  DynamicPrompt       — Builds a system prompt dynamically from session context,
                        memory type, detected user name, and registered PromptPlugins.
  RuntimeMemoryAccessor — Provides direct programmatic access to any session's
                          live memory state without going through the HTTP API.

Note on langchain.agents.middleware:
  The import path `from langchain.agents.middleware import dynamic_prompt, ModelRequest`
  does not exist in any released LangChain version.  The classes here are an equivalent
  production-quality implementation following the same design philosophy as real
  LangChain/LangGraph internals.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Generic, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ModelRequest / ModelResponse  — request / response envelope
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    """
    Wraps an LLM invocation before it is made.

    Passed to every function registered with @before_model.
    Hooks may mutate .messages or .metadata in place to alter what is sent.

    Attributes:
        messages:        The list of LangChain messages that will be sent to the LLM.
        session_id:      The session this request belongs to.
        memory_backend:  The active memory backend (sql / in_memory / memory_saver / postgres_saver).
        memory_type:     The active memory strategy (buffer / window / summary).
        timestamp:       Unix epoch when the request was created.
        metadata:        Arbitrary key/value pairs hooks may read or write.
    """
    messages: List[BaseMessage]
    session_id: str
    memory_backend: str = "sql"
    memory_type: str = "buffer"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prepend_system(self, content: str) -> None:
        """Inserts (or replaces) the leading SystemMessage."""
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(content=content)
        else:
            self.messages.insert(0, SystemMessage(content=content))

    def append_system_note(self, note: str) -> None:
        """Appends an additional SystemMessage at the end of the message list."""
        self.messages.append(SystemMessage(content=note))

    def trim_to_last_n(self, n: int, keep_system: bool = True) -> None:
        """
        Trims .messages so at most n non-system messages remain.
        If keep_system is True, the leading SystemMessage is preserved.
        """
        system_msgs = [m for m in self.messages if isinstance(m, SystemMessage)]
        non_system = [m for m in self.messages if not isinstance(m, SystemMessage)]
        trimmed = non_system[-n:] if len(non_system) > n else non_system
        if keep_system and system_msgs:
            self.messages = [system_msgs[0]] + trimmed
        else:
            self.messages = trimmed


@dataclass
class ModelResponse:
    """
    Wraps an LLM response after it is received.

    Passed to every function registered with @after_model.
    Hooks may read fields for logging / enrichment but should not alter .content
    (since it is already returned to the user).

    Attributes:
        content:         The text returned by the LLM.
        request:         The ModelRequest that produced this response.
        duration_ms:     Wall-clock time the LLM call took (milliseconds).
        token_estimate:  Rough token count estimate (4 chars ≈ 1 token).
        metadata:        Arbitrary key/value pairs hooks may read or write.
    """
    content: str
    request: ModelRequest
    duration_ms: float = 0.0
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimate: all message chars + response chars / 4
            all_text = "".join(
                getattr(m, "content", "") for m in self.request.messages
            ) + self.content
            self.token_estimate = max(1, len(all_text) // 4)


# ---------------------------------------------------------------------------
# Hook registry — @before_model / @after_model
# ---------------------------------------------------------------------------

_before_model_hooks: List[Callable[[ModelRequest], None]] = []
_after_model_hooks:  List[Callable[[ModelResponse], None]] = []
_hooks_lock = threading.Lock()


def before_model(fn: Callable[[ModelRequest], None]) -> Callable[[ModelRequest], None]:
    """
    Decorator that registers a function to run BEFORE the LLM is called.

    The decorated function receives a ModelRequest and may mutate it in place.
    Hooks are called in registration order.

    Example::

        @before_model
        def log_request(req: ModelRequest) -> None:
            logger.info(f\"[before_model] session={req.session_id} msgs={len(req.messages)}\")
    """
    with _hooks_lock:
        _before_model_hooks.append(fn)
    logger.debug(f"[before_model] Registered hook: {fn.__name__}")
    return fn


def after_model(fn: Callable[[ModelResponse], None]) -> Callable[[ModelResponse], None]:
    """
    Decorator that registers a function to run AFTER the LLM returns.

    The decorated function receives a ModelResponse and may read or enrich it.
    Hooks are called in registration order.

    Example::

        @after_model
        def log_response(resp: ModelResponse) -> None:
            logger.info(f\"[after_model] {resp.duration_ms:.0f}ms ~{resp.token_estimate} tokens\")
    """
    with _hooks_lock:
        _after_model_hooks.append(fn)
    logger.debug(f"[after_model] Registered hook: {fn.__name__}")
    return fn


def run_before_model_hooks(request: ModelRequest) -> None:
    """Executes all registered @before_model hooks in order."""
    with _hooks_lock:
        hooks = list(_before_model_hooks)
    for hook in hooks:
        try:
            hook(request)
        except Exception as e:
            logger.error(f"[before_model] Hook '{hook.__name__}' raised: {e}")


def run_after_model_hooks(response: ModelResponse) -> None:
    """Executes all registered @after_model hooks in order."""
    with _hooks_lock:
        hooks = list(_after_model_hooks)
    for hook in hooks:
        try:
            hook(response)
        except Exception as e:
            logger.error(f"[after_model] Hook '{hook.__name__}' raised: {e}")


def invoke_with_hooks(
    llm: Any,
    request: ModelRequest,
) -> ModelResponse:
    """
    Runs the full before → LLM → after pipeline:
      1. Calls all @before_model hooks (may mutate request.messages).
      2. Invokes the LLM with request.messages.
      3. Wraps the result in ModelResponse.
      4. Calls all @after_model hooks.
      5. Returns the ModelResponse.

    Args:
        llm:     Any object with an .invoke(messages) method.
        request: A ModelRequest to send.

    Returns:
        ModelResponse with the LLM reply and timing/token metadata.
    """
    run_before_model_hooks(request)

    t0 = time.perf_counter()
    raw = llm.invoke(request.messages)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    content = getattr(raw, "content", str(raw))
    response = ModelResponse(
        content=content,
        request=request,
        duration_ms=elapsed_ms,
    )

    run_after_model_hooks(response)
    return response


# ---------------------------------------------------------------------------
# Built-in @before_model / @after_model hooks (always registered)
# ---------------------------------------------------------------------------

@before_model
def _log_request(req: ModelRequest) -> None:
    logger.info(
        f"[@before_model] session={req.session_id} backend={req.memory_backend} "
        f"messages={len(req.messages)}"
    )


@after_model
def _log_response(resp: ModelResponse) -> None:
    logger.info(
        f"[@after_model] session={resp.request.session_id} "
        f"duration={resp.duration_ms:.1f}ms tokens~{resp.token_estimate}"
    )


# ---------------------------------------------------------------------------
# SummarizationMiddleware
# ---------------------------------------------------------------------------

class SummarizationMiddleware:
    """
    Middleware that automatically compresses old conversation messages into a
    running summary when the message list grows beyond a configurable threshold.

    Acts as both a @before_model hook (trimming messages) and a @after_model
    hook (updating the session summary field).

    Usage::

        middleware = SummarizationMiddleware(max_messages=10, llm_client=my_llm)
        middleware.register()   # wires before_model + after_model hooks

    How it works:
      - Before the LLM is called: if len(non-system messages) > max_messages,
        the oldest messages are summarized into a SystemMessage that is injected
        at the start.  Only the most recent `keep_last` messages are kept verbatim.
      - After the LLM returns: logs the compression stats and can optionally
        persist the new summary back to the session.
    """

    def __init__(
        self,
        max_messages: int = 10,
        keep_last: int = 4,
        llm_client: Optional[Any] = None,
        session_updater: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Args:
            max_messages:    Trigger compression when non-system messages exceed this count.
            keep_last:       Number of recent messages to keep verbatim after compression.
            llm_client:      Optional LLM used to generate the summary.
                             If None, a simple extractive summary is used.
            session_updater: Optional callable(session_id, new_summary) to persist
                             the updated summary back to the database.
        """
        self.max_messages = max_messages
        self.keep_last = keep_last
        self.llm_client = llm_client
        self.session_updater = session_updater
        self._compression_log: Dict[str, int] = {}  # session_id -> times compressed

    def _extractive_summary(self, messages: List[BaseMessage], existing_summary: str = "") -> str:
        """Generates a simple extractive summary without calling an LLM."""
        lines = []
        if existing_summary:
            lines.append(f"Previous context: {existing_summary}")
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            snippet = msg.content[:80] + ("…" if len(msg.content) > 80 else "")
            lines.append(f"{role}: {snippet}")
        return " | ".join(lines)

    def _llm_summary(self, messages: List[BaseMessage], existing_summary: str = "") -> str:
        """Generates a summary using the LLM client."""
        convo = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        )
        prompt = (
            f"Summarize the following conversation excerpt into 2-3 concise sentences.\n"
            f"Focus on key facts, user intent, and outcomes.\n"
            + (f"Prior summary: {existing_summary}\n\n" if existing_summary else "")
            + f"Conversation:\n{convo}\n\nSummary:"
        )
        try:
            res = self.llm_client.invoke(prompt)
            return getattr(res, "content", str(res)).strip()
        except Exception as e:
            logger.warning(f"[SummarizationMiddleware] LLM summary failed: {e}. Falling back.")
            return self._extractive_summary(messages, existing_summary)

    def before_hook(self, req: ModelRequest) -> None:
        """
        Pre-model hook: compresses messages if threshold is exceeded.
        Called automatically when registered via .register().
        """
        system_msgs = [m for m in req.messages if isinstance(m, SystemMessage)]
        non_system = [m for m in req.messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= self.max_messages:
            return  # No compression needed

        to_compress = non_system[: -self.keep_last]
        to_keep = non_system[-self.keep_last:]

        existing_summary = req.metadata.get("session_summary", "")
        if self.llm_client and not getattr(self.llm_client, "is_mock", False):
            summary_text = self._llm_summary(to_compress, existing_summary)
        else:
            summary_text = self._extractive_summary(to_compress, existing_summary)

        summary_msg = SystemMessage(
            content=f"[SummarizationMiddleware] Conversation summary (older messages compressed):\n{summary_text}"
        )

        # Rebuild: original system prompt + summary + kept recent messages
        req.messages = system_msgs + [summary_msg] + to_keep
        req.metadata["summarized"] = True
        req.metadata["compressed_count"] = len(to_compress)
        req.metadata["new_summary"] = summary_text

        count = self._compression_log.get(req.session_id, 0) + 1
        self._compression_log[req.session_id] = count
        logger.info(
            f"[SummarizationMiddleware] Compressed {len(to_compress)} messages "
            f"for session {req.session_id} (compression #{count})"
        )

    def after_hook(self, resp: ModelResponse) -> None:
        """
        Post-model hook: persists the updated summary to the database if a
        session_updater was provided.
        """
        if not resp.request.metadata.get("summarized"):
            return

        new_summary = resp.request.metadata.get("new_summary", "")
        session_id = resp.request.session_id

        if self.session_updater and new_summary:
            try:
                self.session_updater(session_id, new_summary)
                logger.info(
                    f"[SummarizationMiddleware] Persisted summary for session {session_id}"
                )
            except Exception as e:
                logger.error(
                    f"[SummarizationMiddleware] Failed to persist summary: {e}"
                )

    def register(self) -> "SummarizationMiddleware":
        """
        Registers this middleware's before/after hooks into the global hook registry.
        Returns self for chaining.
        """
        before_model(self.before_hook)
        after_model(self.after_hook)
        logger.info(
            f"[SummarizationMiddleware] Registered (max_messages={self.max_messages}, "
            f"keep_last={self.keep_last})"
        )
        return self

    def manual_summarize(
        self,
        session_id: str,
        messages: List[BaseMessage],
        existing_summary: str = "",
    ) -> str:
        """
        Manually triggers summarization for a session's messages.
        Used by the POST /api/sessions/{id}/summarize endpoint.
        """
        if self.llm_client and not getattr(self.llm_client, "is_mock", False):
            return self._llm_summary(messages, existing_summary)
        return self._extractive_summary(messages, existing_summary)

    def get_stats(self) -> Dict[str, Any]:
        """Returns compression stats for all sessions."""
        return {
            "max_messages": self.max_messages,
            "keep_last": self.keep_last,
            "sessions_compressed": dict(self._compression_log),
            "total_compressions": sum(self._compression_log.values()),
        }


# ---------------------------------------------------------------------------
# DynamicPrompt
# ---------------------------------------------------------------------------

PromptPlugin = Callable[[Dict[str, Any]], Optional[str]]
"""
A PromptPlugin is a callable that receives session context and returns
an optional string to append to the system prompt.

Context keys available:
  session_id, memory_type, memory_backend, memory_k, summary,
  user_name (extracted from history), order_id (extracted from history),
  message_count, session_summary
"""


class DynamicPrompt:
    """
    Builds system prompts dynamically at call time based on session context.

    Core prompt is always included.  Additional sections are appended by:
      - Built-in context extractors (user name, order IDs, memory strategy info).
      - Custom PromptPlugins registered via .add_plugin().

    Usage::

        dp = DynamicPrompt(base_prompt="You are a support agent for CloudShop.")
        dp.add_plugin(my_plugin)

        system_msg = dp.build(
            session_id="abc",
            memory_type="buffer",
            memory_backend="sql",
            messages=history_messages,
        )
    """

    def __init__(self, base_prompt: str = ""):
        self.base_prompt = base_prompt
        self._plugins: List[PromptPlugin] = []
        self._lock = threading.Lock()

    def add_plugin(self, plugin: PromptPlugin) -> "DynamicPrompt":
        """Registers a PromptPlugin callable. Returns self for chaining."""
        with self._lock:
            self._plugins.append(plugin)
        return self

    def _extract_context(
        self,
        messages: List[BaseMessage],
        session_id: str,
        memory_type: str,
        memory_backend: str,
        memory_k: int = 4,
        session_summary: str = "",
    ) -> Dict[str, Any]:
        """Extracts structured context from the message history."""
        user_name: Optional[str] = None
        order_id: Optional[str] = None

        for msg in messages:
            if not isinstance(msg, HumanMessage):
                continue
            content = msg.content.lower()
            # Extract name
            for pattern in ("my name is ", "i'm ", "i am "):
                if pattern in content:
                    candidate = content.split(pattern)[1].split()[0]
                    candidate = re.sub(r"[^a-z]", "", candidate).title()
                    if len(candidate) > 1:
                        user_name = candidate
                        break
            # Extract order ID
            m = re.search(r"(order\s*#?\d+|#\d{3,})", content)
            if m:
                order_id = m.group(1).upper()

        return {
            "session_id": session_id,
            "memory_type": memory_type,
            "memory_backend": memory_backend,
            "memory_k": memory_k,
            "session_summary": session_summary,
            "user_name": user_name,
            "order_id": order_id,
            "message_count": len([m for m in messages if not isinstance(m, SystemMessage)]),
        }

    def build(
        self,
        session_id: str,
        memory_type: str,
        memory_backend: str,
        messages: List[BaseMessage],
        memory_k: int = 4,
        session_summary: str = "",
    ) -> str:
        """
        Builds a complete system prompt string.

        Returns a prompt that includes:
          - The base prompt
          - Memory strategy context
          - User name / order ID if known from history
          - Any extra sections from registered plugins
        """
        ctx = self._extract_context(
            messages, session_id, memory_type, memory_backend, memory_k, session_summary
        )

        sections: List[str] = [self.base_prompt.strip()]

        # --- Memory strategy context ---
        memory_hints = {
            "buffer": "You have access to the full conversation history.",
            "window": f"You have access to the last {memory_k} messages only.",
            "summary": (
                f"The conversation has been summarized. Summary: {session_summary}"
                if session_summary else "This is the start of the conversation."
            ),
        }
        hint = memory_hints.get(memory_type)
        if hint:
            sections.append(f"Memory context: {hint}")

        # --- User personalisation ---
        if ctx["user_name"]:
            sections.append(
                f"The customer's name is {ctx['user_name']}. Address them by name."
            )
        if ctx["order_id"]:
            sections.append(
                f"The customer has mentioned order {ctx['order_id']} in this conversation."
            )

        # --- Run plugins ---
        with self._lock:
            plugins = list(self._plugins)
        for plugin in plugins:
            try:
                extra = plugin(ctx)
                if extra:
                    sections.append(extra.strip())
            except Exception as e:
                logger.warning(f"[DynamicPrompt] Plugin {plugin.__name__} failed: {e}")

        return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# RuntimeMemoryAccessor
# ---------------------------------------------------------------------------

# Type variables for context and state
ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")

@dataclass
class CustomContext:
    """
    Context containing environmental data such as session parameters,
    databases, APIs, and client configurations.
    """
    session_id: str
    memory_backend: str
    memory_type: str
    db_connection_string: Optional[str] = None
    llm_provider: str = "mock"
    additional_env: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomState:
    """
    State representing the runtime state of a specific tool invocation,
    such as memory message history, status, and running summary metadata.
    """
    messages: List[BaseMessage] = field(default_factory=list)
    summary: str = ""
    user_name: Optional[str] = None
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRuntime(Generic[ContextT, StateT]):
    """
    ToolRuntime executes and coordinates tool or model behaviors
    under a given CustomContext and CustomState.
    """
    def __init__(self, context: ContextT, state: StateT):
        self.context = context
        self.state = state

    def update_state(self, **kwargs) -> None:
        """Update runtime state parameters in-place."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                self.state.metadata[key] = value

    def get_runtime_summary(self) -> str:
        """Helper to get summary from runtime state."""
        return self.state.summary


# ---------------------------------------------------------------------------
# Module-level singletons — initialized at import time
# ---------------------------------------------------------------------------

# The global SummarizationMiddleware instance.
# Call summarization_middleware.register() after the LLM is resolved in main.py
# so it has access to the LLM for LLM-based summaries.
summarization_middleware: Optional[SummarizationMiddleware] = None

# The global DynamicPrompt instance with the default CloudShop base prompt.
dynamic_prompt = DynamicPrompt(
    base_prompt=(
        "You are an empathetic, efficient Customer Support AI assistant for 'CloudShop', "
        "an e-commerce platform. Your goal is to help customers resolve issues with shipping, "
        "returns, product catalogs, and accounts. "
        "Keep responses professional, concise, and helpful. "
        "Always refer to past messages if relevant."
    )
)

# The global runtime instance parameterised by CustomContext and CustomState.
runtime: Optional[ToolRuntime[CustomContext, CustomState]] = None


def init_middleware(llm_client: Any, session_updater: Optional[Callable] = None) -> None:
    """
    Initialises the SummarizationMiddleware singleton and registers its hooks.
    Call this once at application startup after the LLM is resolved.

    Args:
        llm_client:      The resolved LLM (used for LLM-based summaries).
        session_updater: Optional callable(session_id, summary) to persist summaries.
    """
    global summarization_middleware
    summarization_middleware = SummarizationMiddleware(
        max_messages=10,
        keep_last=4,
        llm_client=llm_client,
        session_updater=session_updater,
    )
    summarization_middleware.register()
    logger.info("[middleware] init_middleware() complete — all hooks registered.")
