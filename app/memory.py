"""
memory.py — All LangChain / LangGraph memory backends in one place.

Supported backends (controlled by ChatSession.memory_backend):
  "sql"            → SQLChatMessageHistory   (langchain_community) — persistent via Postgres
  "in_memory"      → InMemoryChatMessageHistory (langchain_core)   — RAM only, lost on restart
  "memory_saver"   → MemorySaver (LangGraph)                      — RAM only, lost on restart
  "postgres_saver" → PostgresSaver (LangGraph checkpoint-postgres) — persistent via Postgres

Memory *strategies* (memory_type field) — Buffer / Window / Summary — apply to the "sql"
backend only, where raw messages are loaded and optionally trimmed/summarised before being
injected into the prompt.  For LangGraph backends the checkpointer handles state natively.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from app.database import get_connection_string, ChatSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons so in-process state survives across HTTP requests
# ---------------------------------------------------------------------------

# InMemoryChatMessageHistory store: { session_id -> InMemoryChatMessageHistory }
_in_memory_store: Dict[str, InMemoryChatMessageHistory] = {}
_in_memory_lock = threading.Lock()

# LangGraph InMemorySaver singleton (used for "in_memory" backend)
_in_memory_saver_instance = None
_in_memory_saver_lock = threading.Lock()

# LangGraph MemorySaver singleton (used for "memory_saver" / older-alias backend)
_memory_saver_instance = None
_memory_saver_lock = threading.Lock()

# LangGraph PostgresSaver singleton
_postgres_saver_instance = None
_postgres_saver_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 1. SQLChatMessageHistory  (existing, unchanged) DB storage
# ---------------------------------------------------------------------------

def get_message_history(session_id: str) -> SQLChatMessageHistory:
    """
    Returns a SQL-backed chat message history for the given session.
    Messages are stored in the 'message_store' table in PostgreSQL.
    This is the classic LangChain persistence approach.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=get_connection_string(),
        table_name="message_store",
    )


# ---------------------------------------------------------------------------
# 2. InMemoryChatMessageHistory  (langchain_core) Store in RAM memory
# ---------------------------------------------------------------------------

def get_in_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns (or creates) an InMemoryChatMessageHistory for the session.
    State lives in-process only — it is lost on server restart.

    A module-level dict keyed by session_id ensures the same history object
    is returned on every request within the same process lifetime.
    """
    with _in_memory_lock:
        if session_id not in _in_memory_store:
            _in_memory_store[session_id] = InMemoryChatMessageHistory()
            logger.info(f"[InMemoryChatMessageHistory] Created new in-memory store for session {session_id}")
        return _in_memory_store[session_id]


def clear_in_memory_history(session_id: str) -> None:
    """Removes the in-memory history for a session (used on delete)."""
    with _in_memory_lock:
        if session_id in _in_memory_store:
            del _in_memory_store[session_id]
            logger.info(f"[InMemoryChatMessageHistory] Cleared in-memory store for session {session_id}")


def clear_graph_history(session_id: str) -> None:
    """Removes the GraphRunner in-process history for a session (used on delete)."""
    with _graph_history_lock:
        if session_id in _graph_history_store:
            del _graph_history_store[session_id]
            logger.info(f"[GraphRunner] Cleared graph history for session {session_id}")


# ---------------------------------------------------------------------------
# 3. LangGraph MemorySaver  (in-memory checkpointer — RAM only)
# ---------------------------------------------------------------------------

def get_memory_saver():
    """
    Returns the module-level singleton LangGraph MemorySaver.
    MemorySaver is the current name; in older LangGraph versions it was called InMemorySaver.
    Both resolve to the same class and are imported here explicitly for clarity.

    State lives in-process only — lost on server restart.
    """
    global _memory_saver_instance
    with _memory_saver_lock:
        if _memory_saver_instance is None:
            # Import both names to demonstrate they are accessible.
            # InMemorySaver is the newer canonical name; MemorySaver is the older alias.
            try:
                from langgraph.checkpoint.memory import InMemorySaver, MemorySaver  # noqa: F401
                _memory_saver_instance = MemorySaver()
                logger.info(f"[MemorySaver] Initialized — class: {type(_memory_saver_instance).__name__}")
            except ImportError as exc:
                raise RuntimeError(
                    "langgraph package is required for MemorySaver. "
                    "Run: pip install langgraph"
                ) from exc
        return _memory_saver_instance


# ---------------------------------------------------------------------------
# 4. LangGraph PostgresSaver  (Postgres-backed checkpointer)
# ---------------------------------------------------------------------------

def get_postgres_saver():
    """
    Returns the module-level singleton LangGraph PostgresSaver.

    PostgresSaver.from_conn_string is a contextmanager, so it cannot be stored
    as a plain singleton.  Instead we open a persistent psycopg3 connection
    manually and pass it to PostgresSaver directly, keeping the connection alive
    for the entire process lifetime.

    The checkpointer schema is created automatically via .setup() on first call.
    State is durable — survives server restarts.
    """
    global _postgres_saver_instance
    with _postgres_saver_lock:
        if _postgres_saver_instance is None:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "langgraph-checkpoint-postgres and psycopg are required for PostgresSaver. "
                    "Run: pip install langgraph-checkpoint-postgres psycopg"
                ) from exc

            conn_string = get_connection_string()
            # psycopg3 expects postgresql:// (not postgresql+psycopg2://)
            pg_url = conn_string.replace("postgresql+psycopg2://", "postgresql://")

            logger.info("[PostgresSaver] Opening persistent psycopg3 connection...")
            # Open a persistent connection (autocommit required by LangGraph)
            _pg_conn = psycopg.connect(
                pg_url,
                autocommit=True,
                prepare_threshold=0,
                row_factory=dict_row,
            )
            saver = PostgresSaver(_pg_conn)
            logger.info("[PostgresSaver] Running setup() to create checkpoint tables...")
            saver.setup()  # Creates langgraph_checkpoint tables if they don't exist
            _postgres_saver_instance = saver
            logger.info("[PostgresSaver] Ready.")
        return _postgres_saver_instance


# ---------------------------------------------------------------------------
# Checkpointer factory — maps memory_backend string → LangGraph checkpointer
# ---------------------------------------------------------------------------

def get_checkpointer(memory_backend: str):
    """
    Returns the correct LangGraph checkpointer for the given backend name.

    Args:
        memory_backend: one of "in_memory", "memory_saver", "postgres_saver"

    Returns:
        A LangGraph BaseCheckpointSaver instance.

    Raises:
        ValueError if the backend is unknown or not a LangGraph backend.
    """
    if memory_backend == "in_memory":
        # InMemorySaver — singleton so state persists across requests within the process
        global _in_memory_saver_instance
        with _in_memory_saver_lock:
            if _in_memory_saver_instance is None:
                from langgraph.checkpoint.memory import InMemorySaver
                _in_memory_saver_instance = InMemorySaver()
                print("_in_memory_saver_instance---------", _in_memory_saver_instance)
                logger.info("[get_checkpointer] Created InMemorySaver singleton")
            return _in_memory_saver_instance

    elif memory_backend == "memory_saver":
        # MemorySaver is the older / alias name; both point to the same class
        return get_memory_saver()

    elif memory_backend == "postgres_saver":
        return get_postgres_saver()

    else:
        raise ValueError(
            f"Unknown LangGraph memory_backend: '{memory_backend}'. "
            "Valid LangGraph backends: 'in_memory', 'memory_saver', 'postgres_saver'."
        )


# ---------------------------------------------------------------------------
# In-process history store for GraphRunner (version-agnostic alternative to
# the raw checkpointer get/put API, which changed between LangGraph versions).
# ---------------------------------------------------------------------------

# { thread_id -> List[BaseMessage] }  — persists across HTTP requests within
# the same process lifetime (same guarantee as InMemorySaver).
_graph_history_store: Dict[str, List] = {}
_graph_history_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Graph-style callable (checkpointer-backed, no LangGraph)
# ---------------------------------------------------------------------------

def build_graph(llm, checkpointer, system_prompt: str = ""):
    """
    Returns a plain callable that mimics a compiled graph interface.
    History is stored in the module-level _graph_history_store dict (keyed by
    thread_id) instead of calling the raw checkpointer get/put API directly.
    This avoids version-specific breakage in InMemorySaver's internal contract.

    Args:
        llm:            Any LangChain-compatible chat model (must support .invoke(messages)).
        checkpointer:   A LangGraph checkpointer singleton (kept alive for identity/lifecycle).
        system_prompt:  Optional system message prepended to every invocation.

    Returns:
        A callable with signature: invoke(inputs, config) -> {"messages": [...]}
    """
    from langchain_core.messages import AIMessage
    from app.middleware import invoke_with_hooks, ModelRequest

    # Determine backend label once at build time
    backend_name = "memory_saver"
    if checkpointer and hasattr(checkpointer, "__class__"):
        cls_name = checkpointer.__class__.__name__.lower()
        if "postgres" in cls_name:
            backend_name = "postgres_saver"
        elif "inmemory" in cls_name or "memory" in cls_name:
            backend_name = "in_memory"

    class GraphRunner:
        """
        Wraps the inner invoke function as an object so callers can use
        graph.invoke() and graph.get_state() — matching the LangGraph compiled
        graph interface expected by main.py.

        History is stored in the module-level _graph_history_store dict so it
        survives across multiple HTTP requests in the same process.
        """

        def invoke(self, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Loads history from the in-process store, runs the LLM with middleware
            hooks, appends the response, persists the updated history, and returns
            the full updated message list.

            Args:
                inputs: dict with key "messages" containing the new user message(s).
                config: dict with optional key "configurable" -> {"thread_id": str}.

            Returns:
                {"messages": <full updated message list including AI reply>}
            """
            config = config or {}
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "unknown_thread")

            # --- Load existing history from in-process store ---
            with _graph_history_lock:
                history: List[BaseMessage] = list(_graph_history_store.get(thread_id, []))

            # --- Merge new incoming messages ---
            new_messages: List[BaseMessage] = inputs.get("messages", [])
            messages = history + new_messages

            # --- Prepend system prompt if needed ---
            if system_prompt:
                if not messages or not isinstance(messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + messages

            # --- Run before/after model hooks and call the LLM ---
            req = ModelRequest(
                messages=messages,
                session_id=thread_id,
                memory_backend=backend_name,
                memory_type="buffer",
            )
            resp = invoke_with_hooks(llm, req)
            ai_message = AIMessage(content=resp.content)

            # --- Persist updated history (conversation only, no system prompt) ---
            updated_messages = history + new_messages + [ai_message]
            with _graph_history_lock:
                _graph_history_store[thread_id] = updated_messages
            logger.debug(
                f"[GraphRunner] thread={thread_id} stored {len(updated_messages)} messages"
            )

            return {"messages": updated_messages}

        def get_state(self, config: Dict[str, Any] = None):
            """
            Returns a simple state snapshot for the given thread_id, mimicking
            the LangGraph compiled graph .get_state() interface.

            Returns an object with a .values dict containing "messages".
            """
            config = config or {}
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "unknown_thread")

            with _graph_history_lock:
                msgs = list(_graph_history_store.get(thread_id, []))

            class _State:
                def __init__(self, messages):
                    self.values = {"messages": messages}

            return _State(msgs)

    return GraphRunner()



# ---------------------------------------------------------------------------
# Helpers shared by all backends
# ---------------------------------------------------------------------------

def format_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Converts LangChain BaseMessage objects to plain dicts for JSON serialization."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = getattr(msg, "type", "unknown")

        formatted.append({"role": role, "content": msg.content})
    return formatted


def get_effective_memory(session: ChatSession) -> Dict[str, Any]:
    """
    Computes the effective memory context injected into the LLM prompt
    based on the session's memory_type strategy.

    Only applies to the "sql" backend.  For LangGraph backends the
    checkpointer manages state natively — this function returns a
    descriptive summary instead.
    """
    # ----------------------------------------------------------------
    # LangGraph backends — state is managed by the checkpointer
    # ----------------------------------------------------------------
    if session.memory_backend in ("in_memory", "memory_saver", "postgres_saver"):
        backend_descriptions = {
            "in_memory": (
                "InMemoryChatMessageHistory + InMemorySaver (LangGraph): "
                "Conversation state is held in RAM. Fast, but lost on server restart."
            ),
            "memory_saver": (
                "MemorySaver (LangGraph, older alias of InMemorySaver): "
                "Conversation state is held in RAM via the LangGraph checkpointer. Lost on restart."
            ),
            "postgres_saver": (
                "PostgresSaver (LangGraph): "
                "Conversation state is persisted as LangGraph checkpoints in PostgreSQL. "
                "Survives restarts."
            ),
        }
        return {
            "type": session.memory_backend,
            "messages": [],   # managed internally by LangGraph — not exposed here
            "description": backend_descriptions.get(session.memory_backend, ""),
            "note": "State is managed by the LangGraph checkpointer. Use /history endpoint to view messages.",
        }

    # ----------------------------------------------------------------
    # SQL backend — apply Buffer / Window / Summary strategy
    # ----------------------------------------------------------------
    history = get_message_history(session.session_id)
    raw_messages = history.messages

    if session.memory_type == "buffer":
        return {
            "type": "buffer",
            "messages": format_messages(raw_messages),
            "description": "Conversation Buffer Memory: All historical messages are included in the prompt context.",
        }

    elif session.memory_type == "window":
        k = session.memory_k or 4
        pruned_messages = raw_messages[-k:] if len(raw_messages) > k else raw_messages
        return {
            "type": "window",
            "k": k,
            "messages": format_messages(pruned_messages),
            "description": f"Conversation Buffer Window Memory: Only the last {k} messages are included in the prompt context.",
        }

    elif session.memory_type == "summary":
        summary_text = session.summary or "This is the start of the conversation."
        summary_message = [
            SystemMessage(content=f"System summary of the conversation so far: {summary_text}")
        ]
        return {
            "type": "summary",
            "summary": summary_text,
            "messages": format_messages(summary_message),
            "description": "Conversation Summary Memory: A running summary of the conversation is injected into the context rather than raw messages.",
        }

    else:
        return {
            "type": "none",
            "messages": [],
            "description": "No memory configuration.",
        }


def generate_summary_update(old_summary: str, user_msg: str, ai_msg: str, llm_client: Any = None) -> str:
    """Updates the conversation summary based on the old summary and the new message exchange."""
    if not old_summary:
        old_summary = "There was no previous conversation."

    prompt = (
        f"Progressively summarize the conversation. Focus on the core customer issues and solutions.\n"
        f"Current summary:\n{old_summary}\n\n"
        f"New interaction:\n"
        f"User: {user_msg}\n"
        f"AI: {ai_msg}\n\n"
        f"Create a new concise summary combining the new interaction with the current summary. "
        f"Do not write introductory text, write ONLY the new summary:"
    )

    # Check if we should use Mock summary logic
    is_mock = llm_client is None or getattr(llm_client, "is_mock", False)
    if is_mock:
        clean_user = user_msg[:40] + ("..." if len(user_msg) > 40 else "")
        clean_ai = ai_msg[:40] + ("..." if len(ai_msg) > 40 else "")

        if "no previous conversation" in old_summary.lower():
            return f"The customer started by asking about '{clean_user}', and the support agent responded '{clean_ai}'."
        else:
            parts = old_summary.split(".")
            if len(parts) > 3:
                old_summary = ". ".join(p.strip() for p in parts[-3:] if p.strip())
            return f"{old_summary}. Then, customer asked '{clean_user}' and agent said '{clean_ai}'"

    try:
        if hasattr(llm_client, "invoke"):
            response = llm_client.invoke(prompt)
            content = getattr(response, "content", str(response))
        elif hasattr(llm_client, "predict"):
            content = llm_client.predict(prompt)
        else:
            content = str(llm_client(prompt))

        return content.strip()
    except Exception as e:
        logger.error(f"Error generating summary using LLM: {e}")
        clean_user = user_msg[:30] + ("..." if len(user_msg) > 30 else "")
        clean_ai = ai_msg[:30] + ("..." if len(ai_msg) > 30 else "")
        return f"{old_summary} (Updated: User asked '{clean_user}', AI replied '{clean_ai}')"
