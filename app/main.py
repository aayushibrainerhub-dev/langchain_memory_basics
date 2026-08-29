import os
import re
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Import local database and memory modules
from app.database import init_db, get_db, ChatSession, get_connection_string, get_engine
from app.memory import (
    # SQL backend (existing)
    get_message_history,
    get_effective_memory,
    generate_summary_update,
    format_messages,
    # InMemoryChatMessageHistory backend
    get_in_memory_history,
    clear_in_memory_history,
    clear_graph_history,
    # LangGraph checkpointer factory
    get_checkpointer,
    # LangGraph graph builder
    build_graph,
    # PostgresSaver initializer (called on startup)
    get_postgres_saver,
)

# Load system configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()

# Valid memory backends
VALID_MEMORY_BACKENDS = {"sql", "in_memory", "memory_saver", "postgres_saver"}

def update_session_summary_in_db(session_id: str, new_summary: str):
    """Callback for SummarizationMiddleware to persist summaries back to DB."""
    from sqlalchemy import text
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE chat_sessions SET summary = :summary WHERE session_id = :session_id"),
                {"summary": new_summary, "session_id": session_id}
            )
            conn.commit()
        logger.info(f"Successfully updated summary for session {session_id} in DB.")
    except Exception as e:
        logger.error(f"Error persisting session summary: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize application database tables (chat_sessions, etc.)
    init_db()

    # 2. Pre-initialize PostgresSaver and create its checkpoint schema.
    #    This runs setup() once so the tables exist before any request arrives.
    try:
        get_postgres_saver()
        logger.info("PostgresSaver initialized successfully on startup.")
    except Exception as e:
        logger.warning(
            f"PostgresSaver could not be initialized on startup: {e}. "
            "Sessions using 'postgres_saver' backend will fail until the DB is reachable."
        )

    # 3. Initialize the SummarizationMiddleware singleton and register its hooks.
    try:
        llm = get_llm()
        from app.middleware import init_middleware
        init_middleware(llm, session_updater=update_session_summary_in_db)
        logger.info("SummarizationMiddleware hooks successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize SummarizationMiddleware: {e}")

    yield
    # Cleanup on shutdown (if any)


app = FastAPI(
    title="Customer Support Chatbot with Memory",
    description=(
        "A FastAPI & LangChain/LangGraph application demonstrating all four memory backends:\n"
        "- **sql**: SQLChatMessageHistory (langchain_community) — persistent PostgreSQL messages\n"
        "- **in_memory**: InMemoryChatMessageHistory (langchain_core) — RAM only\n"
        "- **memory_saver**: MemorySaver / InMemorySaver (LangGraph) — RAM only\n"
        "- **postgres_saver**: PostgresSaver (LangGraph) — persistent LangGraph checkpoints\n\n"
        "Memory *strategies* (buffer/window/summary) apply to the `sql` backend only."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LLM implementations
# ---------------------------------------------------------------------------

# Mock LLM for local evaluation when no API key is specified
class MockLLM:
    is_mock = True
    def invoke(self, prompt) -> Any:
        return _MockResponse("This is a mock response from the support agent.")
    def predict(self, prompt: str) -> str:
        return "This is a mock response from the support agent."


class _MockResponse:
    def __init__(self, content: str):
        self.content = content
    def __str__(self):
        return self.content


# Groq LLM client using direct HTTP (no langchain-groq package required)
class GroqLLM:
    is_mock = False

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.6):
        import requests as _requests
        self._requests = _requests
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._url = "https://api.groq.com/openai/v1/chat/completions"

    def _call_api(self, messages: list) -> str:
        """Calls the Groq REST API and returns the assistant reply text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        response = self._requests.post(self._url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def invoke(self, prompt) -> "_GroqResponse":
        """Accepts either a string or a list of LangChain message objects."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            role_map = {
                HumanMessage: "user",
                AIMessage: "assistant",
                SystemMessage: "system",
            }
            messages = []
            for m in prompt:
                role = role_map.get(type(m), "user")
                messages.append({"role": role, "content": m.content})
        content = self._call_api(messages)
        return _GroqResponse(content)

    def predict(self, prompt: str) -> str:
        return self._call_api([{"role": "user", "content": prompt}])


class _GroqResponse:
    """Thin wrapper so .content access works like a LangChain AIMessage."""
    def __init__(self, content: str):
        self.content = content
    def __str__(self):
        return self.content


def get_llm():
    """Resolves and configures the LLM provider selected in the environment."""
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    logger.info(f"Resolving LLM Provider: {provider}")

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is missing. Defaulting to mock.")
            return MockLLM()
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(temperature=0.6)
        except ImportError:
            logger.error("langchain-openai package not installed. Run pip install langchain-openai.")
            return MockLLM()

    elif provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY is missing. Defaulting to mock.")
            return MockLLM()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
        except ImportError:
            logger.error("langchain-google-genai package not installed.")
            return MockLLM()

    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY is missing. Defaulting to mock.")
            return MockLLM()
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0.6)
        except ImportError:
            logger.error("langchain-anthropic package not installed.")
            return MockLLM()

    elif provider == "groq":
        api_key = os.getenv("GROQ_CLOUD_API_KEY", "").strip('"').strip("'")
        if not api_key:
            logger.warning("GROQ_CLOUD_API_KEY is missing. Defaulting to mock.")
            return MockLLM()
        logger.info("Using Groq LLM (llama-3.3-70b-versatile) via direct HTTP.")
        return GroqLLM(api_key=api_key)

    return MockLLM()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an empathetic, efficient Customer Support AI assistant for 'CloudShop', an e-commerce platform. "
    "Your goal is to help customers resolve issues with shipping, returns, product catalogs, and accounts. "
    "Keep responses professional, concise, and helpful. Always refer to past messages if relevant."
)


def generate_mock_support_response(user_message: str, effective_messages: List[Dict[str, str]]) -> str:
    """
    Rule-based mock chatbot to provide an interactive and responsive session memory simulation
    without requiring external APIs. It tracks context details like Name and Order ID from history.
    """
    msg_lower = user_message.lower()
    user_name = None
    order_id = None

    # 1. Parse historical context to demonstrate memory retrieval
    for msg in effective_messages:
        content = msg["content"].lower()
        if msg["role"] == "user":
            if "my name is " in content:
                parts = content.split("my name is ")
                if len(parts) > 1:
                    user_name = parts[1].split()[0].replace(".", "").replace(",", "").title()
            elif "i'm " in content:
                parts = content.split("i'm ")
                if len(parts) > 1:
                    user_name = parts[1].split()[0].replace(".", "").replace(",", "").title()

            # Search order IDs like #12345 or ORD123
            match = re.search(r'(order\s+#?\d+|#\d+)', content)
            if match:
                order_id = match.group(1).upper()

    # 2. Parse current message
    if "my name is " in msg_lower:
        parts = msg_lower.split("my name is ")
        if len(parts) > 1:
            user_name = parts[1].split()[0].replace(".", "").replace(",", "").title()
    elif "i'm " in msg_lower:
        parts = msg_lower.split("i'm ")
        if len(parts) > 1:
            user_name = parts[1].split()[0].replace(".", "").replace(",", "").title()

    match = re.search(r'(order\s+#?\d+|#\d+)', msg_lower)
    if match:
        order_id = match.group(1).upper()

    greeting = f"Hello {user_name}! " if user_name else "Hello! "

    # 3. Formulate responses
    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower:
        return f"{greeting}Welcome to CloudShop Customer Support. How can I help you today? (Note: Mock LLM mode is active)"

    elif "name" in msg_lower and ("what" in msg_lower or "do you know" in msg_lower or "remember" in msg_lower):
        if user_name:
            return f"Yes, I remember you! Your name is {user_name}. What can I do for you?"
        else:
            return "I don't believe you have shared your name yet. What should I call you?"

    elif "order" in msg_lower and ("where" in msg_lower or "status" in msg_lower or "track" in msg_lower):
        if order_id:
            return f"I've pulled up the information for {order_id}. Your shipment is currently in transit and is scheduled to be delivered this Friday. Would you like me to send you the tracking link?"
        else:
            past_order = None
            for m in effective_messages:
                hist_match = re.search(r'(order\s+#?\d+|#\d+)', m["content"].lower())
                if hist_match:
                    past_order = hist_match.group(1).upper()
            if past_order:
                return f"Referring back to {past_order} that you mentioned earlier, it is being packed at our warehouse. I will update you as soon as it ships."
            return "I can certainly check your order status! Could you please share your Order ID (for example, #54321)?"

    elif "return" in msg_lower or "refund" in msg_lower:
        return "Our standard return policy is 30 days from delivery. You can download a prepaid shipping label by visiting the 'Returns' section of your account. Let me know if you need instructions!"

    elif "clear" in msg_lower or "reset" in msg_lower:
        return "To reset or clear this chat memory, please use the 'Clear History' button on the sidebar."

    else:
        ans = f"{greeting}Thanks for reaching out to CloudShop support. "
        if user_name:
            ans += f"I have you registered as {user_name}. "
        if order_id:
            ans += f"I am keeping track of your order {order_id}. "
        ans += "Is there anything else I can check for you? (Tip: Tell me 'My name is Sarah' or type 'Check order #4040' to test session persistence!)"
        return ans


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    name: str
    memory_type: str = "buffer"          # buffer, window, summary — applies to sql backend only
    memory_k: Optional[int] = 4
    memory_backend: str = "sql"          # sql | in_memory | memory_saver | postgres_saver


class MessageIn(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
def create_session(session_data: SessionCreate, db: Session = Depends(get_db)):
    """
    Creates a new chatbot conversation session.

    - **memory_backend**: controls WHERE state is stored:
      - `sql` — SQLChatMessageHistory (PostgreSQL, persistent)
      - `in_memory` — InMemoryChatMessageHistory (RAM, volatile)
      - `memory_saver` — MemorySaver / InMemorySaver (LangGraph, RAM, volatile)
      - `postgres_saver` — PostgresSaver (LangGraph, PostgreSQL, persistent)
    - **memory_type**: only applies to `sql` backend (buffer / window / summary).
    """
    if session_data.memory_type not in ["buffer", "window", "summary"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory_type. Choose from: 'buffer', 'window', 'summary'.",
        )

    if session_data.memory_backend not in VALID_MEMORY_BACKENDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory_backend. Choose from: {sorted(VALID_MEMORY_BACKENDS)}.",
        )

    session_id = str(uuid.uuid4())
    new_session = ChatSession(
        session_id=session_id,
        name=session_data.name,
        memory_type=session_data.memory_type,
        memory_k=session_data.memory_k,
        summary="",
        memory_backend=session_data.memory_backend,
        thread_id=session_id,   # thread_id == session_id by default
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Retrieves all conversations in the database."""
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    return sessions


@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str, db: Session = Depends(get_db)):
    """Gets details and memory context for a specific session."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    effective_data = get_effective_memory(session)

    # For sql backend, also expose raw SQL history
    raw_history = []
    if session.memory_backend == "sql":
        history = get_message_history(session_id)
        raw_history = format_messages(history.messages)
    elif session.memory_backend == "in_memory":
        history = get_in_memory_history(session_id)
        raw_history = format_messages(history.messages)

    return {
        "metadata": {
            "session_id": session.session_id,
            "name": session.name,
            "memory_type": session.memory_type,
            "memory_k": session.memory_k,
            "summary": session.summary,
            "memory_backend": session.memory_backend,
            "thread_id": session.thread_id,
        },
        "raw_history": raw_history,
        "effective_memory": effective_data,
    }


@app.post("/api/sessions/{session_id}/message")
def send_message(session_id: str, payload: MessageIn, db: Session = Depends(get_db)):
    """
    Sends a message to the chatbot.

    Routes to the correct memory backend based on the session's `memory_backend` field:
    - **sql**: loads history from SQLChatMessageHistory, builds prompt manually.
    - **in_memory**: loads history from InMemoryChatMessageHistory, builds prompt manually.
    - **memory_saver / postgres_saver**: invokes a LangGraph compiled graph; the
      checkpointer automatically loads and saves all state.
    """
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    llm = get_llm()
    backend = session.memory_backend or "sql"
    thread_id = session.thread_id or session_id

    # ----------------------------------------------------------------
    # PATH A: LangGraph backends (in_memory / memory_saver / postgres_saver)
    # ----------------------------------------------------------------
    if backend in ("in_memory", "memory_saver", "postgres_saver"):
        # For in_memory backend, we use InMemoryChatMessageHistory to store messages
        # AND InMemorySaver as the LangGraph checkpointer.
        # This demonstrates that both classes are used together in a real workflow.

        checkpointer = get_checkpointer(backend)
        graph = build_graph(llm, checkpointer, system_prompt=SYSTEM_PROMPT)

        config = {"configurable": {"thread_id": thread_id}}

        from langchain_core.messages import HumanMessage as HM

        if isinstance(llm, MockLLM):
            # For mock LLM, generate a rule-based response and store manually
            # via InMemoryChatMessageHistory to keep history visible.
            if backend == "in_memory":
                history = get_in_memory_history(session_id)
                prior_messages = format_messages(history.messages)
                response_text = generate_mock_support_response(user_text, prior_messages)
                history.add_user_message(user_text)
                history.add_ai_message(response_text)
                raw_history = format_messages(history.messages)
            else:
                # memory_saver / postgres_saver — get snapshot from checkpointer
                state = graph.get_state(config)
                prior = state.values.get("messages", []) if state else []
                prior_messages = format_messages(prior)
                response_text = generate_mock_support_response(user_text, prior_messages)
                # Inject as a real graph invocation so checkpointer saves it
                from langchain_core.messages import AIMessage as AM
                graph.invoke(
                    {"messages": [HM(content=user_text), AM(content=response_text)]},
                    config=config,
                )
                raw_history = []
        else:
            # Real LLM: invoke the LangGraph graph normally
            result = graph.invoke({"messages": [HM(content=user_text)]}, config=config)
            last_msg = result["messages"][-1]
            response_text = getattr(last_msg, "content", str(last_msg))
            raw_history = format_messages(result["messages"])

        updated_memory = get_effective_memory(session)
        return {
            "response": response_text,
            "effective_memory": updated_memory,
            "raw_history": raw_history,
            "summary": session.summary,
            "memory_backend": backend,
        }

    # ----------------------------------------------------------------
    # PATH B: SQL backend (updated to use DynamicPrompt + invoke_with_hooks)
    # ----------------------------------------------------------------
    memory_data = get_effective_memory(session)
    effective_messages = memory_data["messages"]

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages_to_send = []
    for msg in effective_messages:
        if msg["role"] == "user":
            messages_to_send.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_to_send.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            messages_to_send.append(SystemMessage(content=msg["content"]))

    messages_to_send.append(HumanMessage(content=user_text))

    # 1. Build prompt dynamically
    from app.middleware import dynamic_prompt, ModelRequest, invoke_with_hooks
    dynamic_sys_prompt = dynamic_prompt.build(
        session_id=session_id,
        memory_type=session.memory_type or "buffer",
        memory_backend=backend,
        messages=messages_to_send,
        memory_k=session.memory_k or 4,
        session_summary=session.summary or "",
    )
    # Prepend the generated dynamic system prompt
    messages_to_send.insert(0, SystemMessage(content=dynamic_sys_prompt))

    # 2. Build the ModelRequest object
    req = ModelRequest(
        messages=messages_to_send,
        session_id=session_id,
        memory_backend=backend,
        memory_type=session.memory_type or "buffer",
        metadata={"session_summary": session.summary or ""}
    )

    # Invoke the model
    if isinstance(llm, MockLLM):
        response_text = generate_mock_support_response(user_text, effective_messages)
    else:
        try:
            # Use runtime hooks pipeline to trigger @before_model and @after_model hooks
            resp_envelope = invoke_with_hooks(llm, req)
            response_text = resp_envelope.content
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            response_text = f"My apologies, I ran into an error communicating with our main LLM models. Details: {str(e)}"

    # Store conversation turn in PostgreSQL
    history = get_message_history(session_id)
    history.add_user_message(user_text)
    history.add_ai_message(response_text)

    # If using Summary memory, trigger summary refresh
    if session.memory_type == "summary":
        new_summary = generate_summary_update(session.summary, user_text, response_text, llm)
        session.summary = new_summary
        db.add(session)
        db.commit()
        db.refresh(session)

    updated_memory_data = get_effective_memory(session)
    return {
        "response": response_text,
        "effective_memory": updated_memory_data,
        "raw_history": format_messages(history.messages),
        "summary": session.summary,
        "memory_backend": backend,
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Deletes a chat session and clears its associated history across all backends."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    backend = session.memory_backend or "sql"

    # Clear SQL message history (always attempt — harmless if not used)
    if backend == "sql":
        history = get_message_history(session_id)
        try:
            history.clear()
        except Exception as e:
            logger.warning(f"Error clearing message_store for session {session_id}: {e}")

    # Clear InMemoryChatMessageHistory store
    if backend == "in_memory":
        clear_in_memory_history(session_id)

    # Clear GraphRunner in-process history (covers memory_saver and postgres_saver too)
    if backend in ("in_memory", "memory_saver", "postgres_saver"):
        clear_graph_history(session_id)

    # Note: LangGraph checkpointer state (memory_saver / postgres_saver)
    # is identified by thread_id; we do not expose a delete API for it here
    # since MemorySaver has no persistence and PostgresSaver checkpoints can
    # be left to expire or cleaned via the checkpoint tables directly.

    db.delete(session)
    db.commit()
    return {
        "message": "Session and message history deleted successfully.",
        "memory_backend": backend,
    }


# ---------------------------------------------------------------------------
# Summarization APIs
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/summarize")
def run_manual_summarization(session_id: str, db: Session = Depends(get_db)):
    """Manually triggers SummarizationMiddleware to summarize conversation history."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    from app.middleware import summarization_middleware
    if not summarization_middleware:
        raise HTTPException(status_code=503, detail="SummarizationMiddleware is not initialized.")
    
    backend = session.memory_backend or "sql"
    from app.memory import get_message_history, get_in_memory_history, get_checkpointer
    
    lc_messages = []
    if backend == "sql":
        lc_messages = get_message_history(session_id).messages
    elif backend == "in_memory":
        lc_messages = get_in_memory_history(session_id).messages
    elif backend in ("memory_saver", "postgres_saver"):
        try:
            cp = get_checkpointer(backend)
            config = {"configurable": {"thread_id": session.thread_id or session_id}}
            state = cp.get(config)
            if state and state.get("channel_values"):
                lc_messages = state["channel_values"].get("messages", [])
        except Exception as e:
            logger.error(f"Error fetching checkpoint messages: {e}")

    if not lc_messages:
        return {"message": "No messages to summarize."}
            
    # Run summarization
    new_summary = summarization_middleware.manual_summarize(
        session_id=session_id,
        messages=lc_messages,
        existing_summary=session.summary or ""
    )
    
    # Update DB
    session.summary = new_summary
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {
        "session_id": session_id,
        "new_summary": new_summary
    }


@app.get("/api/config")
def get_config():
    """Retrieves basic application configuration."""
    db_string = get_connection_string()
    db_masked = db_string
    if "://" in db_string:
        prefix, rest = db_string.split("://", 1)
        if "@" in rest:
            auth, host = rest.split("@", 1)
            db_masked = f"{prefix}://*****@{host}"

    return {
        "llm_provider": LLM_PROVIDER,
        "database_url_resolved": db_masked,
        "is_postgres": db_string.startswith("postgresql"),
        "supported_memory_backends": sorted(VALID_MEMORY_BACKENDS),
        "supported_memory_types": ["buffer", "window", "summary"],
    }


# ---------------------------------------------------------------------------
# Serving Chat Front-End UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the central chat console HTML file."""
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Templates folder index.html missing!</h1>", status_code=500)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
