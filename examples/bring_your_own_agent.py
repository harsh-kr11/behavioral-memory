"""Bring Your Own Agent: integrate behavioral memory into an existing agent.

This example shows how any agent framework (not just LangGraph) can
use the behavioral memory library for trace-backed planning.
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from behavioral_memory import (
    FeedbackPoller,
    GatekeeperPipeline,
    PlanEngine,
    ToolRegistry,
    TraceStore,
)
from behavioral_memory.core.config import Settings
from behavioral_memory.tools.mock_tools import get_tool_schemas

settings = Settings()

# --- Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
store = TraceStore(embeddings=embeddings, settings=settings)

registry = ToolRegistry()
registry.register_many(get_tool_schemas())

engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)
gatekeeper = GatekeeperPipeline(store=store, registry=registry, settings=settings)


# --- Your agent's inference logic ---
def my_agent_handle_query(user_query: str) -> dict:
    """Your agent calls the plan engine at inference time."""
    plan = engine.generate(query=user_query)
    # ... execute the plan with your agent's tool runtime ...
    return {"plan": plan, "steps": len(plan.steps)}


# --- Background: auto-learn from Langfuse feedback ---
def start_feedback_loop():
    """Spin up the feedback poller in a background thread."""
    poller = FeedbackPoller(settings=settings)
    poller.poll_loop(
        callback=lambda trace: gatekeeper.submit(trace),
        max_iterations=100,
    )


# --- Usage ---
if __name__ == "__main__":
    result = my_agent_handle_query("Build a revenue report pipeline and email stakeholders")
    print(f"Plan generated with {result['steps']} steps")
