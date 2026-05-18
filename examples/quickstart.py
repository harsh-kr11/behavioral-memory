"""Minimal quickstart: plug behavioral memory into any agent.

This example shows the simplest integration path — retrieve traces
and generate a plan with any LangChain-compatible model.

Prerequisites:
    pip install behavioral-memory langchain-google-genai
    # PostgreSQL with pgvector running
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from behavioral_memory import PlanEngine, ToolRegistry, TraceStore
from behavioral_memory.tools.mock_tools import get_tool_schemas

# 1. Bring your own models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Initialize the memory store
store = TraceStore(
    embeddings=embeddings,
    connection_url="postgresql+psycopg://user:pass@localhost:5432/behavioral_memory",
)

# 3. Register available tools
registry = ToolRegistry()
registry.register_many(get_tool_schemas())

# 4. Create the plan engine
engine = PlanEngine(llm=llm, store=store, registry=registry)

# 5. Generate a plan
plan = engine.generate(query="Get total revenue and send a report to stakeholders")

print(f"Generated {len(plan.steps)} steps:")
for step in plan.steps:
    print(f"  {step.step_id}: {step.tool_name}({step.parameters})")
