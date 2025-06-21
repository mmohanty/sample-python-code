# Custom Multi-Step Agent using PlanAndExecuteAgentExecutor

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, PlanAndExecuteAgentExecutor, load_tools
from langchain.experimental.plan_and_execute.planners.chat_planner import ChatPlanner
from langchain.experimental.plan_and_execute.executors.agent_executor import load_agent_executor
from langchain_core.runnables import RunnableConfig

# Dummy tool implementations
def text_to_sql(query: str) -> str:
    if "fail" in query.lower():
        return ""
    return f"[SQL] {query}"

def vector_search(query: str) -> str:
    return f"[VECTOR] {query}"

# Register tools
tools = [
    Tool(
        name="text_to_sql",
        func=text_to_sql,
        description="Structured queries for SQL-like questions."
    ),
    Tool(
        name="vector_search",
        func=vector_search,
        description="Semantic search over unstructured documents."
    ),
]

# Setup LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define Planner and Executor
planner = ChatPlanner.from_llm(llm)
executor = load_agent_executor(llm=llm, tools=tools, verbose=True)

# Create Plan-and-Execute agent
agent = PlanAndExecuteAgentExecutor(
    planner=planner,
    executor=executor,
    verbose=True
)

# Run examples
if __name__ == "__main__":
    res1 = agent.invoke({"input": "List all customers who bought insurance in April."})
    print("\nResult 1:", res1)

    res2 = agent.invoke({"input": "Fail this query to trigger a fallback."})
    print("\nResult 2:", res2)
