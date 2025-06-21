
from langchain.agents import Tool
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint import MemorySaver

# Step 1: Tool Definitions with Wrapped Logic
def fallback_guard(tool_fn):
    def wrapped(input_str):
        try:
            result = tool_fn(input_str)
            if not result or result.strip() in ["", "None", "[]", "{}", "No result"]:
                return {"success": False, "output": "", "reason": "empty result"}
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "output": "", "reason": str(e)}
    return wrapped

def dummy_text_to_sql(query: str):
    # Simulate failure for testing fallback
    if "fail" in query.lower():
        return ""
    return f"SQL result for: {query}"

def dummy_vector_search(query: str):
    return f"Vector search result for: {query}"

tools_dict = {
    "text_to_sql": fallback_guard(dummy_text_to_sql),
    "vector_search": fallback_guard(dummy_vector_search),
}

# Tool descriptions
tools = [
    Tool(
        name="text_to_sql",
        func=tools_dict["text_to_sql"],
        description="Use for structured database questions like filters, counts, or lists."
    ),
    Tool(
        name="vector_search",
        func=tools_dict["vector_search"],
        description="Use for semantic document understanding or free-text queries."
    )
]

# Step 2: LLM with system prompt
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    streaming=True,
)

system_prompt = """You are a smart agent deciding between tools:
- Use `text_to_sql` for structured questions on tables or SQL-style logic.
- Use `vector_search` for free-text or document questions.
Choose only one. If unsure, prefer text_to_sql."""

# Step 3: Agent & fallback logic
agent_node = create_react_agent(
    llm=llm,
    tools=tools,
    system_message=system_prompt
)

def tool_with_fallback(state):
    input_question = state["input"]
    chosen_tool = "text_to_sql"  # default first choice

    # First attempt
    first_result = tools_dict[chosen_tool](input_question)

    if not first_result.get("success"):
        fallback_tool = "vector_search"
        second_result = tools_dict[fallback_tool](input_question)
        return {
            "input": input_question,
            "chosen_tool": fallback_tool,
            "tool_output": second_result.get("output"),
            "fallback_used": True,
            "fallback_reason": first_result.get("reason"),
        }

    return {
        "input": input_question,
        "chosen_tool": chosen_tool,
        "tool_output": first_result.get("output"),
        "fallback_used": False
    }

# Step 4: Build graph
workflow = StateGraph()
workflow.add_node("agent", RunnableLambda(tool_with_fallback))
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compile graph
graph = workflow.compile()

# Test
if __name__ == "__main__":
    q1 = graph.invoke({"input": "List all contracts signed in April"})
    print(q1)
    q2 = graph.invoke({"input": "This should fail and trigger fallback"})
    print(q2)
