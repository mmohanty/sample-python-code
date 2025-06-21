
from langchain.agents import Tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

# Step 1: Define distinct tools
def dummy_text_to_sql(input: str):
    return "SQL QUERY RESULT"

def dummy_vector_search(input: str):
    return "DOCUMENT SEARCH RESULT"

tools = [
    Tool(
        name="text_to_sql",
        func=dummy_text_to_sql,
        description="""Use this tool when the user question relates to structured data, database queries, 
        tables, rows, filters, counts, or aggregates. Examples: 
        - "List all vendors"
        - "How many contracts were signed in Q1?"
        """
    ),
    Tool(
        name="vector_search",
        func=dummy_vector_search,
        description="""Use this tool when the user question is semantic, involves document understanding, 
        or does not relate to structured data. Examples:
        - "What does clause 5 mean?"
        - "Summarize the termination clause."
        """
    )
]

# Step 2: Create the LLM with custom system prompt
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    streaming=True,
)

system_prompt = """You are an intelligent agent that must choose between two tools.
Use `text_to_sql` for structured questions that involve SQL-like queries, filtering, or aggregations.
Use `vector_search` for semantic queries over documents or natural language understanding.
Only call one tool at a time. If unsure, prefer text_to_sql for questions with tabular terms.
"""

# Step 3: Few-shot examples (injected via prompt or memory)
# LangGraph React Agent doesn't support native few-shot examples, but prompt can be extended

# Step 4: Use checkpointing
checkpointer = MemorySaver()

# Step 5: Create React Agent
agent_node = create_react_agent(
    llm=llm,
    tools=tools,
    system_message=system_prompt
)

# Step 6: Fallback mechanism (retry second tool if first returns empty)
def tool_with_fallback(state):
    input_question = state["input"]
    first_try = agent_node.invoke(state)
    if isinstance(first_try, AIMessage) and not first_try.content.strip():
        print("First tool returned empty. Retrying with fallback.")
        fallback_tool = "vector_search" if "text_to_sql" in first_try.tool_calls[0]["name"] else "text_to_sql"
        tool_fn = dummy_vector_search if fallback_tool == "vector_search" else dummy_text_to_sql
        result = tool_fn(input_question)
        return {"result": result}
    return {"result": first_try.content}

# Graph setup
workflow = StateGraph()
workflow.add_node("agent", RunnableLambda(tool_with_fallback))
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compile and invoke
graph = workflow.compile()

# Example usage
result = graph.invoke({"input": "How many contracts were created in April?"})
print(result)
