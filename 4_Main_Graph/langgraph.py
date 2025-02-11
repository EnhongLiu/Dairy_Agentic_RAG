from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool

from langgraph.prebuilt import tools_condition
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from IPython.display import Image, display

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("decide_tool", decision_agent)  # decision agent to retrieve or end
# retrieve = ToolNode([jds_rag])
workflow.add_node("jds_retrieve", jds_rag)  # retrieve JDS insights
workflow.add_node("web_search", web_search) # Web search the answer
workflow.add_node("ask_for_more", response)  # Re-write the question

# Connect agent nodes
workflow.add_edge(START, "decide_tool")
workflow.add_conditional_edges("decide_tool", tools_condition,{"tools": "jds_retrieve",  END: END}) # Decide whether to retrieve
workflow.add_conditional_edges("jds_retrieve", grade_jds_answer, {'web':'web_search', 'end': END}) # Decide whether to websearch
workflow.add_conditional_edges("web_search", grade_web_answer,{"response": "ask_for_more", "end": END}) # Decide whether to rewrite
workflow.add_edge("ask_for_more", END)

# Compile
graph = workflow.compile()

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
