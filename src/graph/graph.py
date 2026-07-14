from typing import TypedDict
from langgraph.graph import StateGraph, END

# Define the state schema
class GraphState(TypedDict):
    query: str
    intent: str

# 1. The Classifier Node
def classify_intent(state: GraphState):
    query = state["query"].lower()
    
    explain_triggers = ["explain", "meaning", "what does", "tashreeh", "tarjuma", "translation", "means"]
    thematic_triggers = ["about", "theme", "concept", "philosophy", "message", "poem on", "verses on"]
    
    if any(t in query for t in explain_triggers):
        return {"intent": "translate_explain"}
    elif any(t in query for t in thematic_triggers):
        return {"intent": "thematic_search"}
    else:
        return {"intent": "literal_lookup"}

# 2. The Execution Nodes (Mocked for routing tests)
def node_literal(state: GraphState):
    return state

def node_thematic(state: GraphState):
    return state

def node_explain(state: GraphState):
    return state

# 3. Conditional Edge Logic
def route_to_node(state: GraphState):
    return state["intent"]

# 4. Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("classifier", classify_intent)
workflow.add_node("literal_lookup", node_literal)
workflow.add_node("thematic_search", node_thematic)
workflow.add_node("translate_explain", node_explain)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_to_node,
    {
        "literal_lookup": "literal_lookup",
        "thematic_search": "thematic_search",
        "translate_explain": "translate_explain"
    }
)

workflow.add_edge("literal_lookup", END)
workflow.add_edge("thematic_search", END)
workflow.add_edge("translate_explain", END)

# Compile the runnable graph
query_router = workflow.compile()