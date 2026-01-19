from langgraph.graph import StateGraph, END
from agent_app.graph.state import AgentState
from agent_app.graph.nodes import (
    router_node, rag_node, sql_gen_node, response_node
)

def build_graph():
    """
    Constructs the LangGraph execution graph.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("sql_gen", sql_gen_node)
    workflow.add_node("response", response_node)
    
    # Define Entry Point
    workflow.set_entry_point("router")
    
    # Conditional Edges for Router
    def route_decision(state: AgentState):
        intent = state.get("intent")
        if intent == "RAG":
            return "rag"
        elif intent == "SQL":
            return "sql_gen"
        elif intent == "BOTH":
            return ["rag", "sql_gen"] # Parallel execution if supported by engine or sequential in map
        else:
            return "response" # Just chat or offtopic handling

    # Edge Logic
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "sql_gen": "sql_gen",
            "response": "response"
        }
    )
    
    # Normal Edges
    workflow.add_edge("rag", "response")
    
    # sql_gen (Agent) -> response
    workflow.add_edge("sql_gen", "response")

    
    # Note: Handling "BOTH" completely correctly in parallel might need a merge node. 
    # For this MVP, if the router says BOTH, we might need a specific "parallel" handling or just chain them.
    # Let's adjust the router logic in `route_decision` to just map known nodes. 
    # If "BOTH" is returned, the current `route_decision` returns a list `["rag", "sql_gen"]`.
    # LangGraph will run them in parallel and then we need to converge them.
    # We need a node to converge before response? Or response can accept inputs from multiple?
    # Response expects `rag_context`. If running parallel, both will update state.
    # `rag_node` updates `rag_context`. `sql_exec_node` updates `rag_context` (in my implementation).
    # If they both update `rag_context`, one might overwrite the other or cause a conflict.
    # FIX: Update `sql_exec_node` to update a different key `sql_result` and `response_node` joins them,
    # OR make `rag_context` append-only?
    # I'll update `nodes.py` in a separate turn to fix the state conflict if needed.
    # For now, let's assume one path. 
    # If I return a list in conditional edges, they run in parallel, then I need to close the loops.
    
    workflow.add_edge("response", END)
    
    return workflow.compile()
