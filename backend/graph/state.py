from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of the agent in the LangGraph workflow.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    clarification_needed: bool
    sql_query: Optional[str]
    rag_context: Optional[str]
    sql_result: Optional[str]
    intent: Optional[str] # "RAG", "SQL", "BOTH", "OFFTOPIC"
    filters: Optional[dict] # Metadata filters extracted for RAG
    final_answer: Optional[str]
