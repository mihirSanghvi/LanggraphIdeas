from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from agent_app.services.llm import get_llm
from agent_app.services.vector import get_retriever
from agent_app.tools.rag import retrieve_documents
from agent_app.tools.sql import execute_sql, list_tables
from agent_app.graph.state import AgentState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from agent_app.tools.sql import (
    sql_db_list_tables, sql_db_schema, sql_db_query, 
    sql_db_query_checker, sql_db_column_value_checker, sql_db_is_readonly,
    sql_db_is_aggregate_only
)

llm = get_llm()

class RouterOutput(BaseModel):
    intent: str = Field(..., description="One of: RAG, SQL, BOTH, OFFTOPIC")
    filters: Dict[str, str] = Field(
        default_factory=dict, 
        description="Key-value pairs for metadata filtering (e.g. {'department': 'sales'}). Empty if no specific filters."
    )

def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's latest message and decides the intent and filters.
    """
    messages = state['messages']
    last_message = messages[-1].content
    
    system_prompt = """You are an intelligent router for an AI assistant.
    Your task is to classify the user's query and extract any metadata filters.
    
    INTENT CATEGORIES:
    - "RAG": about documents, text, policies (unstructured).
    - "SQL": about structured data, counting records, statistics.
    - "BOTH": requires both.
    - "OFFTOPIC": unrelated.
    
    FILTERS:
    If the user mentions specific categories, dates, departments, or sources that acts as a filter, extract them.
    Examples:
    "Show me sales reports" -> filters: {"department": "sales"}
    "Policies from 2024" -> filters: {"year": "2024"}
    """
    
    # Use structured output for reliability
    structured_llm = llm.with_structured_output(RouterOutput)
    
    router_chain = (
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        | structured_llm
    )
    
    try:
        result = router_chain.invoke({"input": last_message})
        intent = result.intent.strip().upper()
        filters = result.filters
    except Exception:
        # Fallback
        intent = "OFFTOPIC"
        filters = {}
    
    # Validation
    valid_intents = ["RAG", "SQL", "BOTH", "OFFTOPIC"]
    if intent not in valid_intents:
        intent = "OFFTOPIC"
        
    return {"intent": intent, "filters": filters}

def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes RAG retrieval using the latest user message and extracted filters.
    """
    messages = state['messages']
    query = messages[-1].content
    filters = state.get('filters', {})
    
    # Call retrieve_documents tool directly
    context = retrieve_documents.invoke({
        "query": query, 
        "filters": filters
    })
    
    return {"rag_context": context}



def sql_gen_node(state: AgentState) -> Dict[str, Any]:
    """
    Agentic SQL Node: Uses ReAct agent to discover schema, check values, and execute SQL.
    """
    messages = state['messages']
    query = messages[-1].content
    filters = state.get('filters', {})
    
    filter_context = ""
    if filters:
        filter_context = f"\n    CRITICAL: The user has specified the following context/filters. You MUST apply them:\n    {filters}"
    
    system_prompt = f"""You are an advanced SQL agent for Microsoft SQL Server.
    Your goal is to answer the user's question by querying the database.
    
    CRITICAL SECURITY CONSTRAINT:
    - You must NEVER return raw data rows. 
    - You MUST ONLY return aggregated data (COUNT, SUM, AVG, MIN, MAX, GROUP BY).
    - If the user asks for raw records (e.g. "Show me all sales"), you must aggregate them (e.g. "Count of sales", "Sales by region") or explain you can only show summaries.
    - Queries like `SELECT *` or `SELECT col1, col2` without aggregation are STRICTLY FORBIDDEN.
    
    You have access to tools to:
    1. List tables (`sql_db_list_tables`)
    2. Get table schema (`sql_db_schema`)
    3. Check specific column values for fuzzy matching (`sql_db_column_value_checker`)
    4. Check your query syntax (`sql_db_query_checker`)
    5. Verify query only returns aggregates (`sql_db_is_aggregate_only`)
    6. Execute the query (`sql_db_query`)
    
    Workflow:
    1. Start by listing tables to understand what's available.
    2. Get schema for relevant tables.
    3. If the user filter is vague (e.g. "Sales"), use `sql_db_column_value_checker` to find the exact database value.
    4. CRITICAL: Verify your query is aggregation-only using `sql_db_is_aggregate_only`.
    5. Validate your query logic with `sql_db_query_checker`.
    6. Execute the query with `sql_db_query`.
    
    {filter_context}
    
    Output the final answer based on the query result.
    """
    
    tools = [
        sql_db_list_tables, sql_db_schema, sql_db_query, 
        sql_db_query_checker, sql_db_column_value_checker, sql_db_is_readonly,
        sql_db_is_aggregate_only
    ]
    
    # Create the agent
    agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    # Invoke. We pass the messages.
    # Note: create_react_agent manages its own state scratchpad.
    result = agent_executor.invoke({"messages": messages})
    
    # The result['messages'] includes the tool calls and valid outputs.
    # The final AIMessage contains the synthesized answer.
    last_message = result['messages'][-1]
    
    # detailed_answer = last_message.content
    # We return this as 'sql_result' so 'response_node' can see it.
    # We can also populate 'sql_query' if we want by parsing tool calls, but let's skip for now.
    
    return {"sql_result": last_message.content}

def sql_exec_node(state: AgentState) -> Dict[str, Any]:
    """
    Deprecated: The SQL execution is now handled within the sql_gen_node agent.
    This node acts as a pass-through if still in the graph.
    """
    return {}


def response_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes the final answer using the context.
    """
    messages = state['messages']
    original_query = messages[-1].content
    intent = state.get('intent')
    rag_context = state.get('rag_context', '')
    sql_result = state.get('sql_result', '')
    
    # Combine context
    full_context = ""
    if rag_context:
        full_context += f"Document Context:\n{rag_context}\n\n"
    if sql_result:
        full_context += f"Database Result:\n{sql_result}\n\n"
    
    system_prompt = """You are a helpful assistant.
    Answer the user's question based on the provided context.
    If the context contains database results, interpret them friendly.
    If the context contains document excerpts, cite the source if available.
    If you don't know, say you don't know.
    """
    
    input_text = f"Context:\n{full_context}\n\nQuestion: {original_query}"
    
    response_chain = (
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        | llm
        | StrOutputParser()
    )
    
    answer = response_chain.invoke({"input": input_text})
    
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
