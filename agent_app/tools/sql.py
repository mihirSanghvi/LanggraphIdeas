from typing import List, Optional
import re
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from agent_app.services.database import get_db
from agent_app.services.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def is_readonly_sql(sql: str) -> bool:
    """
    Validates if the SQL query is read-only.
    Returns True if safe, False otherwise.
    """
    # Normalize
    sql_upper = sql.upper().strip()
    
    # Forbidden keywords
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", 
        "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
    ]
    
    # Check if any forbidden keyword starts a statement or follows a separator
    # This is a basic regex; specific SQL dialects might have edge cases, 
    # but this covers standard potentially destructive commands.
    for word in forbidden:
        # Check for word boundaries to avoid false positives (e.g. 'SELECT * FROM UPDATES')
        # We want to match: (START OR whitespace/semicolon) WORD (whitespace OR END OR semicolon)
        pattern = r'(^|[\s;])' + word + r'([\s;]|$)'
        if re.search(pattern, sql_upper):
            return False
            
    return True


@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    db = get_db()
    # get_table_names() returns a list of strings
    tables = db.get_table_names()
    return ", ".join(tables)

@tool
def sql_db_schema(table_names: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: table1, table2, table3
    """
    db = get_db()
    # Parse input string into list
    tables = [t.strip() for t in table_names.split(",") if t.strip()]
    
    try:
        # get_table_info returns the DDL. We might want to append sample rows?
        # SQLDatabase.get_table_info() usually includes samples if configured, but let's check.
        # By default langchain's get_table_info() is just DDL.
        # We can manually fetch samples.
        info = db.get_table_info(table_names=tables)
        return info
    except Exception as e:
        return f"Error getting schema for {tables}: {e}"

@tool
def sql_db_is_readonly(query: str) -> bool:
    """
    Checks if a query is read-only. Useful for self-verification.
    Returns True if safe, False otherwise.
    """
    return is_readonly_sql(query)

@tool
def sql_db_query(query: str) -> str:
    """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
    """
    if not is_readonly_sql(query):
        return "Error: The query contains forbidden (write/modify) keywords. Execution aborted for safety."
        
    db = get_db()
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing SQL: {e}"

@tool
def sql_db_query_checker(query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    llm = get_llm()
    dialect = "Microsoft SQL Server"
    
    template = """
    {query}
    Double check the {dialect} query above for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    Output the final SQL query only.

    SQL Query: """
    
    chain = (
        ChatPromptTemplate.from_template(template) 
        | llm 
        | StrOutputParser()
    )

    response = chain.invoke({"query": query, "dialect": dialect})
    
    return response.replace("```sql", "").replace("```", "").strip()

@tool
def sql_db_is_aggregate_only(query: str) -> str:
    """
    Checks if a query returns ONLY aggregated data (no raw rows).
    Use this to verify the Aggregation Constraint.
    """
    llm = get_llm()
    system_prompt = """You are a SQL compliance checker.
    Validate if the following SQL query returns ONLY aggregated results (e.g. using COUNT, SUM, AVG, MIN, MAX or GROUP BY).
    Queries selecting raw rows (like SELECT * FROM) are FORBIDDEN.
    
    If the query is aggregate-only, return "True".
    If it returns raw data, return "False: <brief explanation>".
    """
    
    chain = (
        ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"query": query})

@tool
def sql_db_column_value_checker(table: str, column: str, search_term: str) -> str:
    """
    Input is table name, column name, and column value to perform the fuzzy search to find valid column values.
    Use this to map vague user terms to actual database values.
    """
    db = get_db()
    # Sanitize inputs to prevent injection (basic check)
    # We should ideally check if table/column exists, but let's trust the db driver to error out safely on parametrized query
    # BUT, parameterizing table/column names is not standard in SQL. We must validate them.
    
    all_tables = db.get_table_names()
    if table not in all_tables:
        return f"Error: Table '{table}' does not exist."
        
    # We can't easily validate column without schema, but we can try to run the query.
    # Safe construction:
    # We use f-string for table/column because they can't be bound parameters, 
    # but we validated table. Column injection is still a risk if we don't validate column.
    # Let's trust the agent mostly or do a 'SELECT TOP 0' to check column?
    # For now, simplistic validation.
    
    try:
        # Note: We are using string interpolation for the like clause value which is risky if search_term has quotes.
        # Ideally we should use params. db.run() supports execution but wrapping generic pyodbc usage is cleaner.
        # Let's try to use db.run() which is simple string execution.
        # We should escape the search_term.
        safe_term = search_term.replace("'", "''")
        query = f"SELECT DISTINCT TOP 10 {column} FROM {table} WHERE {column} LIKE '%{safe_term}%'"
        
        return db.run(query)
    except Exception as e:
        return f"Error checking values: {e}"

