from langchain_core.tools import tool
from backend.services.vector import get_retriever

@tool
def retrieve_documents(query: str, k: int = 4, filters: dict = None) -> str:
    """
    Search for documents relevant to the user's query using the vector database.
    
    Args:
        query: The search query.
        k: Number of results to return (default 4).
        filters: Optional metadata filters (e.g. {"department": "sales", "year": "2024"}).
    """
    # OpenSearch adapter handles the dictionary filter conversion
    retriever = get_retriever(k=k, filter=filters)
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant documents found."
        
    # Format docs
    result = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        # Include metadata in output for visibility
        result.append(f"Content: {doc.page_content}\nSource: {source}\nMetadata: {doc.metadata}\n")
        
    return "\n---\n".join(result)
