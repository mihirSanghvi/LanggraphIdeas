import os
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from backend.config import settings
import pandas as pd
from langchain_core.documents import Document

def get_vectorstore():
    """Returns a configured OpenSearch vector store instance."""
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key.get_secret_value()
    )
    
    return OpenSearchVectorSearch(
        opensearch_url=settings.opensearch_url,
        index_name=settings.opensearch_index_name,
        embedding_function=embeddings
    )

def _convert_to_opensearch_filter(filter_dict: dict) -> dict:
    """
    Converts a simple dictionary filter to OpenSearch DSL.
    e.g. {"department": "sales"} -> {"bool": {"filter": [{"term": {"metadata.department.keyword": "sales"}}]}}
    """
    if not filter_dict:
        return None
        
    # If it already looks like a DSL query (has "bool", "term", etc.), return as is
    if any(k in filter_dict for k in ["bool", "term", "match", "range"]):
        return filter_dict
        
    filters = []
    for key, value in filter_dict.items():
        # Handle metadata prefix if not present (LangChain often prefixes with metadata.)
        field_name = key if key.startswith("metadata.") else f"metadata.{key}"
        
        if isinstance(value, str):
            # Use .keyword for exact matching on string fields
            filters.append({"term": {f"{field_name}.keyword": value}})
        elif isinstance(value, list):
            filters.append({"terms": {f"{field_name}.keyword": value}})
        else:
            # Numbers, booleans don't need .keyword
            filters.append({"term": {field_name: value}})
            
    if len(filters) == 1:
        return {"bool": {"filter": filters[0]}}
    return {"bool": {"filter": filters}}

def get_retriever(k: int = 4, filter: dict = None, score_threshold: float = None):
    """
    Returns a retriever from the vector store with optional filtering and thresholding.
    params:
        k: Number of documents to return.
        filter: Metadata filter dictionary.
        score_threshold: Minimum similarity score (0-1) to return.
    """
    vectorstore = get_vectorstore()
    
    search_kwargs = {"k": k}
    if filter:
        # Convert simple dict filter to OpenSearch DSL if needed
        search_kwargs["filter"] = _convert_to_opensearch_filter(filter)
        
    search_type = "similarity"
    if score_threshold is not None:
        search_type = "similarity_score_threshold"
        search_kwargs["score_threshold"] = score_threshold
        
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

# Ingestion Logic
# Ingestion Logic

def ingest_excel(file_path: str, text_column: str, metadata_columns: list = None):
    """
    Ingests data from an Excel file into the vector database.
    Each row is treated as a single document chunk.
    
    Args:
        file_path (str): Path to the Excel file.
        text_column (str): Name of the column containing the main text to embed.
        metadata_columns (list): List of column names to include as metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in Excel file. Available columns: {list(df.columns)}")
        
    documents = []
    
    # Iterate over rows
    for index, row in df.iterrows():
        text_content = str(row[text_column])
        
        # Skip empty content
        if not text_content or text_content.lower() == 'nan':
            continue
            
        metadata = {"source": os.path.basename(file_path), "row_index": index}
        
        if metadata_columns:
            for col in metadata_columns:
                if col in df.columns:
                    metadata[col] = str(row[col])
        
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)
        
    print(f"Prepared {len(documents)} documents. Adding to vector store...")
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest Excel file into Vector DB")
    parser.add_argument("--file", required=True, help="Path to Excel file")
    parser.add_argument("--text-col", required=True, help="Column name for text content")
    parser.add_argument("--meta-cols", nargs="*", help="List of columns to use as metadata")
    
    args = parser.parse_args()
    
    ingest_excel(args.file, args.text_col, args.meta_cols)
