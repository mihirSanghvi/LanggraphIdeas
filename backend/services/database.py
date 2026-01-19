from langchain_community.utilities import SQLDatabase
from backend.config import settings
import logging

logger = logging.getLogger(__name__)

def get_db() -> SQLDatabase:
    """Returns a configured SQLDatabase instance for MSSQL."""
    try:
        # Note: generic 'pyodbc' connection often requires driver installation.
        # Assuming the connection string in env is correct for the environment.
        return SQLDatabase.from_uri(settings.sql_connection_string)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
