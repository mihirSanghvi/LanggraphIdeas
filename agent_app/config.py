from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    sql_connection_string: str = Field(..., env="SQL_CONNECTION_STRING")

    opensearch_url: str = Field("http://localhost:9200", env="OPENSEARCH_URL")
    opensearch_index_name: str = Field("agent_app_docs", env="OPENSEARCH_INDEX_NAME")
    model_name: str = Field("gpt-4o", env="MODEL_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
