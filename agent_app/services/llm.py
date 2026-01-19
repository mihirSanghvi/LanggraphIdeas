from langchain_openai import ChatOpenAI
from agent_app.config import settings

def get_llm():
    """Returns a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0
    )
