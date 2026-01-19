import streamlit as st
import os
from langchain_core.messages import HumanMessage
# Ensure environment is loaded
from dotenv import load_dotenv
load_dotenv()

# Import graph
from backend.graph.graph import build_graph

st.set_page_config(page_title="Agentic AI: RAG + SQL", layout="wide")

st.title("🤖 Agentic AI Assistant")
st.markdown("Supports **RAG** (Vectors) and **SQL** (Database) simultaneously.")

# Initialize Graph
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask about documents or data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Execution
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Inputs
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # Stream execution to show steps (optional, or just get final)
            # For simplicity, we just invoke.
            result = st.session_state.graph.invoke(inputs)
            
            final_answer = result.get("final_answer", "No response generated.")
            intent = result.get("intent", "Unknown")
            
            # Display detailed steps in expander
            with st.expander(f"Agent Logic (Intent: {intent})"):
                if result.get('rag_context'):
                    st.markdown("**RAG Context Used**")
                    st.text(result['rag_context'][:500] + "...")
                if result.get('sql_query'):
                    st.markdown("**SQL Query Generated**")
                    st.code(result['sql_query'], language="sql")
                if result.get('sql_result'):
                    st.markdown("**SQL Result**")
                    st.text(str(result['sql_result'])[:500])

            message_placeholder.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            message_placeholder.error(f"Error: {e}")
