from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

import sqlite3

from dotenv import load_dotenv

# Load OpenAI Key from .env
load_dotenv()

# Define your LLM model
llm = ChatOpenAI(
    model = 'gpt-4.1-mini'
)

# Define WorkFlow State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define NODE logic
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Persistence
# Checkpointer for saving state over different execution of workflow
# checkpointer = InMemorySaver() # old

conn =sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

# Define Graph
graph = StateGraph(ChatState)

# Define Graph Nodes
graph.add_node('chat_node', chat_node)

# Define Graph Edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile Graph with Persistence Checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

# Load All Past Conversation Thread ID from Database
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

# Delete chat history with thread ID
def delete_thread(thread_id):
    checkpointer.delete_thread(thread_id)