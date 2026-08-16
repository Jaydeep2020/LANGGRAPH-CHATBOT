from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

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
checkpointer = InMemorySaver()

# Define Graph
graph = StateGraph(ChatState)

# Define Graph Nodes
graph.add_node('chat_node', chat_node)

# Define Graph Edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile Graph with Persistence Checkpointer
chatbot = graph.compile(checkpointer=checkpointer)