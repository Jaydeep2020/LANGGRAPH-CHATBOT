import os.path
import tempfile

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# new imports for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig

import sqlite3
import requests

from dotenv import load_dotenv

# Load OpenAI Key from .env
load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(
    model = 'gpt-4o-mini'
)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')


# -------------------
# 2. PDF retriever store (per thread)
# -------------------

LOCAL_INDEX_DIR = "./faiss_indexes"

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch retriever from RAM or load from disk if available."""
    if not thread_id:
        return None

    thread_key = str(thread_id)

    # 1. Return cached RAM retriever
    if thread_key in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_key]

    # 2. Reload index from disk if server restarted
    save_path = os.path.join(LOCAL_INDEX_DIR, thread_key)
    if os.path.exists(save_path):
        vector_store = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[thread_key] = retriever
        return retriever

    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        # Save vector index to disk for persistence across server restarts
        save_path = os.path.join(LOCAL_INDEX_DIR, str(thread_id))
        vector_store.save_local(save_path)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass



# -------------------
# 3. Tools
# -------------------

# Search Tool
search_tool = DuckDuckGoSearchRun(region="us-en")

# Calculator Tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == 'add':
            result = first_num + second_num
        elif operation == 'sub':
            result = first_num - second_num
        elif operation == 'mul':
            result = first_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {"error": "division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ESZSGUPU3X6K4L0M"
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread."""
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat thread. Please ask the user to upload a PDF.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,  # Fixed typo ('conrtext' -> 'context')
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# Make Tools List
tools = [search_tool, calculator, get_stock_price, rag_tool]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------

# Define WorkFlow State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------

# Define NODE logic
def chat_node(state: ChatState, config=None):
    """LLM node that automatically handles tools via bound RunnableConfig."""
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool`. You can also use web search, stock price, and "
            "calculator tools when helpful."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)  # Execute tool calls


# -------------------
# 6. Checkpointer
# -------------------

# Persistence
# Checkpointer for saving state over different execution of workflow
# checkpointer = InMemorySaver() # old

conn =sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

checkpointer.setup()

# -------------------
# 7. Graph
# -------------------

# Define Graph
graph = StateGraph(ChatState)

# Define Graph Nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Define Graph Edges
graph.add_edge(START, 'chat_node')

# graph.add_edge('chat_node', END)

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges('chat_node', tools_condition)

graph.add_edge('tools', 'chat_node')

# Compile Graph with Persistence Checkpointer
chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------

# Load All Past Conversation Thread ID from Database
# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config['configurable']['thread_id'])
#
#     return list(all_threads)

from datetime import datetime

def retrieve_all_threads():
    """Return all thread IDs sorted by latest checkpoint timestamp (most recent first)."""
    thread_timestamps = {}

    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        ts = checkpoint_tuple.checkpoint.get("ts")

        # Parse timestamp; fallback to minimum if missing
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                dt = datetime.min
        else:
            dt = datetime.min

        # Store the latest timestamp for each thread
        if thread_id not in thread_timestamps or dt > thread_timestamps[thread_id]:
            thread_timestamps[thread_id] = dt

    # Sort threads by timestamp descending (most recent first)
    sorted_threads = sorted(
        thread_timestamps.keys(),
        key=lambda tid: thread_timestamps[tid],
        reverse=True
    )
    return sorted_threads


# Delete chat history with thread ID
def delete_thread(thread_id):
    checkpointer.delete_thread(thread_id)



def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})