import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads, delete_thread, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import uuid  # for Generating Thread ID for different chats
import time


# ***************************************************** Utility Functions *****************************************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values.get('messages', [])


# ***************************************************** Session Setup *****************************************************

# Problem : Below code reset python list everytime it reRun
# message_history = []

# Use Streamlit st.session_state - it maintain state after every reRun, it only erases state after hard refresh or program close.
# st.session_state -> is Dictionary only
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []
#
# if 'thread_id' not in st.session_state:
#     st.session_state['thread_id'] = generate_thread_id()
#
# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads'] = retrieve_all_threads()

def messages_to_ui(messages):
    """Convert LangChain messages to UI-friendly dicts."""
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        result.append({"role": role, "content": message.content})
    return result

# Initialize chat_threads first
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# Initialize thread_id
if 'thread_id' not in st.session_state:
    # Always start with a new chat on a fresh session
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
elif 'message_history' not in st.session_state:
    # Fallback in case thread_id exists but message_history is missing
    st.session_state['message_history'] = []

# Initialize message_history from the selected thread
if 'message_history' not in st.session_state:
    messages = load_conversation(st.session_state['thread_id'])
    st.session_state['message_history'] = messages_to_ui(messages)


if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ***************************************************** SideBar UI *****************************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)


st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:

    messages = load_conversation(thread_id)

    # Default title
    chat_title = "New Conversation"

    # Find first user message
    for message in messages:
        if isinstance(message, HumanMessage):
            chat_title = message.content[:30]
            break

    # =========================================================
    # ONE ROW FOR EACH CHAT
    # =========================================================

    chat_row = st.sidebar.container()

    with chat_row:

        col1, col2 = st.columns([5, 1], vertical_alignment="center")

        # -----------------------------------------------------
        # Chat button
        # -----------------------------------------------------

        with col1:

            if st.button(
                chat_title,
                key=f"chat_{thread_id}",
                use_container_width=True
            ):

                st.session_state['thread_id'] = thread_id

                temp_messages = []

                for message in messages:

                    if isinstance(message, HumanMessage):
                        role = 'user'
                    else:
                        role = 'assistant'

                    temp_messages.append({
                        'role': role,
                        'content': message.content
                    })

                st.session_state['message_history'] = temp_messages

                st.rerun()

        # -----------------------------------------------------
        # Delete button
        # -----------------------------------------------------

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{thread_id}",
                help="Delete chat"
            ):

                # Delete from SQLite
                delete_thread(thread_id)

                # Remove from Streamlit session
                st.session_state['chat_threads'].remove(thread_id)

                # If currently opened chat was deleted
                if st.session_state['thread_id'] == thread_id:

                    st.session_state['thread_id'] = generate_thread_id()
                    st.session_state['message_history'] = []

                st.rerun()



# ***************************************************** Main UI *****************************************************

# Loading the conversation history after every reload
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type Here...")

if user_input:

    # Display and Store User Message
    st.session_state['message_history'].append({'role':'user', 'content':user_input})

    with st.chat_message('user'):
        st.text(user_input)


    # Display and Store AI message

    # Before implementing Streaming Feature
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    # ai_message = response['messages'][-1].content
    #
    # st.session_state['message_history'].append({'role': 'ai', 'content': ai_message})
    #
    # with st.chat_message('ai'):
    #     st.text(ai_message)

    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # with it we can Trace Thread in Langsmith
    CONFIG = {
        "configurable": {"thread_id": str(st.session_state["thread_id"])},
        "metadata": {"thread_id": str(st.session_state["thread_id"])},
        "run_name": "chat_turn",
    }

    # After Implementing Streaming Feature
    with st.chat_message('assistant'):

        # we did chatbot.invoke() -> chatbot.stream()
        # chatbot.stream() returns a Generator Object

        # Holds the Streamlit status box
        status_holder = {"box": None}

        # Stores tool names used in this response
        tool_names = []


        def slow_generator():

            for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode='messages'
            ):

                # =====================================================
                # TOOL CALL
                # =====================================================

                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    # Store tool name
                    if tool_name not in tool_names:
                        tool_names.append(tool_name)

                    # Create status box when first tool is called
                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True
                        )

                    # Update existing status box
                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True
                        )

                # =====================================================
                # AI RESPONSE
                # =====================================================

                if isinstance(message_chunk, AIMessage):

                    # Stream only AI content
                    if message_chunk.content:
                        yield message_chunk.content

                        # Controls Streaming Speed
                        time.sleep(0.02)


        # st.write_stream takes Generator as Input
        ai_message = st.write_stream(slow_generator())

        # =====================================================
        # TOOL FINISHED
        # =====================================================

        if status_holder["box"] is not None:
            # Show which tool was actually used
            tools_used = ", ".join(
                f"`{tool}`" for tool in tool_names
            )

            status_holder["box"].update(
                label=f"✅ Used {tools_used}",
                state="complete",
                expanded=False
            )

        # Save assistant message
        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message
        })