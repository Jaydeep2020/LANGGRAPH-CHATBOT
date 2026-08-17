import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage

import uuid  # for Generating Thread ID for different chats
import time


# ***************************************************** Utility Functions *****************************************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

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
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ***************************************************** SideBar UI *****************************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:

    messages = load_conversation(thread_id)

    # Default title
    chat_title = "New Conversation"

    # Find first user message
    for message in messages:
        if isinstance(message, HumanMessage):
            chat_title = message.content[:30]
            break

    if st.sidebar.button(chat_title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id


        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role='user'
            else:
                role='assistant'

            temp_messages.append({'role': role, 'content' : message.content})

        st.session_state['message_history'] = temp_messages



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

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # After Implementing Streaming Feature
    with st.chat_message('assistant'):

        # we did chatbot.invoke() -> chatbot.stream()
        # chatbot.stream() returns a Generator Object

        message_generator = (
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        )
        )

        # Controls Streaming Speed
        def slow_generator():
            for chunk in message_generator:
                yield chunk
                time.sleep(0.02)

        # st.write_stream takes Generator as Input
        ai_message = st.write_stream(slow_generator())

        st.session_state['message_history'].append({
            'role': 'ai',
            'content': ai_message
        })