import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

import time

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Problem : Below code reset python list everytime it reRun
# message_history = []

# Use Streamlit st.session_state - it maintain state after every reRun, it only erases state after hard refresh or program close.
# st.session_state -> is Dictionary only
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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