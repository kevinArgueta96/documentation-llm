from backend.core import runLLM

import streamlit as st
from streamlit_chat import message

st.header("LLM for Codigo de trabajo")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

promt = st.text_input("Promt", placeholder="Ingrese su pregunta")

if promt:
    with st.spinner("Generando respuesta"):
        generated_response = runLLM(query=promt)

        formatted_response = (
            f"{generated_response}"
        )

        st.session_state["user_prompt_history"].append(promt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)