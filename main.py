from backend.core import runLLM

import streamlit as st
from streamlit_chat import message

st.header("LLM for Codigo de trabajo")

if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

prompt = st.text_input("Prompt", placeholder="Ingrese su pregunta") or st.button(
    "Enviar"
)

if prompt:
    with st.spinner("Generando respuesta"):
        generated_response = runLLM(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        formatted_response = (
            f"{generated_response}"
        )

        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)
        st.session_state.chat_history.append((prompt, formatted_response))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                              st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)
