from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_community.chat_message_histories import SQLChatMessageHistory
load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///memory.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)


config = {"configurable": {"session_id": "test_session_id"}}

response = chain_with_history.invoke({"question": "En que equipo juego?"}, config=config)

print(response)