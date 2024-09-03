from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    runnable_with_history = RunnableWithMessageHistory(
        llm,
        get_session_history,
    )

    resp = runnable_with_history.invoke(
        [HumanMessage(content="hi - im bob!")],
        config={"configurable": {"session_id": "1"}},
    )

    print(resp.content)

