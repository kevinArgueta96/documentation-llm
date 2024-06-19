import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from consts import INDEX_NAME

load_dotenv()

template = """
Usa las siguientes piezas de contexto para responder la pregunta al final.

Si no sabes la respuesta, simplemente di que no sabes, no intentes inventar una respuesta.
Usa un máximo de tres oraciones y mantén la respuesta lo más concisa posible.

Siempre di "¡Gracias por preguntar!" al final de la respuesta.
Responde siempre en español.

Si tienes la fuente de la respuesta, di (Obtenida de: "AQUÍ LA URL O FUENTE").

Contexto disponible:
{context}

Pregunta: {question}

Usa el siguiente historial de conversación para proporcionar una respuesta informada:
{chat_history}

Respuesta útil:
"""

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


def runLLM(query: str, chat_history: List[Dict[str, Any]] = []):
    embedding = OpenAIEmbeddings()
    # llm = ChatOpenAI(verbose=True,temperature=0)
    llm = Ollama(model="llama3", verbose=True, temperature=0)

    vectorstore = PineconeVectorStore(embedding=embedding, index_name=INDEX_NAME)

    print(chat_history)
    custom_rag_prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough(),
             "chat_history": RunnablePassthrough(lambda _: chat_history)}
            | custom_rag_prompt
            | llm
    )

    return rag_chain.invoke(query)

if __name__ == "__main__":
    runLLM("Cuantas vacaciones por ley tiene que tener un empleado")
