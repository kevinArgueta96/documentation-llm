import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from consts import INDEX_NAME

load_dotenv()

template = """Use the following pieces of context to answer the question at the end.

        If you don't know the answer, just say that you don't know, don't try to make up an answer
        Use three sentences maximum and keep the answer as concise as possible.

        Always say "Gracias por preguntar!" at the end of the answer.
        Always response in spanish language

        If you have the source of the answer, said (Obtenida de: "SET HER THE URL OR SOURCE))
        {context}

        Question: {question}

        Helpful Answer:"""

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


def runLLM(query: str) -> Any:
    embedding = OpenAIEmbeddings()
    #llm = ChatOpenAI(verbose=True,temperature=0)
    llm = Ollama(model="mistral",verbose=True,temperature=0,)

    vectorstore = PineconeVectorStore(embedding=embedding, index_name=INDEX_NAME)

    custom_rag_prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
    )

    res = rag_chain.invoke(query)

    return res


if __name__ == "__main__":
    runLLM("Cuantas vacaciones por ley tiene que tener un empleado")
