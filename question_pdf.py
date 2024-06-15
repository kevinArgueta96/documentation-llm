import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME
from pinecone import Pinecone

# USE LOCAL LLM
from langchain_community.llms import Ollama

load_dotenv()


def format_docs(docs):
    print(docs)
    return "\n\n".join(docs.page_content for doc in docs)


if __name__ == "__main__":
    print("Welcome to question to codigo de trabajo")

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
    )

    #query = "¿Qué estipula el artículo 13 sobre la contratación de trabajadores guatemaltecos en comparación con trabajadores extranjeros?"
    #query = "¿Cuales son las horas que un trabajador debe tener y que jornadas existen?, si puedes extenderte, hazlo"
    #query = "¿Cuales son las horas que un trabajador debe tener y que jornadas existen?, si puedes extenderte, hazlo"
    query = "Cuantas hpras laborales son las que un trabajador debe laboral"
    embeddings = OpenAIEmbeddings()
    llm = Ollama(model="mistral")
    # llm = ChatOpenAI()

    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

    """"
    // We obtain the prompt template from here:
    https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    langchain-ai/retrieval-qa-chat
    SYSTEM
    Answer any use questions based solely on the context below:
    <context>
    {context}
    </context>
    PLACEHOLDER
    chat_history
    HUMAN
    {input}
    """
    retrieval_qa_chain_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chain_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    template = """Use the following pieces of context to answer the question at the end.

    If you don't know the answer, just say that you don't know, don't try to make up an answer
    Use three sentences maximum and keep the answer as concise as possible.

    Always say "Gracias por preguntar!" at the end of the answer.
    Always response in spanish language
    
    If you have the source of the answer, said (Obtenida de: "SET HER THE URL OR SOURCE))
    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    #print(vectorstore.as_retriever())
    rag_chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
    )

    # result = retrieval_chain.invoke(input={"input": query})
    res = rag_chain.invoke(query)
    print(res)
