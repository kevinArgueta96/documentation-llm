from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from consts import INDEX_NAME
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone


## NEW UPLOAD PINECONE LIBRARIES
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

import os

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

# load PDF file using loaders from langchain
loader = PyPDFLoader("pdf-files/codigo-de-trabajo.pdf")
document = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len
)

docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()

print("****Loading to vectorstore***")
PineconeLangChain.from_documents(docs, embeddings, index_name=INDEX_NAME)
print("****Loading to vectorstore***")

#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#embeddings = OpenAIEmbeddings()

#PineconeLangChain.from_documents(documents, embeddings, index_name=INDEX_NAME)

#print("PART OF CHUNK:" + chunks[7])
#print(len(chunks))


#TEST EMBEDDIGNS
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#embeddings = OpenAIEmbeddings()

#knowledge_base = FAISS.from_texts(chunks, embeddings)

#question = "¿Cuáles son las obligaciones específicas que debe cumplir el agente reclutador o la empresa, según el artículo 34, para garantizar la protección y bienestar de los trabajadores guatemaltecos que son contratados para trabajar fuera del país?"
#docs = knowledge_base.similarity_search(question, 3)

#PineconeLangChain.from_documents(text, embeddings, index_name=INDEX_NAME)

"""
prompt = ChatPromptTemplate.from_template(
    question +
    "\n\n{context}"
)

print("START TO PROCESS THE INFORMATION WITH THE LLM")
#model = ChatOpenAI(model="gpt-3.5-turbo-0125", verbose = True, temperature=0)
model = Ollama(model="mistral", verbose=True, temperature=0)

chain = create_stuff_documents_chain(model, prompt)

new_prompt = chain.invoke({"context": docs})

print(new_prompt)
"""
