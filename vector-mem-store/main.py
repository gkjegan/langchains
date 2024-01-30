import os
from decouple import config
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI

OPENAI_API_KEY = config("OPENAI_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")

if __name__ == "__main__":
    print("Hello Vector Mem Store")
    pdf_path = "2210.03629.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    dpcument = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(dpcument)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)