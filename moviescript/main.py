from decouple import config
from langchain_community.document_loaders import WikipediaLoader

OPENAI_API_KEY = config("OPENAI_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain_community.vectorstores import Milvus

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
def generate_movie():
    list_of_names = ["Moondram Pirai", "Mullum Malarum", "Michael Madana Kama Rajan"]
    doc_list=[]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    for movie in list_of_names:
        docs = WikipediaLoader(query=movie, load_max_docs=2).load()
        docs = text_splitter.split_documents(docs)
        for doc in docs:
            doc_list.append(doc)

    print(len(doc_list))
    vectorstore = FAISS.from_documents(doc_list, embeddings)
    vectorstore.save_local("faiss_movie_index")




if __name__ == "__main__":
    print("Script Generator")
    generate_movie()
    # vector_db = Milvus(
    #         embeddings,
    #         connection_args={"host": "https://in03-f2a5b15fc667867.api.gcp-us-west1.zillizcloud.com", "port": "80"},
    #         collection_name="tamil_movies",
    #     )
    new_vectorstore = FAISS.load_local("faiss_movie_index", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.run("Create a new story plot with twist and turns")
    print(res)
    print("End")
