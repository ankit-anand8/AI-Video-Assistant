from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def build_vector_store(full_transcript:str,collection_name: str)->Chroma: 
    #The -> Chroma part means: "This function is expected to return a Chroma object."  It is called a return type hint.
    print("Building the vector store")

    # Delete old collection if it exists
    old_vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory="chroma_db"
    )
    old_vectorstore.delete_collection()

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_text(full_transcript)

    docs=[
        Document(page_content=chunk,metadata={"chunk_index":i})
        for i,chunk in enumerate(chunks)
    ]
    # Take a text chunk and wrap it in a LangChain Document object.
    # page_content=chunk → stores the actual transcript text
    # metadata={"chunk_index": i} → stores extra information about the chunk, such as its position
    # The resulting Document can then be given to Chroma/vector store.

    embeddings=get_embeddings()
    vectorstore=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="chroma_db"
    )
    return vectorstore
