from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.database import build_vector_store


#1. GET VIDEO / AUDIO INPUT
source = input("Enter YouTube URL or local file path: ")

# 2. AUDIO PROCESSING

print("\n========== AUDIO PROCESSING ==========\n")
chunks = process_input(source)
print(f"Created {len(chunks)} audio chunks.")

# 3. TRANSCRIPTION
print("\n========== TRANSCRIPTION ==========\n")
full_transcript = transcribe_all(chunks)
print("Transcription completed.")


# 4. BUILD VECTOR DATABASE
print("\n========== VECTOR DATABASE ==========\n")

# REMOVE OLD VIDEO COLLECTION
collection_name = "current_video"

vectorstore = build_vector_store(full_transcript,collection_name)  
#means main.py sends the transcript to database.py, and database.py creates the vector store and returns it to main.py
print("Vector database created successfully.")


# 5. CREATE RETRIEVER
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":5,
        "fetch_k":15,
        "lambda_mult":0.5
    }
)


# 6. FORMAT RETRIEVED DOCUMENTS
def format_docs(docs):
    text = ""
    for doc in docs:
        text += doc.page_content + "\n\n"
    return text
#It takes the retrieved Document objects and extracts only their actual text.


# 7. PROMPT
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate([
    ("system", """You are a helpful Video Assistant.

    Answer the user's question using only the information
    provided in the video context.

    If the user asks for a summary, provide a clear summary
    with a proper heading and subheadings.

    If the answer cannot be found in the video context, say:
    I Could Not Find The Answer From The Video.

    Do not make up information."""),

    ("user","""
    Context : 
    "{context}" 
    
    Question :
    "{question}"
    """)
])

 
# 8. LOAD LLM
from langchain_google_genai import GoogleGenerativeAI
model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

print("-----------------------------RAG System Created--------------------------")
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Press 0 to Exit")

# 9. CREATE NORMAL RAG CHAIN
normal_chain=(
    {"context":retriever | format_docs,"question":RunnablePassthrough()} | template | model|StrOutputParser() 
    )

#10. CREATE SUMMARY CHAIN

summary_chain = ( template | model | StrOutputParser() )
SUMMARY_WORDS = ["summary","summarize", "overview", "main points","key points", "about this video"] 
#words like summary,summarize,overview,main points,"Tell me about this video", etc all means the user want the summary

while True:
    query = input("\nYou : ")
    if query == "0":
        break
    if not query:
        continue
    #Empty input. Pressing Enter sends an empty question to the API. To fix this we use continue.

     # Summary → use the entire transcript
    if any(word in query.lower() for word in SUMMARY_WORDS):       
    #If at least one summary-related word/phrase exists in the user's question, use the summary chain.
        final_answer = summary_chain.invoke({"context": full_transcript,"question": query})

     # Normal question → use RAG retrieval
    else:
        final_answer = normal_chain.invoke(query)
    print("\nBot :", final_answer)



#Why RunnablePassthrough()?
#It simply means: "Take the original input and pass it through unchanged."

#So if:
#query = "What did Obama say about education?"

#then:
#RunnablePassthrough() passes that exact question to:"question"
# while the retriever uses the same question to find relevant transcript chunks.