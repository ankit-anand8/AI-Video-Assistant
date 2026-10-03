from dotenv import load_dotenv
import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.database import build_vector_store

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI


load_dotenv()

COLLECTION_NAME = "current_video"
SUMMARY_WORDS = [
    "summary",
    "summarize",
    "overview",
    "main points",
    "key points",
    "about this video",
]


def format_docs(docs):
    text = ""
    for doc in docs:
        text += doc.page_content + "\n\n"
    return text


def create_chains(vectorstore, full_transcript):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.5},
    )

    template = ChatPromptTemplate(
        [
            (
                "system",
                """You are a helpful Video Assistant.

Answer the user's question using only the information
provided in the video context.

If the user asks for a summary, provide a clear summary
with a proper heading and subheadings.

If the answer cannot be found in the video context, say:
I Could Not Find The Answer From The Video.

Do not make up information.""",
            ),
            (
                "user",
                'Context :\n"{context}"\n\nQuestion :\n"{question}"',
            ),
        ]
    )

    model = GoogleGenerativeAI(model="gemini-3.5-flash-lite")
    normal_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | template
        | model
        | StrOutputParser()
    )
    summary_chain = template | model | StrOutputParser()
    return normal_chain, summary_chain


st.set_page_config(page_title="TalkToVideo", page_icon="🎥")
st.title("🎥 TalkToVideo")
st.write("🔗 Enter a YouTube URL or a local audio/video file path to get started.")

if "normal_chain" not in st.session_state:
    st.session_state.normal_chain = None
    st.session_state.summary_chain = None

with st.form("source_form"):
    source = st.text_input("📂 YouTube URL or local file path")
    process_clicked = st.form_submit_button("🚀 Process source")

if process_clicked:
    if not source.strip():
        st.warning("⚠️ Enter a YouTube URL or local file path.")
    else:
        try:
            with st.spinner("⏳ Processing audio and creating the video assistant…"):
                chunks = process_input(source.strip())
                full_transcript = transcribe_all(chunks)
                vectorstore = build_vector_store(full_transcript, COLLECTION_NAME)
                normal_chain, summary_chain = create_chains(
                    vectorstore, full_transcript
                )

            st.session_state.normal_chain = normal_chain
            st.session_state.summary_chain = summary_chain
            st.session_state.full_transcript = full_transcript
            st.success("✅ Transcription and vector database are ready.")
        except Exception as error:
            st.error(f"❌ Could not process the source: {error}")

if st.session_state.normal_chain is not None:
    st.subheader("💬 Ask about the video")
    with st.form("question_form"):
        query = st.text_input("❓ Your question")
        ask_clicked = st.form_submit_button("✨ Ask")

    if ask_clicked:
        query = query.strip()
        if query:
            try:
                with st.spinner("🤔 Preparing an answer…"):
                    if any(word in query.lower() for word in SUMMARY_WORDS):
                        answer = st.session_state.summary_chain.invoke(
                            {
                                "context": st.session_state.full_transcript,
                                "question": query,
                            }
                        )
                    else:
                        answer = st.session_state.normal_chain.invoke(query)
                st.markdown(answer)
            except Exception as error:
                st.error(f"❌ Could not answer the question: {error}")
        else:
            st.warning("⚠️ Enter a question.")