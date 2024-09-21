import os
from dotenv import load_dotenv
import tempfile
import whisper
from pytube import YouTube
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# This is the YouTube video we are using and questions are to be asked based on this video.
YOUTUBE_VIDEO = "https://youtu.be/BBpAmxU_NQo?si=BA_ZktU_lMrgDfcA" #Basic DSA-Mosh

# Do this only if you haven't created the transcription file yet.
if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    # Load the base model.
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as file:
            file.write(transcription)

loader = TextLoader("transcription.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20) # Adjust as needed
documents = text_splitter.split_documents(text_documents)

index_name = "youtube-index"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

chain.invoke("How to implement and use arrays?")