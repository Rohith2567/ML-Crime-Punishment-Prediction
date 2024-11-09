import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

st.title("CRIME PREDICTION AND PUNISHMENT MODEL ⚖️🧑🏻‍⚖️👩🏻‍⚖️")
st.link_button("Reference to IPC", "https://www.indiacode.nic.in/repealedfileopen?rfilename=A1860-45.pdf")

loader = PyPDFLoader("IPC.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# store_path = os.path.join("C:", "Users", "rohit", "OneDrive", "Desktop", "Crime Prediction", "Ch_store")
store_path = os.path.join("Cache")

if os.path.exists(store_path):
    vectorstore = Chroma(
        persist_directory=store_path,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=store_path
    )

# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
#     persist_directory=store_path
# )


@st.cache_resource
def load_vectorstore():
    return vectorstore


retriever = load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=None, timeout=60)

st.write("Let’s Uncover the Legal Outcome!")
query = st.chat_input("Describe the Incident...")
# query = st.text_input("Describe the scene")
# prompt = query

system_prompt = (
    "You are a legal expert specializing in Indian criminal law."
    "Use the following pieces of retrieved context to answer the question"
    "You have to identify the crime type, IPC sections, and legal outcome based on the description given"
    "to you along with court references if that particular crime has one."
    "If the given description is not a crime mention that it is not a crime and why it is not a crime"
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# query = input("Enter: ")

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    st.header("Response: ")
    st.write(response["answer"])
    # print(response["answer"])
