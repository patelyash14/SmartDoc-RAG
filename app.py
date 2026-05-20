import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("📚 Chat with your PDF (RAG)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Embeddings (HuggingFace API)
    embedding_model = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # LLM
    llm = ChatMistralAI(model="mistral-small-2506")

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Use only the provided context to answer the question.

If answer not found, say:
"I could not find the answer in the document."
"""),
        ("human", """Context: {context}
Question: {question}
""")
    ])

    # Chat UI
    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your question")

    if query:
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.write("### 🤖 Answer:")
        st.write(response.content)