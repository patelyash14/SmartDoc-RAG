from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings



load_dotenv()

embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory= "chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 2,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)


llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , """You are a helpful AI assistant.
         Use only the provided context to answer the question.

         If the answer is not present int the context,
         say: "I could not find the answer in the document."
        """),

        ("human", """Context: {context}
         Question: {question}
        """)
    ]
)

print("RAG system created ")

print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)
    print(f"\n AI : {response.content}")