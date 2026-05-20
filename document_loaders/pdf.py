from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter

data = PyPDFLoader("document_loaders/deep.pdf")

docs = data.load()
print(len(docs))
