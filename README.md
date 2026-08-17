## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
Develop a PDF-based Question-Answering Chatbot using LangChain that can extract text from a PDF document, convert it into vector embeddings, store the embeddings in a vector database, and retrieve relevant information to answer user queries accurately using a Large Language Model (LLM). The chatbot should also maintain conversation history to provide context-aware responses                                                                      
### DESIGN STEPS:

#### STEP 1: Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: :Initialize the LangChain QA Pipeline 
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

### PROGRAM: 
 import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()

# Load PDF
loader = PyPDFLoader("AI_Notes.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create Vector Database
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Create QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=db.as_retriever()
)

# Ask Questions
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = qa.run(question)
    print("Bot:", answer)

### OUTPUT:
<img width="1788" height="677" alt="image" src="https://github.com/user-attachments/assets/5bb81100-92da-43ae-a652-a0fefaf0cfc5" />


### RESULT:
The LangChain-based PDF question-answering chatbot was successfully designed and implemented. It extracted information from the PDF and provided relevant and accurate answers to different user queries. The chatbot showed good performance in terms of accuracy, relevance, and completeness of responses.
