#libraries for deployment and db
import os
import time
from dotenv import load_dotenv
import pinecone 
import streamlit as st
from pinecone import ServerlessSpec
from langchain.vectorstores import Pinecone

#core libraries for project
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
#from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

#loading environment variables using load_dotenv()
#uncomment the below line if you are running on a local machine and update .env with your API keys
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#configuring the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
#initializing pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

#defining our llm and embedding model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

#defining index name
index_name = 'nlqa-rag'


#functions to handle data -> text from pdf -> chunking -> vectorizing and indexing

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    #delete the index from previous session
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # we create a new index
    pc.create_index(
            index_name,
            dimension=768,  # dimensionality of text-embedding-models/embedding-001
            metric='cosine',
            spec = ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    #upsert the data 
    Pinecone.from_texts(text_chunks, embedding=embeddings, index_name=index_name)

    return 



def get_doc_chain():

    prompt_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, ask for more "context" but don't answer questions that are not relevant to the context. 
    Just say "I don't know." 
    Use three sentences maximum and keep the answer concise.

    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    return doc_chain

def get_retriver():
    #Creating a Vector Store from index
    index = pc.Index(index_name)
    vectorstore = Pinecone(index, embeddings, "text")
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    #reframing the query using llm for retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question 
                which might reference context in the chat history, formulate a standalone question 
                which can be understood without the chat history. Do NOT answer the question, 
                just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    #creating history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm= llm,
        retriever = retriever,
        prompt= contextualize_q_prompt
    ) 

    return history_aware_retriever


def user_input(user_question, conversation_history=None,):
    
    if conversation_history is None:
        conversation_history = []
    
    retriever = get_retriver()
    doc_chain = get_doc_chain()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)


    response = retrieval_chain.invoke(
        {"input": user_question,
         "chat_history" : conversation_history })

    return response["answer"]



def main():
    #page layout
    st.set_page_config("ReviewRush")
    st.header("ReviewRush")
    user_question = st.chat_input("Ask a Question from the PDF Files")
    flag = 0
    #UI for uploading pdf files
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("You can upload multiple PDFs", type = "pdf",accept_multiple_files=True)
        if st.button("Submit & Process"):
            if len(pdf_docs)!=0:
                flag = 1
                with st.spinner("Processing..."):
                    #processing the uploaded pdf files
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("No file selected")


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []   

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question:
        #check if there is any file uploaded
        if len(pdf_docs)!=0:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_question)

            #getting the response on user query
            response = user_input(user_question, st.session_state.conversation_history)

            #Display response
            with st.chat_message('assistant'):
                st.markdown(response)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            st.session_state.conversation_history.append(HumanMessage(content=user_question))   
            st.session_state.conversation_history.append(AIMessage(content=response))
        else:
            st.warning("Please Upload the files first. This will help me to understand you better:)")



if __name__ == "__main__":
    main()
