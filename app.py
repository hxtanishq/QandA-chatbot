## RAG Q&A CONVERSATION WITH PDF INCLUDING CHAT HISTORY


import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


st.title("RAG converastion with PDF upload and chat history")
st.write("Upload your PDF File Here")
api_key = st.text_input("Enter your GROQ API key: ",type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name = "Gemma2-9b-It")
    session_id = st.text_input("Session ID:", value="default_session")
    

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files =st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False) 

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f'./temp.pdf'
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file)
                # file_name = uploaded_file.name
            if os.path.exists(temp_pdf) and os.path.getsize(temp_pdf) > 0:
                try:
                    loader = PyPDFLoader(temp_pdf)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading PDF: {e}")
                else:
                    print("File write error: The file was not written correctly.")
            # loader = PyPDFLoader(temp_pdf)
            # docs = loader.load()
            # documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever() 
                
        context_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood" 
            "without the chat history. Do NOT answer the question "
            "Just reformulate it if needed and otherwise return it as is"
        )
        
        context_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",context_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,context_prompt)
        
        system_prompt =(
            "You are an assistant for a question-answering task. "
            " Use the following pieces of retrieved context to answer "
            " the question. If you don't know the answer, say that you" 
            "  don't know. Use three sentences maximum and keep" 
            " the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
                
            ]
        )
        
        
        que_ans_chain =create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,que_ans_chain)
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                
            return st.session_state.store[session_id]
        
        conv_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" 
        )
        
        user_input = st.text_input("enter the question")    
        if user_input:
            session_history =get_session_history(session_id)
            response =conv_rag_chain.invo(
                {"input":user_input},
                config ={
                    "configurable":{"session_id":session_id}
                },
            )

            st.write(st.session_state.store)
            st.success("Assistant: ",response["answer"])
            st.write("Chat history:",session_history.messages)
            
    
    
else:
    st.warning("Please enter your GROQ API key")
