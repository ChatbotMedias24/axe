import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message  # Importez la fonction message
import toml
import docx2txt
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []
st.markdown(
    """
    <style>

        .user-message {
            text-align: left;
            background-color: #E8F0FF;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: 10px;
            margin-right: -40px;
            color:black;
        }

        .assistant-message {
            text-align: left;
            background-color: #F0F0F0;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: -10px;
            margin-right: 10px;
            color:black;
        }

        .message-container {
            display: flex;
            align-items: center;
        }

        .message-avatar {
            font-size: 25px;
            margin-right: 20px;
            flex-shrink: 0; /* Empêcher l'avatar de rétrécir */
            display: inline-block;
            vertical-align: middle;
        }

        .message-content {
            flex-grow: 1; /* Permettre au message de prendre tout l'espace disponible */
            display: inline-block; /* Ajout de cette propriété */
}
        .message-container.user {
            justify-content: flex-end; /* Aligner à gauche pour l'utilisateur */
        }

        .message-container.assistant {
            justify-content: flex-start; /* Aligner à droite pour l'assistant */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    #logoo_path = "axee.png"
    st.sidebar.image(logo_path,width=150)
    #st.sidebar.image(logoo_path,width=150)
    
st.sidebar.subheader("Suggestions:")
questions = [
        "Donnez-moi un résumé du rapport ",
        "Quels sont les thèmes ou sujets phares abordés par le rapport ?",
        "Comment le rapport évalue-t-il l'efficacité des politiques et des programmes en place ?",
        "Quels défis et opportunités le rapport identifie-t-il pour le pays dans les années à venir ?",
        "Quelles recommandations globales le rapport propose-t-il pour améliorer la gouvernance et la gestion des ressources publiques ?"
    ]    
 
load_dotenv(st.secrets["OPENAI_API_KEY"])
conversation_history = StreamlitChatMessageHistory()

def main():
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique
    st.header("Rapport annuel de la Cour des comptes au titre de 2022-2023 💬")
    # upload a PDF file
    docx = 'axe.docx'
 
    # st.write(pdf)
    if docx is not None:
        text = docx2txt.process(docx)
         # Get the first page as an image
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        # st.write(chunks)
 
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open("aaa.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        st.markdown('<p style="margin-bottom: 0;"><h7><b>Posez vos questions ci-dessous:</b></h7></p>', unsafe_allow_html=True)
        query_input = st.text_input("")
        selected_questions = st.sidebar.radio("****Choisir :****",questions)
        if query_input and query_input not in st.session_state.previous_question:
            query = query_input
            st.session_state.previous_question.append(query_input)
        elif selected_questions:
            query = selected_questions
        else:
            query=""
    
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                if "Donnez-moi un résumé du rapport" in query:
                    response = "Le rapport annuel 2022-2023 de la Cour des comptes du Maroc met en lumière plusieurs aspects clés. Il aborde la gestion des événements majeurs comme le séisme d'Al Haouz et la préparation pour la Coupe du monde de football 2030. Le rapport souligne l'importance des réformes structurelles, en particulier dans les secteurs de l'infrastructure, de la santé, de l'éducation, et de la culture. Il met également en avant les défis économiques auxquels le Maroc fait face, exacerbés par la pandémie de Covid-19 et la crise russo-ukrainienne, tout en montrant une résilience dans les finances publiques et une amélioration des recettes budgétaires. En outre, il se concentre sur la nécessité de gérer efficacement les infrastructures et services publics, tout en soulignant l'importance de la transparence et de la bonne gouvernance dans la gestion des affaires publiques."

                conversation_history.add_user_message(query)
                conversation_history.add_ai_message(response)  # Utilisez add_ai_message
        
            formatted_messages = []

            for msg in conversation_history.messages:
                role = "user" if msg.type == "human" else "assistant"
                avatar = "🧑" if role == "user" else "🤖"
                css_class = "user-message" if role == "user" else "assistant-message"
        
                message_div = f'<div class="{css_class}">{msg.content}</div>'
                avatar_div = f'<div class="avatar">{avatar}</div>'
        
                if role == "user":
                    formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                else:
                    formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
        
                formatted_messages.append(formatted_message)

            messages_html = "\n".join(formatted_messages)
            st.markdown(messages_html, unsafe_allow_html=True)

            # Affichage des avatars à l'extérieur de la div des messages
            

if __name__ == '__main__':
    main()
