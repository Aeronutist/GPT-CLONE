import streamlit as st from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM from langchain_community.vectorstores import FAISS from langchain_community.embeddings import HuggingFaceEmbeddings from langchain.chains import ConversationalRetrievalChain from langchain.memory import ConversationBufferMemory from langchain_community.llms import HuggingFacePipeline import torch import os

st.set_page_config(page_title="Claude/Gemini Clone", layout="wide") st.title("💬 Claude/Gemini Clone with Mistral + LangChain")

Set HF token securely

HF_TOKEN = st.secrets.get("HF_TOKEN")

Load model

@st.cache_resource def load_llm(): model_id = "mistralai/Mistral-7B-Instruct-v0.2" tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN) model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN) pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512) return HuggingFacePipeline(pipeline=pipe)

Load FAISS

@st.cache_resource def load_vectorstore(): embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = load_llm() retriever = load_vectorstore().as_retriever() memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm( llm=llm, retriever=retriever, memory=memory )

Chat interface

if "chat_history" not in st.session_state: st.session_state.chat_history = []

question = st.chat_input("Ask anything about the document...")

if question: with st.spinner("Thinking..."): result = qa_chain({"question": question}) answer = result["answer"] st.session_state.chat_history.append((question, answer))

for q, a in st.session_state.chat_history: st.chat_message("user").markdown(f"You: {q}") st.chat_message("assistant").markdown(f"AI: {a}")

