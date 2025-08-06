# app.py

import streamlit as st
import torch
import os
from typing import Optional, List, Tuple

# Check if we're in a resource-constrained environment
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD") or "streamlit" in os.getcwd().lower()

st.set_page_config(page_title="Claude/Gemini Clone", layout="wide")
st.title("💬 Claude/Gemini Clone with LangChain")

# Set HF token securely
HF_TOKEN = st.secrets.get("HF_TOKEN") if hasattr(st, 'secrets') else os.getenv("HF_TOKEN")

# SOLUTION 1: Use OpenAI-like API approach with Hugging Face
@st.cache_resource
def load_hf_inference_llm():
    """Use Hugging Face Inference API with proper client"""
    try:
        from huggingface_hub import InferenceClient
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        
        class HuggingFaceInferenceLLM(LLM):
            client: InferenceClient = None
            model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
            max_new_tokens: int = 512
            temperature: float = 0.7
            
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.client = InferenceClient(model=self.model_id, token=HF_TOKEN)
            
            @property
            def _llm_type(self) -> str:
                return "huggingface_inference"
            
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs
            ) -> str:
                try:
                    response = self.client.text_generation(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        stop_sequences=stop or [],
                        return_full_text=False
                    )
                    return response
                except Exception as e:
                    return f"Error generating response: {str(e)}"
        
        return HuggingFaceInferenceLLM()
    except Exception as e:
        st.error(f"Error setting up HF Inference: {str(e)}")
        return None

# SOLUTION 2: Use a local lightweight model with better compatibility
@st.cache_resource
def load_local_llm():
    """Load a small local model that works reliably"""
    try:
        from transformers import pipeline
        from langchain_community.llms import HuggingFacePipeline
        
        # Use a very small model for Streamlit Cloud
        model_name = "microsoft/DialoGPT-small" if IS_STREAMLIT_CLOUD else "gpt2"
        
        pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256,  # GPT-2 EOS token
            device=-1,  # Force CPU
            torch_dtype=torch.float32
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading local model: {str(e)}")
        return None

# SOLUTION 3: Simple fallback using OpenAI (if user has key)
@st.cache_resource
def load_openai_llm():
    """Fallback to OpenAI if available"""
    try:
        from langchain_openai import ChatOpenAI
        openai_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, 'secrets') else os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=openai_key
            )
        return None
    except Exception as e:
        st.warning(f"OpenAI not available: {str(e)}")
        return None

# SOLUTION 4: Mock LLM for testing
class MockLLM:
    """Simple mock LLM for testing when other options fail"""
    
    def __call__(self, inputs):
        if isinstance(inputs, dict) and "question" in inputs:
            question = inputs["question"]
        elif isinstance(inputs, str):
            question = inputs
        else:
            question = str(inputs)
        
        # Simple rule-based responses for testing
        responses = {
            "hello": "Hello! I'm a mock AI assistant. How can I help you today?",
            "how are you": "I'm doing well, thank you for asking! I'm here to help with your questions.",
            "what is": f"You asked about '{question}'. This is a mock response since the main model isn't available.",
            "default": f"I understand you're asking: '{question}'. This is a mock response for testing purposes. The actual model may not be loaded properly."
        }
        
        # Simple keyword matching
        question_lower = question.lower()
        for key, response in responses.items():
            if key in question_lower:
                return response
        
        return responses["default"]

# Load embeddings and vectorstore
@st.cache_resource
def load_vectorstore():
    """Load or create vectorstore"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.docstore.document import Document
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Try to load existing FAISS index
        try:
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except:
            # Create a sample vectorstore for testing
            sample_docs = [
                Document(page_content="This is a sample document about artificial intelligence and machine learning.", 
                        metadata={"source": "sample1"}),
                Document(page_content="Natural language processing is a subfield of AI that focuses on language understanding.", 
                        metadata={"source": "sample2"}),
                Document(page_content="Large language models like GPT and BERT have revolutionized NLP tasks.", 
                        metadata={"source": "sample3"}),
            ]
            vectorstore = FAISS.from_documents(sample_docs, embeddings)
            return vectorstore
    except Exception as e:
        st.error(f"Error with vectorstore: {str(e)}")
        return None

# Simple QA Chain without complex dependencies
class SimpleQAChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chat_history = []
    
    def __call__(self, inputs):
        question = inputs["question"]
        
        # Get relevant documents
        try:
            if self.retriever:
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs[:2]])
            else:
                context = "No context available."
        except:
            context = "Error retrieving context."
        
        # Format prompt
        if hasattr(self.llm, '_call') or hasattr(self.llm, 'predict'):
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            try:
                if hasattr(self.llm, 'predict'):
                    answer = self.llm.predict(prompt)
                else:
                    answer = self.llm._call(prompt)
            except:
                answer = self.llm({"question": question})
        else:
            # For mock LLM
            answer = self.llm(question)
        
        # Store in history
        self.chat_history.append((question, answer))
        
        return {"answer": answer, "source_documents": []}

# UI for model selection
st.sidebar.title("🔧 Configuration")

model_options = {
    "Hugging Face API": "hf_api",
    "Local Small Model": "local",
    "OpenAI (if available)": "openai",
    "Mock LLM (Testing)": "mock"
}

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    list(model_options.keys())
)

model_type = model_options[selected_model]

# Load selected model
with st.spinner(f"Loading {selected_model}..."):
    if model_type == "hf_api" and HF_TOKEN:
        llm = load_hf_inference_llm()
    elif model_type == "local":
        llm = load_local_llm()
    elif model_type == "openai":
        llm = load_openai_llm()
    else:
        llm = MockLLM()
        st.sidebar.warning("Using Mock LLM for testing")

# Load vectorstore
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever() if vectorstore else None

# Create QA chain
qa_chain = SimpleQAChain(llm, retriever)

# Display status
if HF_TOKEN:
    st.sidebar.success("✅ HF Token configured")
else:
    st.sidebar.warning("⚠️ HF Token missing - using fallback options")

st.sidebar.info(f"Model: {selected_model}")
st.sidebar.info(f"Retriever: {'✅ Active' if retriever else '❌ Inactive'}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"question": prompt})
                response = result["answer"]
                
                # Clean up response
                if len(response) > 1000:
                    response = response[:1000] + "..."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Help section
with st.sidebar.expander("ℹ️ Help"):
    st.write("""
    **Model Options:**
    - **HF API**: Uses Hugging Face API (requires token)
    - **Local**: Small model running locally
    - **OpenAI**: Uses OpenAI API (requires key)
    - **Mock**: Simple test responses
    
    **Troubleshooting:**
    - Add HF_TOKEN to Streamlit secrets
    - For OpenAI, add OPENAI_API_KEY
    - Use Mock LLM for testing without API keys
    """)
